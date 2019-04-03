from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import re 
import os



##
config = {
    "run_offset": 0,
    "num_runs": 5,

    "architecture": "meta", # meta or conditioned
    "num_task_hidden_layers": 3,
    "num_hyper_hidden_layers": 3,
    "num_language_layers": 2,
    "num_hidden": 64,
    "num_hidden_hyper": 512, 
    "convolution_specs": [ # (kernel_size, depth, stride, pool, pool_stride)
        (5, 10, 2, 3, 3),
        (2, 32, 2, 1, 1)
    ], 
    "task_weight_weight_mult": 1., 

    "init_learning_rate": 1e-4,
    "lr_decay": 0.85,
    "lr_decays_every": 100,
    "min_learning_rate": 3e-8,

    "internal_nonlinearity": tf.nn.leaky_relu,

    "vocab": ["PAD", "EOS", "is", "triangle", "circle", "square", "above", "below", "right_of", "left_of", "red", "green", "blue"],

    "image_size": 30,
    "data_path": "shapes_data/",
    "results_path": "/mnt/fs2/lampinen/meta_shapes/results/"
}


### load data

data_path = config["data_path"]
data = {}
max_query_len = 0

vocab = config["vocab"] 
vocab_to_ints = dict(zip(vocab, range(len(vocab))))

def process_query_data(query_data, max_query_len):
    query_data = [re.sub('[()]', '', x).split() for x in query_data]
    query_data = [["PAD"] * (max_query_len - len(x)) + x + ["EOS"] for x in query_data]
    query_data = [[vocab_to_ints[w] for w in x] for x in query_data]
    return np.array(query_data, dtype=np.int32)
    

for dataset in ["train.large", "val", "test"]:
    data[dataset] = {}
    data[dataset]["input"] = np.load(data_path + dataset + ".input.npy")
    data[dataset]["output"] = 1*(np.loadtxt(data_path + dataset + ".output", dtype=np.str) == 'true')
    data[dataset]["query_raw"] = np.loadtxt(data_path + dataset + ".query", dtype=np.str, delimiter=",")
    max_query_len = max(max_query_len, max([len(x.split()) for x in data[dataset]["query_raw"]]))

for dataset in ["train.large", "val", "test"]:
    data[dataset]["query"] = process_query_data(data[dataset]["query_raw"], max_query_len) 

config["max_seq_len"] = max_query_len + 1 # + 1 for EOS


### model 

class shape_model(object):
    def __init__(self, config):
        num_hidden = config["num_hidden"]
        num_hidden_hyper = config["num_hidden_hyper"]
        num_task_hidden_layers = config["num_task_hidden_layers"]
        num_language_layers = config["num_language_layers"]
        num_hyper_hidden_layers = config["num_hyper_hidden_layers"]
        internal_nonlinearity = config["internal_nonlinearity"]
        image_size = config["image_size"]
        max_seq_len = config["max_seq_len"]
        architecture_is_meta = config["architecture"] == "meta"
        vocab_size = len(config["vocab"])

        self.visual_input_ph = tf.placeholder(
            tf.float32, shape=[None, image_size, image_size, 3]) 
        self.query_ph = tf.placeholder(
            tf.int32, shape=[1, max_seq_len])
        self.target_ph = tf.placeholder(tf.int32, shape=[None,])

        # query processing
        with tf.variable_scope('query') as scope:
            self.word_embeddings = tf.get_variable(
                "embeddings", shape=[vocab_size, num_hidden_hyper])
            self.embedded_language = tf.nn.embedding_lookup(self.word_embeddings,
                                                            self.query_ph)

            with tf.variable_scope('lstm') as lstm_scope:
                cells = [tf.nn.rnn_cell.LSTMCell(
                    num_hidden_hyper,
                    activation=internal_nonlinearity) for _ in range(num_language_layers)]

                stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(cells)

                state = stacked_lstm.zero_state(1, tf.float32)
                for t in range(max_seq_len):
                    this_output, state = stacked_lstm(self.embedded_language[:, t, :], state)
                    lstm_scope.reuse_variables()

            self.query_embedding = slim.fully_connected(this_output, num_hidden_hyper,
                                                        activation_fn=None)

        # input processing
        with tf.variable_scope('vision'):
            vision_hidden = self.visual_input_ph 
#            print(vision_hidden.get_shape())
            for kernel_size, depth, stride, pool, pool_stride in config["convolution_specs"]:
                vision_hidden = slim.convolution2d(
                    vision_hidden, depth, kernel_size, stride, 
                    activation_fn=internal_nonlinearity, padding="SAME")
#                print(vision_hidden.get_shape())
                if pool > 1:
                    vision_hidden = slim.max_pool2d(vision_hidden, pool, pool_stride)
#                    print(vision_hidden.get_shape())


            vision_hidden = slim.flatten(vision_hidden)
            vision_hidden = slim.fully_connected(vision_hidden, num_hidden_hyper,
                                                 activation_fn=internal_nonlinearity)
            self.processed_input = slim.fully_connected(vision_hidden, num_hidden_hyper,
                                                        activation_fn=None)
#            print(self.processed_input.get_shape())

        if architecture_is_meta:

            tw_range = config["task_weight_weight_mult"]/np.sqrt(
                num_hidden * num_hidden_hyper) # yields a very very roughly O(1) map
            task_weight_gen_init = tf.random_uniform_initializer(-tw_range,
                                                                 tw_range)

            def _hyper_network(function_embedding, reuse=True):
                with tf.variable_scope('hyper', reuse=reuse):
                    hyper_hidden = function_embedding
                    for _ in range(config["num_hyper_hidden_layers"]):
                        hyper_hidden = slim.fully_connected(hyper_hidden, num_hidden_hyper,
                                                            activation_fn=internal_nonlinearity)
#                        hyper_hidden = tf.nn.dropout(hyper_hidden, self.keep_prob_ph)

                    hidden_weights = []
                    hidden_biases = []

                    task_weights = slim.fully_connected(hyper_hidden, num_hidden*(num_hidden_hyper +(num_task_hidden_layers-1)*num_hidden + num_hidden_hyper),
                                                        activation_fn=None,
                                                        weights_initializer=task_weight_gen_init)
#                    task_weights = tf.nn.dropout(task_weights, self.keep_prob_ph)

                    task_weights = tf.reshape(task_weights, [-1, num_hidden, (num_hidden_hyper + (num_task_hidden_layers-1)*num_hidden + num_hidden_hyper)])
                    task_biases = slim.fully_connected(hyper_hidden, num_task_hidden_layers * num_hidden + num_hidden_hyper,
                                                       activation_fn=None)

                    Wi = tf.transpose(task_weights[:, :, :num_hidden_hyper], perm=[0, 2, 1])
                    bi = task_biases[:, :num_hidden]
                    hidden_weights.append(Wi)
                    hidden_biases.append(bi)
                    for i in range(1, num_task_hidden_layers):
                        Wi = tf.transpose(task_weights[:, :, num_hidden_hyper+(i-1)*num_hidden:num_hidden_hyper+i*num_hidden], perm=[0, 2, 1])
                        bi = task_biases[:, num_hidden*i:num_hidden*(i+1)]
                        hidden_weights.append(Wi)
                        hidden_biases.append(bi)
                    Wfinal = task_weights[:, :, -num_hidden_hyper:]
                    bfinal = task_biases[:, -num_hidden_hyper:]

                    for i in range(num_task_hidden_layers):
                        hidden_weights[i] = tf.squeeze(hidden_weights[i], axis=0)
                        hidden_biases[i] = tf.squeeze(hidden_biases[i], axis=0)

                    Wfinal = tf.squeeze(Wfinal, axis=0)
                    bfinal = tf.squeeze(bfinal, axis=0)
                    hidden_weights.append(Wfinal)
                    hidden_biases.append(bfinal)
                    return hidden_weights, hidden_biases

            self.task_params = _hyper_network(self.query_embedding, reuse=False)

            # task network
            def _task_network(task_params, processed_input):
                hweights, hbiases = task_params
                task_hidden = processed_input
                for i in range(num_task_hidden_layers):
                    task_hidden = internal_nonlinearity(
                        tf.matmul(task_hidden, hweights[i]) + hbiases[i])

                raw_output = tf.matmul(task_hidden, hweights[-1]) + hbiases[-1]

                return raw_output


            self.base_raw_output = _task_network(self.task_params,
                                                 self.processed_input)

        else:
            raise NotImplementedError("Not yet implemented!")


        with tf.variable_scope('output'):
            self.base_output_logits = slim.fully_connected(self.base_raw_output,
                                                           2,
                                                           activation_fn=None)

        self.base_output = tf.nn.softmax(self.base_output_logits)

        self.processed_labels = tf.one_hot(self.target_ph, depth=2)

        self.item_losses = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.base_output_logits,
            labels=self.processed_labels)

        self.total_loss = tf.reduce_mean(self.item_losses)

        corr_scores = tf.reduce_sum(tf.multiply(self.base_output_logits, self.processed_labels), axis=-1) 
        incorr_scores = tf.reduce_sum(tf.multiply(self.base_output_logits, 1.-self.processed_labels), axis=-1) 
        self.item_scores = tf.cast(tf.greater(corr_scores, incorr_scores), tf.float32)
        self.pct_correct = tf.reduce_mean(self.item_scores)

        self.lr_ph = tf.placeholder(tf.float32)
        optimizer = tf.train.RMSPropOptimizer(self.lr_ph)

        self.train = optimizer.minimize(self.total_loss)

        # Saver
        self.saver = tf.train.Saver()

        # initialize
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())

shape_model(config)
