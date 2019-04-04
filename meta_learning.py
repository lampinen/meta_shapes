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
    "num_hidden": 32,
    "num_hidden_hyper": 256, 
    "convolution_specs": [ # (kernel_size, depth, stride, pool, pool_stride)
        (6, 10, 2, 1, 1),
        (3, 32, 1, 3, 3),
        (2, 32, 2, 1, 1)
    ], 
    "task_weight_weight_mult": 1., 

    "init_learning_rate": 1e-4,
    "lr_decay": 0.85,
    "lr_decays_every": 50,
    "min_lr": 1e-6,

    "train_keep_prob": 1., # dropout on language network
#    "train_batch_subset": 64, # DEACTIVATED -- how much of train batch to take at a time -- further stochasticity
    "l2_penalty_weight": 1e-4,

    "train_with_meta": True, # if meta, whether to also train with meta->hyper
    "meta_batch_size": 32,
    "meta_embedding_scale": 1e-3, # try to match initial scale of embeddings 
                                  # from language and meta

    "num_epochs": 5000,
    "eval_every": 10,

    "internal_nonlinearity": tf.nn.leaky_relu,

    "vocab": ["PAD", "EOS", "is", "triangle", "circle", "square", "above", "below", "right_of", "left_of", "red", "green", "blue"],

    "image_size": 30,
    "data_path": "shapes_data/",
    "results_path": "/mnt/fs2/lampinen/meta_shapes/results_with_meta/"
}

def _save_config(filename, config):
    with open(filename, "w") as fout:
        fout.write("key, value\n")
        for key, value in config.items():
            fout.write(key + ", " + str(value) + "\n")

# from GloVe 6B 50d
initial_embeddings = np.array([
        [0.] * 50, # PAD
        [0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973, -0.43478, -0.31086, -0.44999, -0.29486, 0.16608, 0.11963, -0.41328, -0.42353, 0.59868, 0.28825, -0.11547, -0.041848, -0.67989, -0.25063, 0.18472, 0.086876, 0.46582, 0.015035, 0.043474, -1.4671, -0.30384, -0.023441, 0.30589, -0.21785, 3.746, 0.0042284, -0.18436, -0.46209, 0.098329, -0.11907, 0.23919, 0.1161, 0.41705, 0.056763, -6.3681e-05, 0.068987, 0.087939, -0.10285, -0.13931, 0.22314, -0.080803, -0.35652, 0.016413, 0.10216], #, EOS, (.)
        [0.6185, 0.64254, -0.46552, 0.3757, 0.74838, 0.53739, 0.0022239, -0.60577, 0.26408, 0.11703, 0.43722, 0.20092, -0.057859, -0.34589, 0.21664, 0.58573, 0.53919, 0.6949, -0.15618, 0.05583, -0.60515, -0.28997, -0.025594, 0.55593, 0.25356, -1.9612, -0.51381, 0.69096, 0.066246, -0.054224, 3.7871, -0.77403, -0.12689, -0.51465, 0.066705, -0.32933, 0.13483, 0.19049, 0.13812, -0.21503, -0.016573, 0.312, -0.33189, -0.026001, -0.38203, 0.19403, -0.12466, -0.27557, 0.30899, 0.48497], #, "is"
        [0.26777, 0.39051, -0.80204, 0.87605, 0.56361, 0.62825, 1.0976, -0.88538, 0.36826, -0.18403, -0.10373, -0.52103, -0.60596, 0.22897, -0.79127, 0.22233, 0.63658, 0.54065, -0.55847, 0.33736, -0.27757, -0.55417, -0.038542, 0.59959, -1.0194, -0.75978, -0.52453, 0.93808, 0.37005, -1.1082, 1.4721, -0.43972, 0.037059, -0.051395, -0.17195, -0.16822, -0.70491, -0.40332, -0.40064, 0.56881, -0.15365, 0.17935, -0.12581, -0.012767, -0.18841, -0.069272, 0.90065, -0.79688, -0.52345, -0.36574], #, "triangle"
        [0.031194, 1.3965, 0.099333, -0.28257, 0.698, 0.060701, -0.063591, -0.13732, -0.056184, -0.51164, -0.44842, -0.29044, -0.58677, 0.96193, -0.87526, 0.041329, 0.70571, -0.36101, -0.6187, -0.34575, 0.33935, -0.1803, -0.019442, 0.37123, -0.14978, -0.99516, -0.80352, 0.48491, -0.019706, -0.69561, 1.9556, 0.20836, 0.079642, -0.15071, -0.61761, -0.23789, -0.57757, 0.24111, -0.17408, 0.41456, 0.34125, 0.15863, -0.41312, 0.34164, -0.82359, -0.44607, 0.46473, -0.90231, -0.53438, -0.57996], #, "circle"
        [0.58757, 0.88152, 0.62398, -0.36961, 1.2049, -0.33949, -0.099526, -1.4267, -0.50716, -0.5901, -0.89797, -2.1013, 0.10816, -1.0414, 0.29457, 0.35242, 0.76798, 0.68436, -1.2588, -0.37382, 0.76932, 0.050303, -0.70036, -0.26569, -0.42484, -0.082767, 0.023186, 0.96022, -0.20701, -0.65148, 2.5441, 0.084749, 0.15318, 0.15437, -0.92624, 0.06355, 0.7533, 0.10222, 0.054847, 0.62167, 0.40819, 0.3641, -0.46259, -0.076742, -0.94524, -0.1997, -0.44044, -1.4929, -0.58814, -0.76186], #, "square"
        [0.20461, 1.1844, 0.88444, -1.0012, 0.79837, -0.56763, 0.28192, -0.65914, -0.15603, -0.39741, -0.13706, -0.56106, 0.14916, -0.23118, 0.032053, 0.41856, 0.093921, -0.21944, -1.175, -1.0742, -0.27582, -0.32302, 0.26617, -0.30197, 0.06571, -0.75212, -0.049711, 1.3707, -0.023043, 0.11972, 3.4593, -0.15812, 0.92542, 0.070879, 0.11557, -0.472, 0.28877, -0.30514, 0.25907, -0.21814, 0.03788, -0.063147, 0.60011, 0.72842, -1.035, -0.26292, 0.79245, -0.43415, -0.15939, -0.73824], #, "above"
        [-0.1761, 1.0209, 1.1094, -0.94708, 0.5056, -0.22629, -0.11544, -0.73112, -0.017969, -0.4826, 0.065433, -0.5686, 0.25488, -0.10177, 0.48094, 0.59728, -0.075237, -0.55092, -0.94344, -0.70571, 0.048117, -0.53517, 0.76428, -0.26941, 0.043954, -0.3497, -0.31752, 1.1031, -0.11156, 0.41239, 3.4776, -0.18738, 1.1889, 0.067102, 0.37153, -0.40691, 0.3275, -0.29483, 0.55618, -0.48239, -0.44232, -0.27762, 0.79069, 0.42688, -1.0616, -0.35802, 0.976, -0.25116, 0.17372, -0.68949], #, "below"
        [-0.31905, -0.09507, -0.049458, -0.45461, 0.81447, 0.5573, -0.37148, 0.22142, -0.18506, -0.058731, 0.19556, 0.25934, -0.92544, 0.60346, -0.11877, 0.33422, 0.5861, -0.93036, 0.057848, -0.3782, -1.1279, 0.11801, -0.26057, 0.043562, 0.25741, -2.2502, 0.14761, 0.43313, 0.6941, -0.94411, 3.5944, 0.47921, -0.61145, 0.16474, -0.68397, -0.091051, 0.39399, 0.1769, 0.33553, -0.13054, 0.12659, 0.26288, -0.47432, 0.79129, -0.75714, -0.117, -0.093022, -0.23004, 0.081364, -0.035147], #, "right_of"
        [0.46783, -0.035872, 0.41065, -0.4438, 0.58944, 0.22964, -1.1107, 0.87497, -0.72342, -0.91589, -0.17989, -0.38281, -1.0384, 0.47202, 0.23037, -0.039563, -0.079483, -0.50856, -0.72968, 0.19254, -0.5322, 0.65468, 0.25873, -0.12893, 0.21562, -1.4202, 0.63648, 0.52826, 0.28329, -0.073296, 3.2336, 0.22421, 0.031033, 0.015244, -0.28903, 0.54683, 0.01121, 0.21479, 0.96489, 0.10511, -0.20664, 0.21379, -0.45599, 0.23589, -0.27467, 0.21216, -0.097545, -0.51355, -0.11455, -0.84802], #, "left_of"
        [-0.12878, 0.8798, -0.60694, 0.12934, 0.5868, -0.038246, -1.0408, -0.52881, -0.29563, -0.72567, 0.21189, 0.17112, 0.19173, 0.36099, 0.032672, -0.2743, -0.19291, -0.10909, -1.0057, -0.93901, -1.0207, -0.69995, 0.57182, -0.45136, -1.2145, -1.1954, -0.32758, 1.4921, 0.54574, -1.0008, 2.845, 0.26479, -0.49938, 0.34366, -0.12574, 0.5905, -0.037696, -0.47175, 0.050825, -0.20362, 0.13695, 0.26686, -0.19461, -0.75482, 1.0303, -0.057467, -0.32327, -0.7712, -0.16764, -0.73835], #, "red"
        [-0.5767, 0.86953, -0.49108, -0.1078, 0.65377, 0.32548, -1.326, -1.0114, -0.20658, -0.79937, -0.41455, 0.084769, 0.25426, -0.10999, -0.64696, 0.17882, 0.68277, -0.019661, -0.59745, -1.0414, -0.55979, -0.18503, 0.49271, -0.53623, -0.63925, -0.91267, 0.15709, 1.3784, 0.45686, -0.76237, 2.5792, -0.22079, -0.59114, -0.17395, 0.27835, 0.30161, -0.12137, 0.21278, 0.23306, -0.74012, 0.12318, -0.028115, -0.421, -0.60167, 0.71802, 0.19434, -0.11682, -1.1335, 0.49009, -0.020496], #, "green"
        [-0.83751, 0.69563, -0.51408, 0.23689, 0.59192, -0.027491, -1.2076, -0.98796, -0.27658, -0.4618, 0.4715, 0.13072, 0.50393, 0.50556, -0.66766, 0.069073, -0.60984, -0.22776, -1.2481, -1.3521, -0.56051, -0.17952, 0.22886, -0.69244, -1.1734, -0.98778, -0.81551, 1.5513, 0.36517, -1.1162, 2.632, 0.21987, 0.10695, 0.28438, -0.10348, -0.29667, -0.17645, -0.75838, 0.085523, -0.83641, -0.12174, -0.063165, -0.072053, -0.30712, 0.61861, -0.30867, 0.012374, -1.1966, 0.041525, -0.23966], #, "blue"
    ], dtype=np.float32)

### load data

data_path = config["data_path"]
data = {}
max_query_len = 0

vocab = config["vocab"] 
vocab_to_ints = dict(zip(vocab, range(len(vocab))))

def process_query(query, max_query_len):
    query = re.sub('[()]', '', query).split()
    query = ["PAD"] * (max_query_len - len(query)) + query + ["EOS"]
    query = [vocab_to_ints[w] for w in query]
    return np.array([query], dtype=np.int32)
    

for dataset in ["train.large", "val", "test"]:
    data[dataset] = {}
    data[dataset]["input"] = np.load(data_path + dataset + ".input.npy")
    data[dataset]["output"] = 1*(np.loadtxt(data_path + dataset + ".output", dtype=np.str) == 'true')
    data[dataset]["query_raw"] = np.loadtxt(data_path + dataset + ".query", dtype=np.str, delimiter=",")
    max_query_len = max(max_query_len, max([len(x.split()) for x in data[dataset]["query_raw"]]))

for dataset in ["train.large", "val", "test"]:
    raw_queries = data[dataset]["query_raw"]
    raw_query_list = list(set(raw_queries))
    this_input = []
    this_output = []
    this_query = []
    for query in raw_query_list:
        index_vec = raw_queries == query
        this_query.append(process_query(query, max_query_len)) 
        this_input.append(data[dataset]["input"][index_vec])
        this_output.append(data[dataset]["output"][index_vec])

    data[dataset]["input"] = this_input
    data[dataset]["output"] = this_output 
    data[dataset]["query_raw"] = raw_query_list 
    data[dataset]["query"] = this_query 

config["max_seq_len"] = max_query_len + 1 # + 1 for EOS


### model 

class shape_model(object):
    def __init__(self, config, data):
        self.config = config
        self.data = data
        #self.train_batch_subset = config["train_batch_subset"]
        self.train_with_meta = config["train_with_meta"]
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

        self.keep_ph = tf.placeholder(tf.float32)

        # query processing
        with tf.variable_scope('query') as scope:
            self.word_embeddings = tf.get_variable(
                "embeddings", initializer=initial_embeddings)
            self.embedded_language = tf.nn.embedding_lookup(self.word_embeddings,
                                                            self.query_ph)

            with tf.variable_scope('lstm') as lstm_scope:
                cells = [tf.nn.rnn_cell.LSTMCell(
                    num_hidden_hyper,
                    activation=internal_nonlinearity) for _ in range(num_language_layers)]
                cells = [tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_ph) for cell in cells]

                stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(cells)

                state = stacked_lstm.zero_state(1, tf.float32)
                for t in range(max_seq_len):
                    this_output, state = stacked_lstm(self.embedded_language[:, t, :], state)
                    lstm_scope.reuse_variables()

            self.query_embedding = slim.fully_connected(this_output, num_hidden_hyper,
                                                        activation_fn=None)
            self.query_embedding = slim.dropout(self.query_embedding, keep_prob=self.keep_ph)

        # input processing
        with tf.variable_scope('vision'):
            vision_hidden = self.visual_input_ph 
#            print(vision_hidden.get_shape())
            for kernel_size, depth, stride, pool, pool_stride in config["convolution_specs"]:
                vision_hidden = slim.convolution2d(
                    vision_hidden, depth, kernel_size, stride, 
                    activation_fn=internal_nonlinearity, padding="SAME")
                vision_hidden = tf.nn.dropout(
                    vision_hidden, keep_prob=self.keep_ph,
                    noise_shape=tf.constant([1, 1, vision_hidden.get_shape()[3]],
                                            dtype=tf.int32))
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


        # target processing 

        self.processed_labels = tf.one_hot(self.target_ph, depth=2)
        self.target_processor = tf.get_variable("target_processor", shape=[2, num_hidden_hyper])
        embedded_targets = tf.matmul(self.processed_labels, self.target_processor)

        if architecture_is_meta:
            if self.train_with_meta:
                self.guess_input_mask_ph = tf.placeholder(tf.bool, shape=[None]) # which datapoints get excluded from the guess

                def _meta_network(embedded_inputs, embedded_targets,
                                   mask_ph=self.guess_input_mask_ph, reuse=True):
                    with tf.variable_scope('meta', reuse=reuse):
                        guess_input = tf.concat([embedded_inputs,
                                                 embedded_targets], axis=-1)
                        guess_input = tf.boolean_mask(guess_input,
                                                      self.guess_input_mask_ph)
                        guess_input = tf.nn.dropout(guess_input, self.keep_ph)

                        gh_1 = slim.fully_connected(guess_input, num_hidden_hyper,
                                                    activation_fn=internal_nonlinearity)
                        gh_1 = tf.nn.dropout(gh_1, self.keep_ph)
                        gh_2 = slim.fully_connected(gh_1, num_hidden_hyper,
                                                    activation_fn=internal_nonlinearity)
                        gh_2 = tf.nn.dropout(gh_2, self.keep_ph)
                        gh_2b = tf.reduce_max(gh_2, axis=0, keep_dims=True)
                        gh_3 = slim.fully_connected(gh_2b, num_hidden_hyper,
                                                    activation_fn=internal_nonlinearity)
                        gh_3 = tf.nn.dropout(gh_3, self.keep_ph)

                        guess_embedding = slim.fully_connected(gh_3, num_hidden_hyper,
                                                               activation_fn=None)
                        guess_embedding = tf.nn.dropout(guess_embedding, self.keep_ph)
                        return guess_embedding

                self.guess_function_emb = _meta_network(self.processed_input,
                                                        embedded_targets,
                                                        reuse=False)
                self.guess_function_emb *= config["meta_embedding_scale"]
                
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
                        hyper_hidden = tf.nn.dropout(hyper_hidden, self.keep_ph)

                    hidden_weights = []
                    hidden_biases = []

                    task_weights = slim.fully_connected(hyper_hidden, num_hidden*(num_hidden_hyper +(num_task_hidden_layers-1)*num_hidden + num_hidden_hyper),
                                                        activation_fn=None,
                                                        weights_initializer=task_weight_gen_init)

                    task_weights = tf.reshape(task_weights, [-1, num_hidden, (num_hidden_hyper + (num_task_hidden_layers-1)*num_hidden + num_hidden_hyper)])
                    task_biases = slim.fully_connected(hyper_hidden, num_task_hidden_layers * num_hidden + num_hidden_hyper,
                                                       activation_fn=None)

                    task_biases = tf.nn.dropout(task_biases, self.keep_ph)

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
                        hidden_weights[i] = tf.squeeze(tf.nn.dropout(hidden_weights[i], self.keep_ph), axis=0)
                        hidden_biases[i] = tf.squeeze(tf.nn.dropout(hidden_biases[i], self.keep_ph), axis=0)

                    Wfinal = tf.squeeze(Wfinal, axis=0)
                    bfinal = tf.squeeze(bfinal, axis=0)
                    hidden_weights.append(Wfinal)
                    hidden_biases.append(bfinal)
                    return hidden_weights, hidden_biases

            self.task_params = _hyper_network(self.query_embedding, reuse=False)
            if self.train_with_meta:
                self.meta_task_params = _hyper_network(self.guess_function_emb, reuse=True)

            # task network
            def _task_network(task_params, processed_input):
                hweights, hbiases = task_params
                task_hidden = processed_input
                for i in range(num_task_hidden_layers):
                    task_hidden = internal_nonlinearity(
                        tf.matmul(task_hidden, hweights[i]) + hbiases[i])

                raw_output = tf.matmul(task_hidden, hweights[-1]) + hbiases[-1]

                return raw_output


            self.lang_raw_output = _task_network(self.task_params,
                                                 self.processed_input)
            if self.train_with_meta:
                self.meta_raw_output = _task_network(self.meta_task_params,
                                                     self.processed_input)

        else:
            raise NotImplementedError("Not yet implemented!")

        def _output_network(raw_output, reuse=True):
            with tf.variable_scope('output', reuse=reuse):
                output_logits = slim.fully_connected(raw_output,
                                                     2,
                                                     activation_fn=None)
                return output_logits

        self.lang_output_logits = _output_network(self.lang_raw_output, 
                                                  reuse=False)
        self.lang_output = tf.nn.softmax(self.lang_output_logits)

        self.lang_item_losses = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.lang_output_logits,
            labels=self.processed_labels)

        self.lang_total_loss = tf.reduce_mean(self.lang_item_losses)

        if architecture_is_meta and self.train_with_meta:
            self.meta_output_logits = _output_network(self.meta_raw_output, 
                                                      reuse=True)
            self.meta_output = tf.nn.softmax(self.meta_output_logits)
            self.meta_item_losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.meta_output_logits,
                labels=self.processed_labels)

            self.meta_total_loss = tf.reduce_mean(self.meta_item_losses)

        # l2 reg
        weight_vars = [v for v in tf.trainable_variables() if 'bias' not in v.name and 'embeddings' not in v.name]
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in weight_vars]) * config["l2_penalty_weight"]

        self.train_loss = self.lang_total_loss + self.l2_loss 
        if architecture_is_meta and self.train_with_meta:
            self.train_loss += self.meta_total_loss


        corr_scores = tf.reduce_sum(tf.multiply(self.lang_output_logits, self.processed_labels), axis=-1) 
        incorr_scores = tf.reduce_sum(tf.multiply(self.lang_output_logits, 1.-self.processed_labels), axis=-1) 
        self.item_scores = tf.cast(tf.greater(corr_scores, incorr_scores), tf.float32)
        self.pct_correct = tf.reduce_mean(self.item_scores)

        self.lr_ph = tf.placeholder(tf.float32)
        optimizer = tf.train.RMSPropOptimizer(self.lr_ph)

        self.train = optimizer.minimize(self.train_loss)

        # Saver
        self.saver = tf.train.Saver()

        # initialize
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())


    def _random_guess_mask(self, dataset_length, meta_batch_size=None):
        if meta_batch_size is None:
            meta_batch_size = config["meta_batch_size"]
        mask = np.zeros(dataset_length, dtype=np.bool)
        indices = np.random.permutation(dataset_length)[:meta_batch_size]
        mask[indices] = True
        return mask


    def evaluate(self, dataset):
        inputs = dataset["input"]
        targets = dataset["output"]
        queries = dataset["query"]
        loss = 0.
        meta_loss = 0.
        pct_correct = 0.
        if self.train_with_meta:
            fetches = [self.lang_total_loss, 
                       self.pct_correct,
                       self.meta_total_loss]
        else:
            fetches = [self.lang_total_loss, 
                       self.pct_correct]

        for i, query in enumerate(queries):
            this_q_inputs = inputs[i]
            this_q_targets = targets[i]

            feed_dict = {
                self.keep_ph: 1.,
                self.query_ph: query,
                self.visual_input_ph: this_q_inputs,
                self.target_ph: this_q_targets
            }
            if self.train_with_meta:
                feed_dict[self.guess_input_mask_ph] =  np.ones([len(this_q_targets)])

            results =  self.sess.run(fetches, 
                                     feed_dict=feed_dict)

            if self.train_with_meta:
                this_loss, this_pctc, this_meta_loss = results
                meta_loss += this_meta_loss
            else:
                this_loss, this_pctc = results
            loss += this_loss
            pct_correct += this_pctc

        loss /= i + 1
        pct_correct /= i + 1
        if self.train_with_meta:
            meta_loss /= i + 1
            return loss, pct_correct, meta_loss
        else:
            return loss, pct_correct


    def all_eval(self):
        names = []
        results = []
        for name, dataset in self.data.items():
            this_results = self.evaluate(dataset)
            results += this_results 
            if self.train_with_meta:
                names += [name + "_loss", name + "_pct_correct", name + "_meta_loss"]
            else:
                names += [name + "_loss", name + "_pct_correct"]

        return names, results


    def train_step(self, dataset, lr):
        inputs = dataset["input"]
        targets = dataset["output"]
        queries = dataset["query"]
        order = np.random.permutation(len(queries))
        for i in order:
            query = queries[i]
            this_q_inputs = inputs[i]
            this_q_targets = targets[i]
#            subset = np.random.permutation(len(this_q_targets))[:self.train_batch_subset]
#            this_q_inputs = this_q_inputs[subset]
#            this_q_targets = this_q_targets[subset]

            feed_dict = {
                self.lr_ph: lr,
                self.keep_ph: config["train_keep_prob"],
                self.query_ph: query,
                self.visual_input_ph: this_q_inputs,
                self.target_ph: this_q_targets
            }
            if self.train_with_meta:
                feed_dict[self.guess_input_mask_ph] =  self._random_guess_mask(len(this_q_targets))

            self.sess.run(self.train, 
                          feed_dict=feed_dict)


    def do_training(self, filename):
        with open(filename, "w") as fout:
            names, losses = self.all_eval()
            print(names)
            fout.write(", ".join(["epoch"] + names) + "\n")
            loss_format = ", ".join(["%i"] + ["%f"] * len(losses))
            fout.write(loss_format % tuple([0] + losses))
            print(loss_format % tuple([0] + losses))
            lr = self.config["init_learning_rate"] 
            eval_every = self.config["eval_every"]
            lr_decay = self.config["lr_decay"]
            decay_every = self.config["lr_decays_every"]
            min_lr = self.config["min_lr"]
            for epoch in range(1, self.config["num_epochs"] + 1): 
                self.train_step(self.data["train.large"], lr)

                if epoch % eval_every == 0: 
                    _, losses = self.all_eval()
                    fout.write(loss_format % tuple([epoch] + losses))
                    print(loss_format % tuple([epoch] + losses))

                if epoch % decay_every == 0 and lr > min_lr:
                    lr *= lr_decay


_save_config(config["results_path"] + "config.csv", config)
for run_i in range(config["run_offset"], config["run_offset"] + config["num_runs"]):
    np.random.seed(run_i)
    tf.set_random_seed(run_i)
    model = shape_model(config, data)
    model.do_training(config["results_path"] + "run%i_model%s_losses.csv" % (run_i, config["architecture"]))
    tf.reset_default_graph()

