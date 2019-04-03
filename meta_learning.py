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
    "num_hidden": 64,
    "num_hidden_hyper": 512, 

    "init_learning_rate": 1e-4,
    "lr_decay": 0.85,
    "lr_decays_every": 100,
    "min_learning_rate": 3e-8,

    "internal_nonlinearity": tf.nn.leaky_relu,

    "data_path": "shapes_data/",
    "results_path": "/mnt/fs2/lampinen/meta_shapes/results/"
}


### load data

data_path = config["data_path"]
data = {}
max_query_len = 0

vocab = ["PAD", "EOS", "is", "triangle", "circle", "square", "above", "below", "right_of", "left_of", "red", "green", "blue"]
vocab_to_ints = dict(zip(vocab, range(len(vocab))))

def process_query_data(query_data, max_query_len):
    query_data = [re.sub('[()]', '', x).split() for x in query_data]
    query_data = [["PAD"] * (max_query_len - len(x)) + x + ["EOS"] for x in query_data]
    query_data = [[vocab_to_ints[w] for w in x] for x in query_data]
    return np.array(query_data, dtype=np.int32)
    

for dataset in ["train.large", "val", "test"]:
    data[dataset] = {}
    data[dataset]["input"] = np.load(data_path + dataset + ".input.npy")
    data[dataset]["output"] = np.loadtxt(data_path + dataset + ".output", dtype=np.str) == 'true'
    data[dataset]["query_raw"] = np.loadtxt(data_path + dataset + ".query", dtype=np.str, delimiter=",")
    max_query_len = max(max_query_len, max([len(x.split()) for x in data[dataset]["query_raw"]]))

for dataset in ["train.large", "val", "test"]:
    data[dataset]["query"] = process_query_data(data[dataset]["query_raw"], max_query_len) 


