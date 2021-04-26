"""
Code adapted from Aryan Arbabi, https://github.com/a-arbabi/NeuralCR
"""
SEED=42
import ast
import argparse
import numpy as np
import os
import json
import fasttext as fastText
#import fastText
import pickle
import tensorflow as tf
#from tqdm import tqdm
#import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from timeit import timeit
tf.reset_default_graph()
from timeit import timeit



def load_ncr(pretrain, max_sequence_length, globalcontext=True):
        # ************** Initialize variables ***************** #
    tf.reset_default_graph()
    with tf.Graph().as_default():
        tf.set_random_seed(SEED)
    # ************** Initialize matrix dimensions ***************** #
    ncr_cui2id, ncr_id2cui = {}, {}
    #try:
    with open("{}/cui2id_ncr.pickle".format(pretrain), 'rb') as fhandle:
        ncr_cui2id = pickle.load(fhandle)
    with open("{}/id2cui_ncr.pickle".format(pretrain), 'rb') as fhandle:
        ncr_id2cui = pickle.load(fhandle)
    #except: pass
    concepts_size = len(ncr_id2cui)
    label = tf.placeholder(tf.int32, shape=[None])
    g = tf.placeholder(tf.float32, shape=[None, 100])
    seq = tf.placeholder(tf.float32, shape=[None, max_sequence_length, 100])
    seq_len = tf.placeholder(tf.int32, shape=[None])
    is_training = tf.placeholder(tf.bool)
    print(seq)
    # ************** Compute dense ancestor matrix from LIL matrix format ***************** #

    # ************** Encoder for sentence embeddings ***************** #
    layer1 = tf.layers.conv1d(seq, 100, 1, activation=tf.nn.elu, \
                              kernel_initializer=tf.initializers.he_normal(seed=SEED), \
                              bias_initializer=tf.initializers.he_normal(seed=SEED), use_bias=True)

    if globalcontext:
        layer2 = tf.layers.dense(tf.concat([tf.reduce_max(layer1, [1]), g], 1), 50, activation=tf.nn.relu, \
                             kernel_initializer=tf.initializers.he_normal(seed=SEED),
                              bias_initializer=tf.initializers.he_normal(seed=SEED),
                             use_bias=True)
    else:
        layer2 = tf.layers.dense(tf.reduce_max(layer1, [1]), 50, activation=tf.nn.relu, \
                             kernel_initializer=tf.initializers.he_normal(seed=SEED),
                             bias_initializer=tf.initializers.he_normal(seed=SEED),
                             use_bias=True)
    seq_embedding = tf.nn.l2_normalize(layer2, axis=1)
    # ************** Concept embeddings ***************** #
    embeddings = tf.get_variable("embeddings", shape=[concepts_size, 50],
                                      initializer=tf.random_normal_initializer(stddev=1e-1, seed=SEED), dtype=tf.float32)

    aggregated_w = embeddings

    last_layer_b = tf.get_variable('last_layer_bias', shape=[concepts_size],dtype=tf.float32)

    score_layer = tf.matmul(seq_embedding, tf.transpose(aggregated_w)) + last_layer_b
    # ************** Loss ***************** #
    loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(label,
                                                                      score_layer))  # + reg_constant*tf.reduce_sum(reg_losses)

    pred = tf.nn.softmax(score_layer)
    init_op = tf.global_variables_initializer() 
    saver = tf.train.Saver()
    sess = tf.Session()
    
    restore_path = "{}/checkpoints/best_validation".format(pretrain)
    saver.restore(sess, restore_path) 

    # print(sess.graph.get_operations())
    print("Model restored from {}.".format(restore_path))
    vars = tf.trainable_variables()
    vars_vals = sess.run(vars)
    layer1_weights, layer1_bias, layer2_weights, layer2_bias, cui_embedding_weights, cui_embedding_bias = None, None, None, None, None, None
    for var, val in zip(vars, vars_vals):
        print(var.name)
        if var.name == "conv1d/bias:0":
            layer1_bias = val
        if var.name == "conv1d/kernel:0":
            layer1_weights = val
        if var.name == "dense/kernel:0":
            layer2_weights = val
        if var.name == "dense/bias:0":
            layer2_bias = val 
        if var.name == "embeddings:0":
            cui_emedding_weights = val
        if var.name == "last_layer_bias:0":
            cui_embedding_bias = val
    return layer1_weights, layer1_bias, layer2_weights, layer2_bias, cui_emedding_weights, cui_embedding_bias, ncr_cui2id, ncr_id2cui


#with open("y/cui2id_ncr.pickle", 'rb') as fhandle:
 #   cui2id_ncr = pickle.load(fhandle)
  #  size = len(cui2id_ncr)

#layer1_weights, layer1_bias, layer2_weights, layer2_bias, cui_emeddinn
#g_weights, cui_embedding_bias = load_ncr("y", size, 6)
#a, b, c, d, e, f = load_ncr("y", size, 6)
#print(type(a))
#print(a.shape, b.shape, c.shape, d.shape, e.shape, f.shape)
