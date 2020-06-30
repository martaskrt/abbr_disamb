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


def read_word_weightings(idf_dict):
    s = open(idf_dict, 'r').read()
    whip = ast.literal_eval(s)
    return whip

idfWeights_dict = "word_weighting_dict.txt"
IDF_WEIGHTS = read_word_weightings(idfWeights_dict)


class ConceptEmbedModel():
    def phrase2vec(self, phrase_list, max_length, source):
        phrase_vec_list = []
        phrase_seq_lengths = []
        global_context_list = []
        for phrase in phrase_list:
            total_weighting = 0
            global_context = np.zeros(100)

            content = phrase.split("|")[:-1]
            if source == "casi":
                assert len(content) == 5
            else:
                assert len(content) == 7
            features_left = content[-2].split()
            features_right = content[-1].split()
            ###################2020-02-16###########
            doc_left = content[-3]
            try:
                doc_right = content[-4].split(doc_left)[1].split()
            except ValueError:
                doc_right = content[-4].split()
            doc_left = doc_left.split()

            num_right = min(20, len(doc_right))
            num_left = min(20, len(doc_left))

            doc_right = doc_right[:num_right]
            doc_left = doc_left[len(doc_left)-num_left:]
            doc = doc_left + doc_right
            #########################################
            #doc = content[-4].split()
            for z in doc:
                try:
                    current_word_weighting = IDF_WEIGHTS[z]
                except KeyError:
                    current_word_weighting = 0
                #current_word_weighting = IDF_WEIGHTS[z]
                global_context = np.add(global_context, (current_word_weighting * self.word_model.get_word_vector(z)))
                total_weighting += current_word_weighting
            if total_weighting > 0:
                global_context = global_context / total_weighting
            start_left = int(max(len(features_left) - (max_length / 2), 0))
            tokens = features_left[start_left:]
            end_right = int(min(len(features_right), (max_length / 2)))
            tokens_right = features_right[:end_right]
            tokens.extend(tokens_right)
            phrase_vec_list.append(
                [self.word_model.get_word_vector(tokens[i]) if i < len(tokens) else [0] * 100 for i in
                 range(max_length)])
            global_context_list.append(global_context)
            phrase_seq_lengths.append(len(tokens))
        return np.array(phrase_vec_list), np.array(phrase_seq_lengths), np.array(global_context_list)

    def __init__(self, config, word_model, casi_test, exp2id, id2exp):
            # ************** Initialize variables ***************** #
        tf.reset_default_graph()
        with tf.Graph().as_default():
            tf.set_random_seed(SEED)
        self.embed_dim = 100
        self.word_model = word_model
        self.config = config
        self.abbr = config.abbr
        self.trainLoss_array = []
        self.valLoss_array = []
        self.buckets = {}
        self.buckets['ancestors'] = {}
        self.x = set()
        self.cui_probs = {}
        self.exp2id = exp2id
        self.id2exp = id2exp
        self.distance_dict = {}
        self.config.concepts_size = len(self.id2exp)
        # ************** Initialize matrix dimensions ***************** #

        self.label = tf.placeholder(tf.int32, shape=[None])
        self.g = tf.placeholder(tf.float32, shape=[None, self.embed_dim])
        self.seq = tf.placeholder(tf.float32, shape=[None, config.max_sequence_length, self.embed_dim])
        self.seq_len = tf.placeholder(tf.int32, shape=[None])
        self.is_training = tf.placeholder(tf.bool)
        # ************** Compute dense ancestor matrix from LIL matrix format ***************** #

        # ************** Encoder for sentence embeddings ***************** #
        layer1 = tf.layers.conv1d(self.seq, 100, 1, activation=tf.nn.elu, \
                                  kernel_initializer=tf.initializers.he_normal(seed=SEED), \
                                  bias_initializer=tf.initializers.he_normal(seed=SEED), use_bias=True)

       # layer2 = tf.layers.dense(tf.concat([tf.reduce_max(layer1, [1]), self.g], 1), 200, activation=tf.nn.relu, \
        #                         kernel_initializer=tf.initializers.he_normal(seed=SEED),
         #                        bias_initializer=tf.initializers.he_normal(seed=SEED),
          #                       use_bias=True)
        if self.config.globalcontext:
            layer2 = tf.layers.dense(tf.concat([tf.reduce_max(layer1, [1]), self.g], 1), 50, activation=tf.nn.relu, \
                                 kernel_initializer=tf.initializers.he_normal(seed=SEED),
                                  bias_initializer=tf.initializers.he_normal(seed=SEED),
                                 use_bias=True)
        else:
            layer2 = tf.layers.dense(tf.reduce_max(layer1, [1]), 50, activation=tf.nn.relu, \
                                 kernel_initializer=tf.initializers.he_normal(seed=SEED),
                                 bias_initializer=tf.initializers.he_normal(seed=SEED),
                                 use_bias=True)
        self.seq_embedding = tf.nn.l2_normalize(layer2, axis=1)
        # ************** Concept embeddings ***************** #
        self.embeddings = tf.get_variable("embeddings", shape=[self.config.concepts_size, 50],
                                          initializer=tf.random_normal_initializer(stddev=1e-1, seed=SEED))

        aggregated_w = self.embeddings

        last_layer_b = tf.get_variable('last_layer_bias', shape=[self.config.concepts_size])

        self.score_layer = tf.matmul(self.seq_embedding, tf.transpose(aggregated_w)) + last_layer_b
        # ************** Loss ***************** #
        self.loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(self.label,
                                                                          self.score_layer))  # + reg_constant*tf.reduce_sum(reg_losses)

        self.pred = tf.nn.softmax(self.score_layer)
        init_op = tf.global_variables_initializer() 
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.saver.restore(self.sess, "{}/checkpoints/{}/{}".format(self.config.output, self.abbr, self.abbr))
        print("Model restored.")

    def get_probs(self, samples, source):
        seq, seq_len, g = self.phrase2vec(samples, self.config.max_sequence_length, source)
        querry_dict = {self.seq: seq, self.seq_len: seq_len, self.g: g, self.is_training: False}
        res_querry = self.sess.run(self.pred, feed_dict=querry_dict)
        return res_querry

    def label_data(self, args, data, abbr, source, mimic=False):
       
        batch_size = 512
        head = 0
        while head < len(data):
            querry_subset = data[head:min(head + batch_size, len(data))]

            res_tmp = self.get_probs(querry_subset, source) #!!!!!!!!!!!!!!!!!!!!!!!!!
            if head == 0:
                res_querry = res_tmp  # self.get_probs(querry_subset)
            else:
                res_querry = np.concatenate((res_querry, res_tmp))

            head += batch_size

        closest_concepts = []
        for s in range(len(data)):
            indecies_querry = np.argsort(-res_querry[s, :])
            tmp_res = []
            for i in indecies_querry:
                tmp_res.append((self.id2exp[i], res_querry[s, i], i))

            closest_concepts.append(tmp_res)
        counter = 0
        score = []
        total = []

        for tmp_res in closest_concepts:
            pred = tmp_res[0][0]
            label = tmp_res[0][2]
            if not mimic:
                ground_truth_label = data[counter].split("|")[0].replace(",", "")
            else:
                ground_truth_label = data[counter].split("|")[2].replace(",", "")
            ground_truth = self.exp2id[ground_truth_label]

        for tmp_res in closest_concepts:
            pred = tmp_res[0][0]
            label = tmp_res[0][2]
            if not mimic:
                ground_truth_label = data[counter].split("|")[0].replace(",", "")
            else:
                ground_truth_label = data[counter].split("|")[2].replace(",", "")
            ground_truth = self.exp2id[ground_truth_label]


            if label == ground_truth:
                #score += 1
                score.append(1)
            else:
                score.append(0)
            #total += 1
            total.append(1)
            counter += 1
        results = [score, total]
        return results
 

def run_testset(args, exp2id, id2exp, casi_test, mimic_test, word_model):
   
    model = ConceptEmbedModel(args, word_model, casi_test, exp2id, id2exp)

    param_dir = args.output
    abbr = args.abbr 
    results = []

    casi_results = model.label_data(args, casi_test, abbr, source="casi")
    mimic_results = model.label_data(args, mimic_test, abbr, source="mimic", mimic=True)

    samples_casi = len(casi_test)
    samples_mimic = len(mimic_test)
    start = timeit()
    for i in range(args.bootstrap_runs):
        #curr_bootstrap_sample_casi = np.random.choice(casi_test, size=samples_casi, replace=True)
        #curr_bootstrap_sample_mimic = np.random.choice(mimic_test, size=samples_mimic, replace=True)
        #curr_results_casi = model.label_data(args, curr_bootstrap_sample_casi, abbr, source="casi")
        #curr_results_mimic = model.label_data(args, curr_bootstrap_sample_mimic, abbr, source="mimic", mimic=True)
        correct_casi = int(np.random.choice(casi_results[0], size=samples_casi, replace=True).sum())
        total_casi = samples_casi
        micro_acc_casi = correct_casi/total_casi

        correct_mimic = int(np.random.choice(mimic_results[0], size=samples_mimic, replace=True).sum())       
        total_mimic = samples_mimic
        micro_acc_mimic = correct_mimic/total_mimic
        #correct_casi = curr_results_casi[0]
        #total_casi = curr_results_casi[1]
        #micro_acc_casi = correct_casi/total_casi
        
        #correct_mimic = curr_results_mimic[0]
        #total_mimic = curr_results_mimic[1]
        #micro_acc_mimic = correct_mimic/total_mimic

        results.append([correct_casi, total_casi, micro_acc_casi, correct_mimic, total_mimic, micro_acc_mimic])
    end = timeit()
    assert len(results) == args.bootstrap_runs
    return results, casi_results, mimic_results


