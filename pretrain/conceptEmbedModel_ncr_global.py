from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
SEED = 42
np.random.seed(SEED)
import random

random.seed(SEED)
import json
import pickle
import fasttext as fastText
import re
from tqdm import tqdm
from abbrrep_class import AbbrRep
import math
import scipy.sparse
from timeit import timeit

import os
import ast

def read_word_weightings(idf_dict):
    s = open(idf_dict, 'r').read()
    whip = ast.literal_eval(s)
    return whip

idfWeights_dict = "word_weighting_dict.txt"
IDF_WEIGHTS = read_word_weightings(idfWeights_dict)

class ConceptEmbedModel():
    def phrase2vec(self, phrase_list, max_length):
        phrase_vec_list = []
        phrase_seq_lengths = []
        global_context_list = []
        for phrase in phrase_list:
            total_weighting = 0
            global_context = np.zeros(100)

            content = phrase.split("|")
            features_left = content[-2].split()
            features_right = content[-1].split()
            doc = content[-4].split()
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

    def __init__(self, config, data_train, word_model, cui2id, id2cui, ancestors_dict):

        # ************** Initialize variables ***************** #
        print("model with dynamic embedding generation loaded")
        tf.reset_default_graph()
        with tf.Graph().as_default():
            tf.set_random_seed(SEED)
        self.embed_dim = 100
        self.word_model = word_model
        self.config = config
        self.ns = config.ns
        self.trainLoss_array = []
        self.valLoss_array = []
        self.x = set()
        self.ancestor_matrix = ancestors_dict
        self.cui2id = cui2id
        self.id2cui = id2cui
        self.data_train = data_train
        self.config.concepts_size = len(self.id2cui)
        self.best_val_acc = 100000
       # self.config.concepts_size = len(self.ancestor_matrix)
        train_data = {}
        test_data = {}
        valid_data = {}
        self.outputdir = config.output
        self.ckpt_dir = '{}/checkpoints/'.format(self.outputdir)
        self.save_dir = '{}/'.format(self.ckpt_dir)
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_path = os.path.join(self.save_dir, 'best_validation')       

        with open(os.path.join(self.outputdir, "cui2id_ncr.pickle"), 'wb') as fhandle:
            pickle.dump(self.cui2id, fhandle)
        with open(os.path.join(self.outputdir, "id2cui_ncr.pickle"), 'wb') as fhandle:
            pickle.dump(self.id2cui, fhandle)
        sampled_buckets = {}
        num_samples_ = {}
        for key in sorted(self.data_train):
            exp_samples = self.data_train[key]
            if len(exp_samples) > 0:
                exp_samples = shuffle(exp_samples, random_state=SEED)
                max_samples = int(self.ns)
                num_samples = min(len(exp_samples), int(max_samples))

                valid_data[key] = exp_samples[int(num_samples * 0.9):]
                train_data[key] = exp_samples[:int(num_samples * 0.9)]
                num_samples_[key] = len(train_data[key])


        counter = 0
        training_samples = []
        training_labels = []
        for key in sorted(train_data):
            if key not in self.cui2id:
                continue    
            for item in train_data[key]:
                training_samples.append(item)
                training_labels.append(self.cui2id[key])

        print("len training samples::{}".format(len(training_samples)))        
        val_samples = []
        val_labels = []
        for key in sorted(valid_data):
            if key not in self.cui2id:
                continue
            for item in valid_data[key]:
                val_samples.append(item)
                val_labels.append(self.cui2id[key])

        self.training_samples = {}
        self.training_samples['seq'], self.training_samples['seq_len'], self.training_samples['g'] = self.phrase2vec(training_samples, self.config.max_sequence_length)
        self.training_samples['label'] = np.array(training_labels)
        print("training data loaded")

        self.val_samples = {}
        self.val_samples['seq'], self.val_samples['seq_len'], self.val_samples['g'] = self.phrase2vec(val_samples, self.config.max_sequence_length)
        self.val_samples['label'] = np.array(val_labels)

        print("validation data loaded")

        self.label = tf.placeholder(tf.int32, shape=[None])

        self.seq = tf.placeholder(tf.float32, shape=[None, config.max_sequence_length, self.embed_dim])
        self.seq_len = tf.placeholder(tf.int32, shape=[None])
        self.g = tf.placeholder(tf.float32, shape=[None, self.embed_dim])
        self.lr = tf.Variable(config.lr, trainable=False)
        self.is_training = tf.placeholder(tf.bool)
        # ************** Compute dense ancestor matrix from LIL matrix format ***************** #i
        num_rel = 0
        for row in tqdm(self.ancestor_matrix):
            for col in self.ancestor_matrix[row]:
                num_rel += 1
        print("num_rel:{}".format(num_rel))
        sparse_ancestrs = np.zeros((num_rel, 2))
        counter = 0
        for row in tqdm(self.ancestor_matrix):
            for col in self.ancestor_matrix[row]:
                #sparse_ancestrs.append([self.cui2id[row], self.cui2id[col]])
                sparse_ancestrs[counter][0] = self.cui2id[row]
                sparse_ancestrs[counter][1] = self.cui2id[col]
                counter += 1
        print("matrix loaded loaded")
        #sparse_ancestrs = np.array(sparse_ancestrs)
        
        print("num concepts: {}".format(config.concepts_size))

        self.ancestry_sparse_tensor = tf.sparse_reorder(
            tf.SparseTensor(indices=sparse_ancestrs, values=[1.0] * len(sparse_ancestrs),
                            dense_shape=[config.concepts_size, config.concepts_size]))
        # ************** Encoder for sentence embeddings ***************** #
        layer1 = tf.layers.conv1d(self.seq, 100, 1, activation=tf.nn.elu, \
                                  kernel_initializer=tf.initializers.he_normal(seed=SEED), \
                                  bias_initializer=tf.initializers.he_normal(seed=SEED), use_bias=True)
        layer2 = tf.layers.dense(tf.concat([tf.reduce_max(layer1, [1]), self.g], 1), 50, activation=tf.nn.relu, \
                                 kernel_initializer=tf.initializers.he_normal(seed=SEED),
                                  bias_initializer=tf.initializers.he_normal(seed=SEED),
                                 use_bias=True)
        self.seq_embedding = tf.nn.l2_normalize(layer2, axis=1)
        # ************** Concept embeddings ***************** #
        self.embeddings = tf.get_variable("embeddings", shape=[self.config.concepts_size, 50],
                                          initializer=tf.random_normal_initializer(stddev=0.1, seed=SEED))
        self.aggregated_embeddings = tf.sparse_tensor_dense_matmul(self.ancestry_sparse_tensor, self.embeddings)
        aggregated_w = self.aggregated_embeddings

        last_layer_b = tf.get_variable('last_layer_bias', shape=[self.config.concepts_size],
                                       initializer=tf.random_normal_initializer(stddev=1e-1, seed=SEED))
        self.score_layer = tf.matmul(self.seq_embedding, tf.transpose(aggregated_w)) + last_layer_b
        # ************** Loss ***************** #
        self.loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(self.label,
                                                                          self.score_layer))  # + reg_constant*tf.reduce_sum(reg_losses)

        self.pred = tf.nn.softmax(self.score_layer)


        # ************** Backprop ***************** #
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        config_ = tf.ConfigProto()
        config_.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config_)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def save_params(self, epoch, repdir='.'):
        tf.train.Saver().save(self.sess, (repdir + "/e" + str(epoch) + '_params.ckpt').replace('//', '/'))
    def train_epoch(self, epoch):
        report_loss_train = 0
        report_len_train = 0
        head = 0
        training_size = self.training_samples['seq'].shape[0]
        print("training_size::{}".format(training_size))
        shuffled_indecies = shuffle(list(range(training_size)), random_state=SEED)
        random.shuffle(shuffled_indecies)
        if self.config.batch_size == 100:
            self.config.batch_size = training_size
        while head < training_size:
            ending = min(training_size, head + self.config.batch_size)
            batch = {}
            for cat in self.training_samples:
                batch[cat] = self.training_samples[cat][shuffled_indecies[head:ending]]
            report_len_train += ending - head
            head += self.config.batch_size
            batch_feed = {self.seq: batch['seq'], \
                          self.seq_len: batch['seq_len'], \
                          self.g: batch['g'],
                          self.label: batch['label'],
                          self.is_training: True}
            _, batch_loss = self.sess.run([self.train_step, self.loss], feed_dict=batch_feed)
            report_loss_train += batch_loss
        print("{} Epoch".format(epoch))
        if report_len_train > 0:
            print(str(epoch) + " Epoch loss: " + str(report_loss_train / report_len_train))
            self.trainLoss_array.append(report_loss_train / report_len_train)

        report_loss_val = 0
        report_len_val = 0
        head = 0
        val_size = self.val_samples['seq'].shape[0]
        shuffled_indecies = shuffle(list(range(val_size)), random_state=SEED)
        
        while head < val_size:
            ending = min(val_size, head + self.config.batch_size)
            batch = {}
            for cat in self.val_samples:
                batch[cat] = self.val_samples[cat][shuffled_indecies[head:ending]]

            report_len_val += ending - head
            head += self.config.batch_size
            batch_feed = {self.seq: batch['seq'], \
                          self.seq_len: batch['seq_len'], \
                          self.g: batch['g'],
                          self.label: batch['label'],
                          self.is_training: True}
            _, batch_loss = self.sess.run([self.pred, self.loss], feed_dict=batch_feed)
            report_loss_val += batch_loss
        if epoch == 0:
            self.saver.save(sess=self.sess, save_path=self.save_path)
        if report_len_val > 0:
            #print(str(epoch) + " Epoch loss: " + str(report_loss_val / report_len_val))
            
            self.valLoss_array.append(report_loss_val / report_len_val)
            curr_val_loss = report_loss_val / report_len_val   
            if curr_val_loss < self.best_val_acc:
                if report_len_val > 0:
                    self.best_val_acc = curr_val_loss
                    improved_str = "*"
                self.saver.save(sess=self.sess, save_path=self.save_path)
            else:
                improved_str = ""
            print("{} Epoch loss: {} {}".format(epoch, curr_val_loss, improved_str))
            #f_handle.write("{} temp {} Epoch loss: {} {}\n".format(self.temperature, epoch, curr_val_loss, improved_str))
    def get_probs(self, val_samples):
        seq, seq_len, g = self.phrase2vec(val_samples, self.config.max_sequence_length)
        querry_dict = {self.seq: seq, self.seq_len: seq_len, self.g: g, self.is_training: False}
        res_querry = self.sess.run(self.pred, feed_dict=querry_dict)
        return res_querry

    def label_data(self, data, abbr, source, mimic=False):
        results = {}
        results["mimic"] = {}
        results["casi"] = {}
        batch_size = 512
        head = 0
        while head < len(data):
            querry_subset = data[head:min(head + batch_size, len(data))]

            res_tmp = self.get_probs(querry_subset) 
            if head == 0:
                res_querry = res_tmp 
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
        score = 0
        total = 0

        for tmp_res in closest_concepts:
            pred = tmp_res[0][0]
            label = tmp_res[0][2]
            if not mimic:
                ground_truth_label = data[counter].split("|")[0].replace(",", "")
            else:
                ground_truth_label = data[counter].split("|")[2].replace(",", "")
            ground_truth = self.exp2id[ground_truth_label]



            if label == ground_truth:
                score += 1
            if self.config.verbose:
                if mimic:
                    print("mimic" + "|" + str(tmp_res[0]) + " |" + str(label) + "|" + pred + "|" + str(ground_truth)  + "|" + ground_truth_label)
                else:
                    print("casi" + "|" + str(tmp_res[0]) + " |" + str(label) + "|" + pred + "|" + str(ground_truth)  + "|" + ground_truth_label)
            total += 1
            counter += 1
        results[source][abbr] = [score, total]
        return results
