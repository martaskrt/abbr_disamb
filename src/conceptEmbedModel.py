from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import random
import ast
SEED=42
random.seed(SEED)
import json
import pickle
import fasttext as fastText
import re
#from abbrrep_class import AbbrRep
import math
import scipy.sparse
from timeit import timeit
import os
import load_ncr

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

            content = phrase.split("|")[:-1]
            assert len(content) == 7

            features_left = content[-2].split()
            features_right = content[-1].split()

            doc = content[-4].split()
            for z in doc:
                try:
                    current_word_weighting = IDF_WEIGHTS[z]
                except KeyError:
                    current_word_weighting = 0
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

    def __init__(self, config, data_train, word_model, exp2id, id2exp, temperature=1):
        # ************** Initialize variables ***************** #
        if config.pretrain:
            layer1_weights, layer1_bias, layer2_weights, layer2_bias, ncr_embedding_weights, ncr_embedding_bias, self.ncr_cui2id, self.ncr_id2cui = load_ncr.load_ncr(config.pretrain, config.max_sequence_length, config.globalcontext) 
           
        tf.reset_default_graph()
        with tf.Graph().as_default():
            tf.set_random_seed(SEED)
        self.embed_dim = 100
        self.word_model = word_model
        self.config = config
        self.ns = config.ns
        self.abbr = config.abbr
        self.trainLoss_array = []
        self.valLoss_array = []
        self.buckets = {}
        self.buckets['ancestors'] = {}
        self.x = set()
        self.epsilon = config.epsilon
        self.cui_probs = {}
        self.exp2id = exp2id
        self.id2exp = id2exp
        self.distance_dict = {}
        self.data_train = data_train
        self.config.concepts_size = len(self.id2exp)
        self.temperature = temperature
        
        self.ckpt_dir = '{}/checkpoints/'.format(self.config.output)
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.save_dir = '{}/{}/'.format(self.ckpt_dir, self.abbr)
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        self.save_path = os.path.join(self.save_dir, '{}_best_validation_{}'.format(self.abbr, self.temperature))
        
        self.best_epoch = 0
        self.best_val_loss = 100000000000000
        # ************** Initialize matrix dimensions ***************** #

        train_data = {}
        test_data = {}
        valid_data = {}


        sampled_buckets = {}
        num_samples_ = {}
        for key in sorted(self.data_train):
            exp_samples = self.data_train[key]['expansion']
            if len(exp_samples) > 0:
                exp_samples = shuffle(exp_samples, random_state=SEED)
                max_samples = int(self.ns)
                num_samples = min(len(exp_samples), int(max_samples))

                test_data[key] = exp_samples[int(num_samples * 0.80):num_samples]
                valid_data[key] = exp_samples[int(num_samples * 0.6):int(num_samples * 0.8)]
                train_data[key] = exp_samples[:int(num_samples * 0.6)]
                num_samples_[key] = len(train_data[key])


        sigmoid_denom = {}
        prob_data = {}
        for i in sorted(list(self.id2exp.keys())):
            key = self.id2exp[i]
            prob_data[key] = {} 
            sigmoid_denom[key] = 0
            denom_present = False
            if key in train_data and len(train_data[key]) > 0:
                sigmoid_denom[key] = np.exp(-self.epsilon/self.temperature)
                denom_present = True
            if config.use_relatives:
                for exp_pair in sorted(self.data_train[key]['relative']):
                    distance = exp_pair[1]
                    sigmoid_denom[key] += np.exp(-distance/self.temperature)
                    denom_present = True
            if key not in self.cui_probs:
                self.cui_probs[key] = {}
            if key in train_data and key in sigmoid_denom and  sigmoid_denom[key] > 0:
                confidence = np.exp(-self.epsilon/self.temperature)/sigmoid_denom[key]
                self.cui_probs[key][key] = confidence
                prob_data[key][key] = [self.epsilon, confidence]
            else: 
                self.cui_probs[key][key] = 0
                prob_data[key][key] = [0, 0]

        if config.use_relatives:
            for expansion in sorted(self.data_train):
                if expansion not in train_data:
                    train_data[expansion] = []
                if expansion not in self.cui_probs:
                    self.cui_probs[expansion] = {}
                for key in self.data_train[expansion]['relative']:
                    relative = key[0]
                    distance_to_exp = key[1]
                    confidence = np.exp(-distance_to_exp/self.temperature) / sigmoid_denom[expansion]
                    prob_data[expansion][relative] = [distance_to_exp, confidence]
                    self.cui_probs[expansion][relative] = confidence
#####################
        sampled_trainining_data ={}
        for i in sorted(self.id2exp):
            key = self.id2exp[i]
            sampled_trainining_data[key] = []
            if len(self.data_train[key]['expansion']) == 0 and not self.config.use_relatives:
                continue
        
            np.random.seed(SEED)
            cuis = []
            cuis_probs = []
            prob_sum = 0
            for cui in sorted(self.cui_probs[key]):
                if cui != key and not self.config.use_relatives:
                    continue
                cuis.append(cui)
                cuis_probs.append(self.cui_probs[key][cui])
                prob_sum += self.cui_probs[key][cui]
            if prob_sum == 0:
                continue
            cuis_probs = [i/prob_sum for i in cuis_probs]
            sampling = np.random.choice(cuis, int(self.ns*0.6), p=cuis_probs, replace=True)
            sampling_count = {}
            sampling_count[key] = {}
            for relative in sampling:
                if relative not in sampling_count[key]:
                    sampling_count[key][relative] = 0
                sampling_count[key][relative] += 1
            for relative in sorted(sampling_count[key]):
                if relative == key and key in train_data and len(train_data[key]) > 0:
                    ancestor_samples = train_data[key]
                elif relative == key:
                    continue
                elif self.config.use_relatives:
                    ancestor_samples = self.data_train[key]['relative'][(relative, prob_data[key][relative][0])]
                elif not self.config.use_relatives:
                    continue
                if self.config.replace:
                    sampled_trainining_data[key] += list(np.random.choice(ancestor_samples, sampling_count[key][relative] , replace=True))
                else:
                    if len(ancestor_samples) >= sampling_count[key][relative]:
                        sampled_trainining_data[key] += list(np.random.choice(ancestor_samples, sampling_count[key][relative] , replace=False))
                    else:
                        sampled_trainining_data[key] += ancestor_samples
                            
        counter = 0
        training_data_stats = {}
        training_samples = []
        training_labels = []
        for key in sorted(sampled_trainining_data):
            training_data_stats[key] = {}
            curr_data = sampled_trainining_data[key]
            for item in curr_data:
                label = item.split("|")[2]
                if label not in training_data_stats[key]:
                    training_data_stats[key][label] = 0
                training_data_stats[key][label] += 1
                training_samples.append(item)
                training_labels.append(self.exp2id[label])


        log_dir_path = os.path.join(self.config.output, "val_logfiles")
        if not os.path.isdir(log_dir_path):
            os.makedirs(log_dir_path)
        log_dir_abbr = '{}/{}/'.format(log_dir_path, self.abbr)
        if not os.path.isdir(log_dir_abbr):
            os.makedirs(log_dir_abbr)
        log_file_path = "{}_relative_stats_{}.txt".format(self.abbr, self.temperature)
        with open(os.path.join(log_dir_abbr, log_file_path), 'w') as g:
            for key in sorted(training_data_stats):
                if key in num_samples_:
                    g.write("{}\t{}\n".format(key, num_samples_[key]))
                else:
                    g.write("{}\t{}\n".format(key, 0))
                for label in sorted(training_data_stats[key]):
                    if label == key:
                        if key in sigmoid_denom: 
                            g.write("\t{}\t{}\t{:.20f}\t{}\n".format(label, self.epsilon, np.exp(-self.epsilon)/sigmoid_denom[key], training_data_stats[key][label]))
                        else:
                            g.write("\t{}\t{}\t{}\t{}\n".format(label, self.epsilon, 1, training_data_stats[key][label]))
                        
                    else:
                        try:
                            g.write("\t{}\t{:.5f}\t{:.20f}\t{}\n".format(label, prob_data[key][label][0], prob_data[key][label][1], training_data_stats[key][label]))
                        except:
                            continue
        counter = 0
        test_samples = []
        test_labels = []
        for key in sorted(test_data):
            for item in test_data[key]:
                label = item.split("|")[2]
                test_samples.append(item)
                test_labels.append(self.exp2id[label])
        
        val_samples = []
        val_labels = []
        for key in sorted(valid_data):
            for item in valid_data[key]:
                label = item.split("|")[2]
                val_samples.append(item)
                val_labels.append(self.exp2id[label])

        self.test_mimic = test_samples
        self.val_mimic = val_samples

        self.training_samples = {}
        start = timeit()
        self.training_samples['seq'], self.training_samples['seq_len'], self.training_samples['g'] = self.phrase2vec(training_samples,
                                                                                     self.config.max_sequence_length)
        self.training_samples['label'] = np.array(training_labels)

        self.val_samples = {}
        self.val_samples['seq'], self.val_samples['seq_len'], self.val_samples['g'] = self.phrase2vec(val_samples,
                                                                               self.config.max_sequence_length)
        self.val_samples['label'] = np.array(val_labels)

        # ************** Initialize matrix dimensions ***************** #
        self.label = tf.placeholder(tf.int32, shape=[None])
        #self.class_weights = tf.Variable(tf.ones([config.concepts_size]), False)

        self.seq = tf.placeholder(tf.float32, shape=[None, config.max_sequence_length, self.embed_dim])
        self.seq_len = tf.placeholder(tf.int32, shape=[None])
        self.g = tf.placeholder(tf.float32, shape=[None, self.embed_dim])
        self.lr = tf.Variable(config.lr, trainable=False)
        self.is_training = tf.placeholder(tf.bool)
        # ************** Compute dense ancestor matrix from LIL matrix format ***************** #

        if config.pretrain:
            with open("allacronyms_name2meta_20190617.pickle", 'rb') as f:
                name2meta = pickle.load(f)
            with open("allacronyms_meta2cui_20190617.pickle", 'rb') as f:
                meta2cui = pickle.load(f)

            embedding_weights = np.zeros((config.concepts_size, 50))
            embedding_bias = np.zeros((config.concepts_size, 1))
            
            for i in sorted(self.id2exp):
                exp = self.id2exp[i] # e.g. intravenous fluid
                possible_cuis = {}
                
                if config.abbr in name2meta and exp in name2meta[config.abbr]:
                    possible_cuis = meta2cui[config.abbr][name2meta[config.abbr][exp]]
                   
                pretrained_cui, pretrained_cui_id = None, None
                for cui in sorted(possible_cuis):
                    if cui in self.ncr_cui2id:
                        pretrained_cui_id, pretrained_cui = self.ncr_cui2id[cui], cui
                
                if pretrained_cui_id:
                    embedding_weights[i] = ncr_embedding_weights[pretrained_cui_id]
                    embedding_bias[i] = ncr_embedding_bias[pretrained_cui_id]
                else:
                    embedding_weights[i] = np.random.normal(0, 0.1, 50)
                    embedding_bias[i] = np.random.normal(0, 0.1, 1)
                    
            layer1 = tf.layers.conv1d(self.seq, 100, 1, activation=tf.nn.elu, kernel_initializer=tf.constant_initializer(layer1_weights), bias_initializer=tf.constant_initializer(layer1_bias), use_bias=True)
            layer2 = tf.layers.dense(tf.concat([tf.reduce_max(layer1, [1]), self.g], 1), 50, activation=tf.nn.relu, kernel_initializer=tf.constant_initializer(layer2_weights), bias_initializer=tf.constant_initializer(layer2_bias), use_bias=True)
            self.seq_embedding = tf.nn.l2_normalize(layer2, axis=1)
            # ************** Concept embeddings ***************** #
            self.embeddings = tf.get_variable("embeddings", shape=[self.config.concepts_size, 50], initializer=tf.constant_initializer(embedding_weights), dtype=tf.float32)
            aggregated_w = self.embeddings
            last_layer_b = tf.get_variable('last_layer_bias', shape=[self.config.concepts_size],
                                       initializer=tf.constant_initializer(embedding_bias), dtype=tf.float32)
        else:
        # ************** Encoder for sentence embeddings ***************** #
            layer1 = tf.layers.conv1d(self.seq, 100, 1, activation=tf.nn.elu, \
                                      kernel_initializer=tf.initializers.he_normal(seed=SEED), \
                                      bias_initializer=tf.initializers.he_normal(seed=SEED), use_bias=True)
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
                                                  initializer=tf.random_normal_initializer(stddev=1e-1, seed=SEED), dtype=tf.float32)
            aggregated_w = self.embeddings
            last_layer_b = tf.get_variable('last_layer_bias', shape=[self.config.concepts_size],
                                           initializer=tf.random_normal_initializer(stddev=1e-1, seed=SEED), dtype=tf.float32)
        self.score_layer = tf.matmul(self.seq_embedding, tf.transpose(aggregated_w)) + last_layer_b
        # ************** Loss ***************** #
        self.loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(self.label,
                                                                          self.score_layer))  # + reg_constant*tf.reduce_sum(reg_losses)

        self.pred = tf.nn.softmax(self.score_layer)


        # ************** Backprop ***************** #
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        
    def save_params(self, epoch, repdir='.'):
        tf.train.Saver().save(self.sess, (repdir + "/e" + str(epoch) + '_params.ckpt').replace('//', '/'))
    def train_epoch(self, epoch):
        report_loss_train = 0
        report_len_train = 0
        head = 0
        training_size = self.training_samples['seq'].shape[0]
        shuffled_indecies = shuffle(list(range(training_size)), random_state=SEED)
        random.shuffle(shuffled_indecies)
        if self.config.batch_size == -1:
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
        log_dir_path = os.path.join(self.config.output, "val_logfiles")
        log_dir_abbr = '{}/{}/'.format(log_dir_path, self.abbr)
        log_file_path = "{}_verbose_{}.txt".format(self.abbr, self.temperature)
        f = open(os.path.join(log_dir_abbr, log_file_path), 'a')
        f.write("{} Epoch temp_val: {}".format(epoch, self.temperature) + '\n')
        if report_len_train > 0:
            f.write(str(epoch) + " Epoch loss: " + str(report_loss_train / report_len_train) + '\n')
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
        if report_len_val > 0:
            curr_val_loss = report_loss_val / report_len_val
            self.valLoss_array.append(report_loss_val / report_len_val)
            if curr_val_loss < self.best_val_loss:
                self.best_val_loss = curr_val_loss
                self.best_epoch = epoch
                self.saver.save(sess=self.sess, save_path=self.save_path)
                improved_str = "*"
            else:
                improved_str = ""
            f.write("{} Epoch loss: {} {}\n".format(epoch, report_loss_val / report_len_val, improved_str))
        elif report_len_val == 0 and epoch == self.config.epochs-1:
             self.saver.save(sess=self.sess, save_path=self.save_path)
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
            total += 1
            counter += 1
        results[source][abbr] = [score, total]
        return results
