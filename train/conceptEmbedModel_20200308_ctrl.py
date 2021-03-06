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
#import fastText
import re
#from tqdm import tqdm
from abbrrep_class import AbbrRep
import math
import scipy.sparse
from timeit import timeit
import os

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

            #content = phrase.split("|")
            ############## 2020-02-16#############
            content = phrase.split("|")[:-1]
            assert len(content) == 7
            #####################################

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

    def __init__(self, config, data_train, word_model, casi_test, exp2id, id2exp):
        # ************** Initialize variables ***************** #
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
        self.temperature = self.config.temperature

        self.ckpt_dir = '{}/checkpoints/'.format(self.config.output)
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.save_dir = '{}/{}/'.format(self.ckpt_dir, self.abbr)
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        self.save_path = os.path.join(self.save_dir, '{}'.format(self.abbr))
        
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
        for i in list(self.id2exp.keys()):
            key = self.id2exp[i]
            prob_data[key] = {} 
            sigmoid_denom[key] = 0
            denom_present = False
            if key in train_data and len(train_data[key]) > 0:
                sigmoid_denom[key] = np.exp(-self.epsilon/self.temperature)
                denom_present = True
            if config.use_relatives:
                for exp_pair in self.data_train[key]['relative']:
                    distance = exp_pair[1]
                    sigmoid_denom[key] += np.exp(-distance/self.temperature)
                    denom_present = True
            if key not in self.cui_probs:
                self.cui_probs[key] = {}
            if key in train_data and key in sigmoid_denom and  sigmoid_denom[key] > 0:
            #if key in train_data and key in sigmoid_denom and denom_present:
                confidence = np.exp(-self.epsilon/self.temperature)/sigmoid_denom[key]
                #confidence = tf.exp(-self.epsilon/self.temperature)/sigmoid_denom[key]
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
                for key in sorted(self.data_train[expansion]['relative']):
                    relative = key[0]
                    distance_to_exp = key[1]
                    confidence = np.exp(-distance_to_exp/self.temperature) / sigmoid_denom[expansion]
                    #curr_relative_samples = shuffle(self.data_train[expansion]['relative'][key], random_state=SEED)
                    prob_data[expansion][relative] = [distance_to_exp, confidence]
                    self.cui_probs[expansion][relative] = confidence
#####################
        sampled_trainining_data ={}
        for i in list(self.id2exp.keys()):
            key = self.id2exp[i]
            sampled_trainining_data[key] = []
            if len(self.data_train[key]['expansion']) == 0 and not self.config.use_relatives:
                continue
        
            np.random.seed(SEED)
            cuis = []
            cuis_probs = []
            prob_sum = 0
            for cui in self.cui_probs[key]:
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
            for relative in sampling_count[key]:
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

        print(self.abbr.upper())
        for key in training_data_stats:
            if key in num_samples_:
                print("{}\t{}".format(key, num_samples_[key]))
            else:
                print("{}\t{}".format(key, 0))
            for label in training_data_stats[key]:
                if label == key:
                    if key in sigmoid_denom: 
                        print("\t{}\t{}\t{:.20f}\t{}".format(label, self.epsilon, np.exp(-self.epsilon)/sigmoid_denom[key], training_data_stats[key][label]))
                    else:
                        print("\t{}\t{}\t{}\t{}".format(label, self.epsilon, 1, training_data_stats[key][label]))
                    
                else:
                    try:
                        print("\t{}\t{:.5f}\t{:.20f}\t{}".format(label, prob_data[key][label][0], prob_data[key][label][1], training_data_stats[key][label]))
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
        end = timeit()
        print(start-end)
        #for i in range(10):
         #   print(self.training_samples['seq'][i][0][:5])
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

        # ************** Encoder for sentence embeddings ***************** #
        #layer1 = tf.layers.conv1d(self.seq, self.config.cl1, 1, activation=tf.nn.elu, \
         #                         kernel_initializer=tf.random_normal_initializer(0.0, 0.1, seed=SEED), \
          #                        bias_initializer=tf.random_normal_initializer(stddev=0.01, seed=SEED), use_bias=True)
        layer1 = tf.layers.conv1d(self.seq, 100, 1, activation=tf.nn.elu, \
                                  kernel_initializer=tf.initializers.he_normal(seed=SEED), \
                                  bias_initializer=tf.initializers.he_normal(seed=SEED), use_bias=True)
        #layer2 = tf.layers.dense(tf.reduce_max(layer1, [1]), self.config.cl2, activation=tf.nn.relu, \
         #                        kernel_initializer=tf.random_normal_initializer(0.0, stddev=0.1, seed=SEED),
          #                       bias_initializer=tf.random_normal_initializer(0.0, stddev=0.01, seed=SEED),
           #                      use_bias=True)
        #red_max = tf.reduce_max(layer1, [1])
        
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
        #self.embeddings = tf.get_variable("embeddings", shape=[self.config.concepts_size, self.config.cl2],
         #                                 initializer=tf.random_normal_initializer(stddev=0.1, seed=SEED))
        self.embeddings = tf.get_variable("embeddings", shape=[self.config.concepts_size, 50],
                                          initializer=tf.random_normal_initializer(stddev=1e-1, seed=SEED))
        aggregated_w = self.embeddings

        last_layer_b = tf.get_variable('last_layer_bias', shape=[self.config.concepts_size],
                                       initializer=tf.random_normal_initializer(stddev=1e-1, seed=SEED))
        #last_layer_b = tf.get_variable('last_layer_bias', shape=[self.config.concepts_size],
         #                              initializer=tf.initializers.he_normal(seed=SEED))
        self.score_layer = tf.matmul(self.seq_embedding, tf.transpose(aggregated_w)) + last_layer_b
        #self.score_layer = tf.matmul(layer2, tf.transpose(aggregated_w)) + last_layer_b
        # ************** Loss ***************** #
        self.loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(self.label,
                                                                          self.score_layer))  # + reg_constant*tf.reduce_sum(reg_losses)

        self.pred = tf.nn.softmax(self.score_layer)


        # ************** Backprop ***************** #
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        #self.train_step = tf.train.MomentumOptimizer(self.lr, 0.9).minimize(self.loss)
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
        #print("A")
        #for i in range(5):
         #   print(self.training_samples['seq'][i][0][:5])
        shuffled_indecies = shuffle(list(range(training_size)), random_state=SEED)
        random.shuffle(shuffled_indecies)
        #print("B2")
        #print(shuffled_indecies[:10])
        #print("B")
        #for i in range(5):
         #   print(self.training_samples['seq'][i][0][:5])
        if self.config.batch_size == 100:
            self.config.batch_size = training_size
        while head < training_size:
            ending = min(training_size, head + self.config.batch_size)
            batch = {}
            for cat in self.training_samples:
                batch[cat] = self.training_samples[cat][shuffled_indecies[head:ending]]
            #print("C")
            #for i in range(5):
             #   print(batch['seq'][i][0][:5])
            #import sys
            #sys.exit(0)
            report_len_train += ending - head
            head += self.config.batch_size
            batch_feed = {self.seq: batch['seq'], \
                          self.seq_len: batch['seq_len'], \
                          self.g: batch['g'],
                          self.label: batch['label'],
                          self.is_training: True}
            _, batch_loss = self.sess.run([self.train_step, self.loss], feed_dict=batch_feed)
            report_loss_train += batch_loss
        print("{} Epoch temp_val: {}".format(epoch, self.temperature))
        if report_len_train > 0:
            print(str(epoch) + " Epoch loss: " + str(report_loss_train / report_len_train))
            self.trainLoss_array.append(report_loss_train / report_len_train)
        report_loss_val = 0
        report_len_val = 0
        head = 0
        val_size = self.val_samples['seq'].shape[0]
        shuffled_indecies = shuffle(list(range(val_size)), random_state=SEED)
        
        #random.shuffle(shuffled_indecies)
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
            #print(str(epoch) + " Epoch loss: " + str(report_loss_val / report_len_val))
            self.valLoss_array.append(report_loss_val / report_len_val)
            if curr_val_loss < self.best_val_loss:
                self.best_val_loss = curr_val_loss
                self.best_epoch = epoch
                self.saver.save(sess=self.sess, save_path=self.save_path)
                improved_str = "*"
            else:
                improved_str = ""
            #print("{} Epoch loss: {} {}".format(epoch, curr_val_loss, improved_str))
            print("{} Epoch loss: {} {}".format(epoch, report_loss_val / report_len_val, improved_str))
    def get_probs(self, val_samples):
        seq, seq_len, g = self.phrase2vec(val_samples, self.config.max_sequence_length)
        querry_dict = {self.seq: seq, self.seq_len: seq_len, self.g: g, self.is_training: False}
        #        res_querry = self.sess.run(self.score_layer, feed_dict = querry_dict)
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

            res_tmp = self.get_probs(querry_subset) #!!!!!!!!!!!!!!!!!!!!!!!!!
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
  #      print(res_querry)
        #for s in range(len(data)):
         #   print(s)
          #  print(res_querry[s])
           # indecies_querry = np.argsort(-res_querry[s])
        #    tmp_res = []
         #   for i in indecies_querry:
          #      tmp_res.append((self.id2exp[i[0]], res_querry[s][i[0]], i[0]))
#
 #           closest_concepts.append(tmp_res)
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


            if abbr == "asa":
                if "acetylsalicyl" in pred or "aspirin" in pred:
                    label = self.exp2id["acetylsalicylic acid"]
                if "acetylsalicyl" in ground_truth_label or "aspirin" in ground_truth_label:
                    ground_truth = self.exp2id["acetylsalicylic acid"]
            elif abbr == "dm":
                if "diabetes" in pred:
                    label = self.exp2id["diabetes mellitus"]
                if "diabetes" in ground_truth_label:
                    ground_truth = self.exp2id["diabetes mellitus"]
                if "dexamethasone" in pred:
                    label = self.exp2id["dexamethasone"]
                if "dexamethasone" in ground_truth_label:
                    ground_truth = self.exp2id["dexamethasone"]
                if "dextromethorphan" in ground_truth_label:
                    ground_truth = self.exp2id["dextromethorphan"]
                if "dextromethorphan" in pred:
                    label = self.exp2id["dextromethorphan"]
            elif abbr == "bmp":
                if "bone" in pred:
                    label = self.exp2id["bone morphogenetic protein"]
                if "bone" in ground_truth_label:
                    ground_truth = self.exp2id["bone morphogenetic protein"]
            elif abbr == "bal":
                if "lavage" in pred:
                    label = self.exp2id["bronchoalveolar lavage"]
                if "lavage" in ground_truth_label:
                    ground_truth = self.exp2id["bronchoalveolar lavage"]
            # elif abbr == "pr":
            #     if "progesterone" in all_names:
            #         label = name2meta[abbr]["progesterone receptor"]
            #     if "progesterone" in data[counter].label:
            #         ground_truth = name2meta[abbr]["progesterone receptor"]

            if label == ground_truth:
                score += 1
            if self.config.verbose:
                if mimic:
                    print("mimic" + "|" + str(tmp_res[0]) + " |" + str(label) + "|" + pred + "|" + str(ground_truth)  + "|" + ground_truth_label)
                else:
                    print("casi" + "|" + str(tmp_res[0]) + " |" + str(label) + "|" + pred + "|" + str(ground_truth)  + "|" + ground_truth_label)
            total += 1
            counter += 1
        # print(abbr + " from " + source + ": CORRECT="+ str(score) + " TOTAL=" + str(total))
        results[source][abbr] = [score, total]
        # print(predictions)
        return results
#from sklearn.utils import shuffle
#import tensorflow as tf
#import numpy as np
#import random
#
#random.seed(SEED)
#import json
#import pickle
#import fastText
#import re
#from tqdm import tqdm
#from abbrrep_class import AbbrRep
#import math
#import scipy.sparse
#from timeit import timeit
#
#
#
#class ConceptEmbedModel():
#    def phrase2vec(self, phrase_list, max_length):
#        phrase_vec_list = []
#        phrase_seq_lengths = []
#        for phrase in phrase_list:
#            start_left = int(max(len(phrase.features_left) - (max_length / 2), 0))
#            tokens = phrase.features_left[start_left:]
#            end_right = int(min(len(phrase.features_right), (max_length / 2)))
#            tokens_right = phrase.features_right[:end_right]
#            tokens.extend(tokens_right)
#            phrase_vec_list.append(
#from sklearn.utils import shuffle
#import tensorflow as tf
#import numpy as np
#import random
#
#random.seed(SEED)
#import json
#import pickle
#import fastText
#import re
#from tqdm import tqdm
#from abbrrep_class import AbbrRep
#import math
#import scipy.sparse
#from timeit import timeit
#
#
#
#class ConceptEmbedModel():
#    def phrase2vec(self, phrase_list, max_length):
#        phrase_vec_list = []
#        phrase_seq_lengths = []
#        for phrase in phrase_list:
#            start_left = int(max(len(phrase.features_left) - (max_length / 2), 0))
#            tokens = phrase.features_left[start_left:]
#            end_right = int(min(len(phrase.features_right), (max_length / 2)))
#            tokens_right = phrase.features_right[:end_right]
#            tokens.extend(tokens_right)
#            phrase_vec_list.append(
#                [self.word_model.get_word_vector(tokens[i]) if i < len(tokens) else [0] * 100 for i in
#                 range(max_length)])
