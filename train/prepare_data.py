from sklearn.utils import shuffle
import numpy as np
import random
import ast
SEED=42
random.seed(SEED)
np.random.seed(SEED)
import json
import pickle
import re
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

#np.random.seed(SEED)
#np.random.RandomState(0)
class ConceptEmbedModel():
    def phrase2vec(self, phrase_list, max_length):
        phrase_vec_list = []
        phrase_seq_lengths = []
        global_context_list = []
        sep_idx = []
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
            start_left = int(max(len(features_left) - (max_length / 2), 0))
            tokens = features_left[start_left:]
            end_right = int(min(len(features_right), (max_length / 2)))
            tokens_right = features_right[:end_right]
            if self.config.sep:
                tokens.append('[SEP]')
            sep_idx.append(len(tokens))
            tokens.extend(tokens_right)
            phrase_vec_list.append(tokens)
        return phrase_vec_list, sep_idx

    def __init__(self, config, data_train, word_model, casi_test, exp2id, id2exp, temperature=1):
        # ************** Initialize variables ***************** #
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
            #if key in train_data and key in sigmoid_denom and denom_present:
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
        #np.random.seed(SEED)
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
        self.training_samples['seq'], self.training_samples["sep_idx"] = self.phrase2vec(training_samples,
                                                                                     self.config.max_sequence_length)
        self.training_samples['label'] = np.array(training_labels)

        self.val_samples = {}
        self.val_samples['seq'], self.val_samples["sep_idx"] = self.phrase2vec(val_samples,
                                                                               self.config.max_sequence_length)
        self.val_samples['label'] = np.array(val_labels)

