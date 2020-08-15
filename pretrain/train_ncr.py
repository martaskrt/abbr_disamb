"""
Code adapted from Aryan Arbabi, https://github.com/a-arbabi/NeuralCR
"""

import argparse
import conceptEmbedModel_ncr_global as conceptEmbedModel_encoder
import numpy as np
import os
import json
import fasttext as fastText
import pickle
# import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from timeit import timeit
from datetime import datetime, date, time
import find_lca

def load_files(abbr):
    src_dir = "pickle_files"
    src_file = os.path.join(src_dir, "child2ancs_{}_d2.6.pickle".format(abbr))
    print("1")
    if not os.path.exists(src_file):
        find_lca.find_lca(abbr)
    with open(src_file, 'rb') as g:
        ancestor_dict = pickle.load(g)
    cui2id = {}
    id2cui = {} 
    counter = 0
    for cui in sorted(ancestor_dict.keys()):
        cui2id[cui] = counter
        id2cui[counter] = cui
        counter += 1 
 
    #f = open("mountpoint/cui2id_20190905.pickle", 'rb')
    #cui2id = pickle.load(f)
    #f.close()

    #f = open("mountpoint/id2cui_20190905.pickle", 'rb')
    #id2cui = pickle.load(f)
    #f.close()

    #f = open("mountpoint/child2ancs_20190905.pickle", 'rb')
    #ancestor_dict = pickle.load(f)
    #f.close()

    return cui2id, id2cui, ancestor_dict

def main():
    parser = argparse.ArgumentParser(description='Hello!')
    parser.add_argument('--datafile', help="address to the datafile(s) (split by ',' if multiple")
    parser.add_argument('--fasttext', help="address to the fasttext word vector file")
    parser.add_argument('--output', help="address to the directroy where the trained model will be stored")

    parser.add_argument('--lr', type=float, help="lr", default=1 / 512)
    parser.add_argument('--batch_size', type=int, help="batch_size", default=2048)
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--max_sequence_length', type=int, help="max_sequence_length", default=6)
    parser.add_argument('--epochs', type=int, help="epochs", default=100)
    parser.add_argument('--ns', type=int, help="numer of samples per expansion to train on", default=1000)
    parser.add_argument('--abbr', type=str, required=True)
    args = parser.parse_args()

    args.fasttext = "word_embeddings_joined_20190708.bin"
    word_model = fastText.load_model(args.fasttext)

    args.datafile  = "/hpf/projects/brudno/marta/mimic_rs_collection/cuis_rs_1000Samples_20190612"
    print(args)

    cui2id, id2cui, ancestor_dict = load_files(args.abbr)
   
    ######### THIS IS A BOTTLENECK ############
    data_train = {}
    for cui in cui2id:
        path = os.path.join(args.datafile, cui + "_1000.txt")
        if os.path.exists(path):
            with open(path) as file_handle:
               curr_exp_data = [line[:-1] for line in file_handle]
               num_samples = min(args.ns, len(curr_exp_data))
               data_train[cui] = shuffle(curr_exp_data, random_state=42)[:num_samples]
    ############################################

    model = conceptEmbedModel_encoder.ConceptEmbedModel(args, data_train, word_model, cui2id, id2cui, ancestor_dict)


    param_dir = args.output
    abbr = args.abbr 
    for epoch in tqdm(range(args.epochs)):
        print("Epoch :: " + str(epoch))
        model.train_epoch(epoch)

        if epoch == args.epochs-1:
            print("TRAIN LOSS ARRAY:\t" + str(model.trainLoss_array))
            print("VAL LOSS ARRAY:\t" + str(model.valLoss_array))
    #model.writer.close()
if __name__ == "__main__":
    main()
