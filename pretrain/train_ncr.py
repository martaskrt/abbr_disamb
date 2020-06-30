"""
Code adapted from Aryan Arbabi, https://github.com/a-arbabi/NeuralCR
"""

import argparse
import conceptEmbedModel_ncr as conceptEmbedModel_encoder
import numpy as np
import os
import json
import fastText
import pickle
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from timeit import timeit
from datetime import datetime, date, time

def load_files():
    f = open("mountpoint/cui2id_20190905.pickle", 'rb')
    cui2id = pickle.load(f)
    f.close()

    f = open("mountpoint/id2cui_20190905.pickle", 'rb')
    id2cui = pickle.load(f)
    f.close()

    f = open("mountpoint/child2ancs_20190905.pickle", 'rb')
    ancestor_dict = pickle.load(f)
    f.close()

    return cui2id, id2cui, ancestor_dict

def main():
    parser = argparse.ArgumentParser(description='Hello!')
    parser.add_argument('--datafile', help="address to the datafile(s) (split by ',' if multiple")
    parser.add_argument('--fasttext', help="address to the fasttext word vector file")
    parser.add_argument('--output', help="address to the directroy where the trained model will be stored")

    parser.add_argument('--phrase_val', help="address to the file containing labeled phrases for validation")
    parser.add_argument('--replace', action="store_true")
    parser.add_argument('--flat', action="store_true")
    parser.add_argument('--cl1', type=int, help="cl1", default=1024)
    parser.add_argument('--cl2', type=int, help="cl2", default=1024)
    parser.add_argument('--lr', type=float, help="lr", default=1 / 512)
    parser.add_argument('--batch_size', type=int, help="batch_size", default=64)
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--max_sequence_length', type=int, help="max_sequence_length", default=16)
    parser.add_argument('--epochs', type=int, help="epochs", default=25)
    parser.add_argument('--ns', type=int, help="numer of samples per expansion to train on", default=500)
    args = parser.parse_args()

    args.fasttext = "word_embeddings_joined_20190708.bin"
    word_model = fastText.load_model(args.fasttext)

    args.datafile = "mountpoint/cuis_rs_1000Samples_20190612/"

    print(args)

    cui2id, id2cui, ancestor_dict = load_files()
   
    ######### THIS IS A BOTTLENECK ############
    data_train = {}
    for subdir, dirs, files in os.walk(args.datafile):
        for file in files:
            if file.lower()[-4:] == ".txt":
                exp = file.split("_")[0]
                with open(os.path.join(subdir, file)) as file_handle:
                   curr_exp_data = [line[:-1] for line in file_handle]
                   num_samples = min(args.ns, len(curr_exp_data))
                   data_train[exp] = shuffle(curr_exp_data, random_state=42)[:num_samples]
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
