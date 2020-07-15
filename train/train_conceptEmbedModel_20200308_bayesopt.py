"""
Code adapted from Aryan Arbabi, https://github.com/a-arbabi/NeuralCR
"""
SEED=42
import argparse
# import conceptEmbedModel_encoder_replace as conceptEmbedModel_encoder
#import conceptEmbedModel_encoder_embedSpace_20190724_bootstrap as conceptEmbedModel_encoder
import conceptEmbedModel_20200308_ctrl as conceptEmbedModel_encoder
import pandas as pd
#import conceptEmbedModel_encoder_embedSpace_5 as conceptEmbedModel_encoder
#import conceptEmbedModel_encoder_embedSpace_20190801_temphyperparam_2 as conceptEmbedModel_encoder
#import conceptEmbedModel_encoder_2 as conceptEmbedModel_encoder
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
#import run_testset_bootstrap_global_ctrl as run_testset
import run_testset_20200308_ctrl as run_testset
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1" or "2", "3";
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from hyperopt import hp, tpe, fmin

def make_id_map(casi_keys):
    exp2id = {}
    id2exp = {}
    sorted_keys = sorted(list(casi_keys))
    counter = 0
    for item in sorted_keys:
        exp2id[item] = counter
        id2exp[counter] =item
        counter += 1
    return exp2id, id2exp

def load_data(opt):
    import math
    #src_dir = "mountpoint/i2b2_testsamples_20200306/"
    #fname = opt.abbr + "_i2b2_20200306.txt"
    src_dir = "casi_sentences_reformatted_20190319"
    fname = opt.abbr + "_casi_20100319.txt"
    with open(os.path.join(src_dir, fname), encoding="utf-8") as casi_data:
        casi_keys = set()
        casi_test = {}
        for line in casi_data:
            content = line[:-1].split("|")
            exp = content[0].replace(",", "")
            if opt.abbr == exp or exp == 'remove' or exp == 'in vitro fertilscreeization' or ":" in exp or exp == "mall of america moa":
                continue
            casi_keys.add(exp)
            if exp not in casi_test:
                casi_test[exp] = set()
            casi_test[exp].add(line[:-1])
      
    casi_test_list = []
    for exp in casi_test:
        casi_test_list.extend(casi_test[exp])
    exp2id, id2exp = make_id_map(casi_keys)
    return casi_test_list, exp2id, id2exp


def train_model(temperature, data):

    args = data['args']
    data_train = data['data_train']
    word_model = data['word_model']
    data_test = data['data_test']
    exp2id = data['exp2id']
    id2exp = data['id2exp']

    model = conceptEmbedModel_encoder.ConceptEmbedModel(args, data_train, word_model, casi_test, exp2id, id2exp)

    abbr = args.abbr 
    for epoch in range(args.epochs):
        print("Epoch :: " + str(epoch))
        model.train_epoch(epoch)
        results_val = model.label_data(model.val_mimic, abbr, source="mimic", mimic=True)

        micro_acc_val = 0
        total_val = 0
        score_val = 0

        for abbr in results_val['mimic']:
            correct = results_val['mimic'][abbr][0]
            total = results_val['mimic'][abbr][1]
            if total > 0:
                micro_acc_val += correct / total
            total_val += total
            score_val += correct
        micro_acc_val /= len(results_val['mimic'])
        macro_acc_val = 0
        if total_val > 0:
            macro_acc_val = score_val / total_val
        
        
        print(str(epoch) + " CORRECT VAL\t" + str(score_val))
        print(str(epoch) + " TOTAL VAL\t" + str(total_val))
        print(str(epoch) + " ACC VAL\t" + str(micro_acc_val))

        if epoch == args.epochs-1:
            print("TRAIN LOSS ARRAY:\t" + str(model.trainLoss_array))
            print("VAL LOSS ARRAY:\t" + str(model.valLoss_array))
    return model.best_val_loss

def main():
    parser = argparse.ArgumentParser(description='Hello!')
    parser.add_argument('--datafile', help="address to the datafile(s) (split by ',' if multiple")
    parser.add_argument('--fasttext', help="address to the fasttext word vector file")
    parser.add_argument('--output', help="address to the directroy where the trained model will be stored")

    parser.add_argument('--phrase_val', help="address to the file containing labeled phrases for validation")
    parser.add_argument('--replace', action="store_true")
    parser.add_argument('--globalcontext', action="store_true")
    parser.add_argument('--flat', action="store_true")
    parser.add_argument('--cl1', type=int, help="cl1", default=1024)
    parser.add_argument('--cl2', type=int, help="cl2", default=1024)
    parser.add_argument('--lr', type=float, help="lr", default=1 / 512)
    parser.add_argument('--batch_size', type=int, help="batch_size", default=64)
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--max_sequence_length', type=int, help="max_sequence_length", default=16)
    parser.add_argument('--use_relatives', action="store_true")
    parser.add_argument('--epochs', type=int, help="epochs", default=25)
    parser.add_argument('--ns', type=int, help="numer of samples per expansion to train on", default=500)
    parser.add_argument('--abbr', help="abbreviation being modelled")
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--epsilon', type=float, help="epsilon value", default=0.001)
    parser.add_argument('--distance', type=float, help="distance value", default=100)
    parser.add_argument('--bootstrap_runs', type=int, default=999)

    args = parser.parse_args()
    args.fasttext = "word_embeddings_joined_20190708.bin"
    word_model = fastText.load_model(args.fasttext)

    args.datafile = "casi_mimic_rs_dataset_20190723/" + args.abbr + "_rs_close_umls_terms_20190723.txt"

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    global_dir_path = os.path.join(args.output, "global_info")
    if not os.path.isdir(global_dir_path):
         os.makedirs(global_dir_path)
    global_file_path = os.path.join(global_dir_path, "{}_global_info.txt")
    with open(global_file_path, 'w') as f:
        f.write(args + '\n')

    #print("All files loaded! Number of samples: " + str(len(data_train)))
    casi_test, exp2id, id2exp = load_data(args)
   
    print("ABBREVIATION BEING MODELLED: " + args.abbr)
    data_train = {}
    rel2exp = {}
    with open(args.datafile) as file_handle:
        for line in file_handle:
            content = line[:-1].split("|")
            closest_exp = content[0].replace(",", "")
            if closest_exp == "mall of america moa":
                continue
            distance = float(content[1])
            relative = content[2]
            rel2exp[relative] = closest_exp
            if closest_exp not in data_train:
                data_train[closest_exp] = {'expansion': [], 'relative': {}}
            if closest_exp == relative:
                data_train[closest_exp]['expansion'].append(line[:-1])
            else:
                if (relative, distance) not in data_train[closest_exp]['relative']:
                    data_train[closest_exp]['relative'][(relative,distance)] = []
                data_train[closest_exp]['relative'][(relative,distance)].append(line[:-1])
    for key in rel2exp:
        closest_exp = rel2exp[key]
        closest_exp_id = exp2id[closest_exp]
        exp2id[key] = closest_exp_id
    with open(global_file_path, 'a') as f:
        f.write(exp2id + '\n' + id2exp + '\n')
    data = {'args': args,
            'data_train': data_train,
            'data_test': data_test,
            'word_model': word_model,
            'exp2id': exp2id,
            'id2exp': id2exp}
    fmin_objective = partial(train_model, data=data)
    best = fmin(fn=fmin_objective,
           space=hp.uniform('temperature', args.temp_lb, args.temp_ub),
           algo=tpe.suggest,
           max_evals=args.max_evals,
           show_progressbar=True,
           verbose=True,
           rstate=np.random.RandomState(42))
    
    best_temp = best['temperature']
    with open(global_file_path, 'a') as f:
        f.write("best_temp:::{}\n".format(best))

    test_results, casi_results, mimic_results = run_testset.run_testset(args, exp2id, id2exp, casi_test, model.test_mimic, word_model)
    #model.writer.close()
    print("casi_test_results:::{}".format(casi_results))
    print("mimic_test_results:::{}".format(mimic_results))
    df = pd.DataFrame(data=test_results, columns=['correct_casi', 'total_casi', 'acc_casi', 'correct_mimic', 'total_mimic', 'acc_mimic'])
    save_dir = 'test_results'
    save_path = os.path.join(args.output, save_dir)
    if not os.path.isdir(save_path):
         os.makedirs(save_path)
    df.to_csv(os.path.join(save_path, "{}_testresults.csv".format(args.abbr)))
if __name__ == "__main__":
    main()
