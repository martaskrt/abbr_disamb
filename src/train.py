SEED=42
import argparse
import conceptEmbedModel as conceptEmbedModel_encoder
import pandas as pd
import numpy as np
import os
import json
import fasttext as fastText
import pickle
import tensorflow as tf
from sklearn.utils import shuffle
from timeit import timeit
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from hyperopt import hp, tpe, fmin
from functools import partial

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
    with open("abbr2exps.txt") as f:
        for line in f:
            abbr, exps = line[:-1].split(":::")
            if abbr == opt.abbr:
                exps = exps.split(",")
                break
    exp2id, id2exp = make_id_map(exps)
    return exp2id, id2exp


def train_model(temperature, data):
    args = data['args']
    data_train = data['data_train']
    word_model = data['word_model']
    exp2id = data['exp2id']
    id2exp = data['id2exp']

    model = conceptEmbedModel_encoder.ConceptEmbedModel(args, data_train, word_model, exp2id, id2exp, temperature)
    results_dir_path = os.path.join(args.output, "val_results")
    if not os.path.isdir(results_dir_path):
         os.makedirs(results_dir_path)
    results_abbr_path = os.path.join(results_dir_path, args.abbr)
    if not os.path.isdir(results_abbr_path):
         os.makedirs(results_abbr_path)
    results_file_path = os.path.join(results_abbr_path, "{}_val_results_{}.txt".format(args.abbr, temperature))
    z = open(results_file_path, 'w')
    abbr = args.abbr 
    for epoch in range(args.epochs):
        z.write("Epoch :: " + str(epoch) + '\n')
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
        
        
        z.write(str(epoch) + " CORRECT VAL\t" + str(score_val) + '\n')
        z.write(str(epoch) + " TOTAL VAL\t" + str(total_val) + '\n')
        z.write(str(epoch) + " ACC VAL\t" + str(micro_acc_val) + '\n')

        if epoch == args.epochs-1:
            z.write("TRAIN LOSS ARRAY:\t" + str(model.trainLoss_array) + '\n')
            z.write("VAL LOSS ARRAY:\t" + str(model.valLoss_array) + '\n')
    return model.best_val_loss

def main():
    parser = argparse.ArgumentParser(description='Hello!')
    parser.add_argument('--fasttext', help="address to the fasttext word vector file")
    parser.add_argument('--output', help="address to the directroy where the trained model will be stored")

    parser.add_argument('--replace', action="store_true")
    parser.add_argument('--globalcontext', action="store_true")
    parser.add_argument('--lr', type=float, help="lr", default=0.01)
    parser.add_argument('--batch_size', type=int, help="batch_size (use -1 if training on all data)", default=-1)
    parser.add_argument('--max_sequence_length', type=int, help="max_sequence_length", default=6)
    parser.add_argument('--use_relatives', action="store_true")
    parser.add_argument('--epochs', type=int, help="epochs", default=100)
    parser.add_argument('--ns', type=int, help="numer of samples per expansion to train on", default=1000)
    parser.add_argument('--abbr', help="abbreviation being modelled")
    parser.add_argument('--temp_lb', type=float, default=0.5)
    parser.add_argument('--temp_ub', type=float, default=2.0)
    parser.add_argument('--epsilon', type=float, help="epsilon value", default=0.001)
    parser.add_argument('--bootstrap_runs', type=int, default=999)
    parser.add_argument('--train_full', action="store_true")
    parser.add_argument("--pretrain", default=None)
    parser.add_argument("--ctrl", action="store_true")
    parser.add_argument("--train_file", required=True)
    args = parser.parse_args()
    args.fasttext = "./fasttext_word_embeddings.bin"
    word_model = fastText.load_model(args.fasttext)


    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    global_dir_path = os.path.join(args.output, "global_info")
    if not os.path.isdir(global_dir_path):
         os.makedirs(global_dir_path)
    global_file_path = os.path.join(global_dir_path, "{}_global_info.txt".format(args.abbr))
    with open(global_file_path, 'w') as f:
        f.write(str(args) + '\n')

   
    print("ABBREVIATION BEING MODELLED: " + args.abbr)
    data_train = {}
    rel2exp = {}
    exps_present = set()
    with open(args.train_file) as file_handle:
        for line in file_handle:
            content = line[:-1].split("|")
            closest_exp = content[0].replace(",", "")
            distance = float(content[1])
            relative = content[2]
            rel2exp[relative] = closest_exp
            exps_present.add(closest_exp)
            if closest_exp not in data_train:
                data_train[closest_exp] = {'expansion': [], 'relative': {}}
            if closest_exp == relative:
                data_train[closest_exp]['expansion'].append(line[:-1])
            else:
                if (relative, distance) not in data_train[closest_exp]['relative']:
                    data_train[closest_exp]['relative'][(relative,distance)] = []
                data_train[closest_exp]['relative'][(relative,distance)].append(line[:-1])
    
    random_seed = 4247596345
    #random_seed = 42
    exp2id, id2exp = load_data(args)
    for key in rel2exp:
        closest_exp = rel2exp[key]
        closest_exp_id = exp2id[closest_exp]
        exp2id[key] = closest_exp_id
    with open(global_file_path, 'a') as f:
        f.write("exp2id:::" + str(exp2id) + '\nid2exp:::' + str(id2exp) + '\n')
        f.write("SEED:::{}\n".format(random_seed))
    data = {'args': args,
            'data_train': data_train,
            'word_model': word_model,
            'exp2id': exp2id,
            'id2exp': id2exp}
    if args.ctrl:
        train_model(1, data)
    else:
        fmin_objective = partial(train_model, data=data)
        best = fmin(fn=fmin_objective,
               space=hp.uniform('temperature', args.temp_lb, args.temp_ub),
               algo=tpe.suggest,
               max_evals=25,
               show_progressbar=True,
               verbose=True,
               rstate=np.random.RandomState(random_seed))
        
        best_temp = best['temperature']
        with open(global_file_path, 'a') as f:
            f.write("best_temp:::{}\n".format(best))

if __name__ == "__main__":
    main()
