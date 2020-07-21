"""
Code adapted from Aryan Arbabi, https://github.com/a-arbabi/NeuralCR
"""
SEED=42
import argparse
import conceptEmbedModel_20200308_bayesopt as conceptEmbedModel_encoder
import pandas as pd
import numpy as np
import os
import json
import fasttext as fastText
import pickle
#import tensorflow as tf
#from tqdm import tqdm
#import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from timeit import timeit
#import run_testset_20200308_ctrl as run_testset
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1" or "2", "3";
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

def load_data(opt, all_exps=None):
    import math
    #src_dir = "mountpoint/i2b2_testsamples_20200306/"
    #fname = opt.abbr + "_i2b2_20200306.txt"
    src_dir = "/hpf/projects/brudno/marta/abbr_disamb/datasets/casi_sentences_reformatted_20190319/"
    fname = opt.abbr + "_casi_20190319.txt"
    with open(os.path.join(src_dir, fname), encoding="utf-8") as casi_data:
        casi_keys = set()
        casi_test = {}
        for line in casi_data:
            content = line[:-1].split("|")
            exp = content[0].replace(",", "")
            if opt.abbr == exp or exp == 'remove' or exp == 'in vitro fertilscreeization' or ":" in exp or exp == "mall of america moa" or exp == "diphtheria tetanusus" or exp == "mistake ez pap" or exp == "metarsophalangeal" or (opt.abbr == "pr" and exp == "pm"):
                continue
            casi_keys.add(exp)
            if exp not in casi_test:
                casi_test[exp] = set()
            casi_test[exp].add(line[:-1])
      
    casi_test_list = []
    for exp in casi_test:
        casi_test_list.extend(casi_test[exp])
    if all_exps:
        for item in all_exps:
            casi_keys.add(item)
    exp2id, id2exp = make_id_map(casi_keys)
    return casi_test_list, exp2id, id2exp


def train_model(temperature, data):
    print(temperature)
    args = data['args']
    data_train = data['data_train']
    word_model = data['word_model']
    data_test = data['data_test']
    exp2id = data['exp2id']
    id2exp = data['id2exp']

    model = conceptEmbedModel_encoder.ConceptEmbedModel(args, data_train, word_model, data_test, exp2id, id2exp, temperature)
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
    print(model.best_val_loss)
    return model.best_val_loss

def main():
    parser = argparse.ArgumentParser(description='Hello!')
    parser.add_argument('--datafile', help="address to the datafile(s) (split by ',' if multiple")
    parser.add_argument('--fasttext', help="address to the fasttext word vector file")
    parser.add_argument('--output', help="address to the directroy where the trained model will be stored")

    parser.add_argument('--replace', action="store_true")
    parser.add_argument('--globalcontext', action="store_true")
    parser.add_argument('--lr', type=float, help="lr", default=0.01)
    parser.add_argument('--batch_size', type=int, help="batch_size", default=100)
    parser.add_argument('--verbose', action="store_true")
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
    args = parser.parse_args()
    args.fasttext = "word_embeddings_joined_20190708.bin"
    word_model = fastText.load_model(args.fasttext)

    args.datafile = "/hpf/projects/brudno/marta/mimic_rs_collection/casi_mimic_rs_dataset_20190723/" + args.abbr + "_rs_close_umls_terms_20190723.txt"

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    global_dir_path = os.path.join(args.output, "global_info")
    if not os.path.isdir(global_dir_path):
         os.makedirs(global_dir_path)
    global_file_path = os.path.join(global_dir_path, "{}_global_info.txt".format(args.abbr))
    with open(global_file_path, 'w') as f:
        f.write(str(args) + '\n')

    #print("All files loaded! Number of samples: " + str(len(data_train)))
    #casi_test, exp2id, id2exp = load_data(args)
   
    print("ABBREVIATION BEING MODELLED: " + args.abbr)
    data_train = {}
    rel2exp = {}
    with open(args.datafile) as file_handle:
        for line in file_handle:
            content = line[:-1].split("|")
            closest_exp = content[0].replace(",", "")
            if closest_exp == "mall of america moa"  or closest_exp == "diphtheria tetanusus" or closest_exp == "mistake ez pap" or closest_exp == "metarsophalangeal" or (args.abbr == "pr" and closest_exp == "pm"):
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
    
    all_exps = None
    if args.train_full:
        all_exps = set()
        all_acronyms_path = "/hpf/projects/brudno/marta/mimic_rs_collection/all_allacronym_expansions/allacronyms_training_mimic_rs_dataset_20191105"
        all_acronyms_file = "{}_rs_close_umls_terms_20190910.txt".format(args.abbr)
        with open("/hpf/projects/brudno/marta/mimic_rs_collection/all_allacronym_expansions/merge_expansions_20191031_final.pickle", 'rb') as pickle_hande:
            all_abbr2exp = pickle.load(pickle_handle)
        with open(os.path.join(all_acronyms_path, all_acronyms_file)) as z:
            for line in z:
                content = line[:-1].split("|")
                closest_exp = content[0].replace(",", "")
                if closest_exp not in all_abbr2exp:
                    continue
                if closest_exp in data_train:
                    continue
                all_exps.add(closest_exp)
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
                
    casi_test, exp2id, id2exp = load_data(args, all_exps)
    #for closest_exp in sorted(data_train):
     #   for exp in sorted(data_train[closest_exp]['relative']):
      #      data_train[closest_exp]['relative'][exp] = shuffle(data_train[closest_exp]['relative'][exp], random_state=SEED)
    for key in rel2exp:
        closest_exp = rel2exp[key]
        closest_exp_id = exp2id[closest_exp]
        exp2id[key] = closest_exp_id
    with open(global_file_path, 'a') as f:
        f.write("exp2id:::" + str(exp2id) + '\nid2exp:::' + str(id2exp) + '\n')
    data = {'args': args,
            'data_train': data_train,
            'data_test': casi_test,
            'word_model': word_model,
            'exp2id': exp2id,
            'id2exp': id2exp}
    fmin_objective = partial(train_model, data=data)
    best = fmin(fn=fmin_objective,
           space=hp.uniform('temperature', args.temp_lb, args.temp_ub),
           algo=tpe.suggest,
           max_evals=25,
           show_progressbar=True,
           verbose=True,
           rstate=np.random.RandomState(SEED))
    
    best_temp = best['temperature']
    with open(global_file_path, 'a') as f:
        f.write("best_temp:::{}\n".format(best))
    model = conceptEmbedModel_encoder.ConceptEmbedModel(args, data_train, word_model, casi_test, exp2id, id2exp, best_temp)
    test_results, casi_results, mimic_results = run_testset.run_testset(args, exp2id, id2exp, casi_test, model.test_mimic, word_model, temperature=best_temp)
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
