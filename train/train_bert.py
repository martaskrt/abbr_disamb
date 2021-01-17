"""
Code adapted from Aryan Arbabi, https://github.com/a-arbabi/NeuralCR
"""
SEED=42
import argparse
import pandas as pd
import prepare_data
import numpy as np
import os
import json
import pickle
from sklearn.utils import shuffle
from timeit import timeit
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1" or "2", "3";
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#from hyperopt import hp, tpe, fmin
#from functools import partial
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertModel, TFBertModel
from transformers import AdamW
from transformers import BertTokenizer
import torch
from torch import nn
from torch.utils.data import DataLoader
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("using device:::{}".format(device))
#import tensorflow
class CustomBERTModel(nn.Module):
#class CustomBERTModel():
    def __init__(self, num_classes, base_model, args):
          super(CustomBERTModel, self).__init__()
         # self.bert = BertModel.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
          self.bert = BertModel.from_pretrained(base_model)
          ### New layers:
          self.linear1 = nn.Sequential(nn.Linear(768, 256), nn.ReLU()) # add nonlinearity 
          self.linear2 = nn.Linear(256, num_classes) ## 3 is the number of classes in this example
          self.args = args
    def forward(self, ids, mask, labels, sep_idx):
          outputs = self.bert(
               ids, 
               attention_mask=mask)
          if self.args.sep:
              sep_idx = sep_idx.view(-1,1,1).expand(-1,1,768)
              linear1_output = self.linear1(torch.gather(outputs.last_hidden_state,1,sep_idx).view(-1,768)) ## extract abbr coken's embeddings
          else:
              linear1_output = self.linear1(outputs.last_hidden_state[:,0,:].view(-1,768)) ## extract the 1st token's embeddings
          #linear1_output = self.linear1(outputs.last_hidden_state.view(-1,768)) ## extract the 1st token's embeddings
          linear2_output = self.linear2(linear1_output)
          return linear2_output

def phrase2vec(phrase_list, max_length, source, exp2id, args):
    phrase_vec_list = []
    phrase_seq_lengths = []
    global_context_list = []
    labels = []
    sep_idx = []
    for phrase in phrase_list:
        total_weighting = 0
        content = phrase.split("|")[:-1]
        if source == "casi":
            assert len(content) == 5
        else:
            assert len(content) == 7
        if source == "mimic":
            label = content[2].replace(",", "")
        else:
            label = content[0].replace(",", "")
        label = exp2id[label]
        features_left = content[-2].split()
        features_right = content[-1].split()
        start_left = int(max(len(features_left) - (max_length / 2), 0))
        tokens = features_left[start_left:]
        end_right = int(min(len(features_right), (max_length / 2)))
        tokens_right = features_right[:end_right]
        if args.sep:
            tokens.append('[SEP]')
        sep_idx.append(len(tokens))
        tokens.extend(tokens_right)
        phrase_vec_list.append(tokens)
        labels.append(label)
    return phrase_vec_list, labels, torch.tensor(sep_idx)

def eval_testset(args, source, exp2id, test, loaded_model, tokenizer, tok_len):
    test, labels,sep_idx = phrase2vec(test, args.max_sequence_length, source, exp2id, args) 
    encoding = tokenizer(test, return_tensors='pt', padding=True, truncation=True, is_split_into_words=True, max_length=tok_len)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    labels = torch.tensor(labels)
    
   # for i in range(10):
    #    print(tokenizer.decode(input_ids[i]), sep_idx[i]); 
    loaded_model.eval()
    total = [1 for _ in labels]
    results = []
    with torch.no_grad():
        outputs = loaded_model(input_ids.to(device), attention_mask.to(device), labels=labels.to(device), sep_idx=sep_idx.to(device))
        if args.custom:
            preds = torch.argmax(outputs, dim=1)
        else:
            preds = torch.argmax(outputs.logits, dim=1)
        assert preds.shape == labels.shape, (preds.shape, labels.shape)
        score = (preds == labels.to(device)).float().sum().cpu().detach().numpy()
        for i in range(len(preds)):
            if preds[i] == labels[i]:
                results.append(1)
            else:
                results.append(0)
    return [results, labels.shape[0]]

def run_testset(args, temperature, exp2id, id2exp, tokenizer, mimic_test, casi_test, base_model, tok_len):
    restore_path = "{}/checkpoints/{}/{}_best_validation_{}".format(args.output, args.abbr, args.abbr, temperature)
    if args.custom:
        loaded_model = CustomBERTModel(len(id2exp), base_model, args).to(device)
        loaded_model.load_state_dict(torch.load(restore_path))
        loaded_model.eval()
    else:
        loaded_model = BertForSequenceClassification.from_pretrained(restore_path).to(device)
    casi_results = eval_testset(args, "casi", exp2id, casi_test, loaded_model, tokenizer, tok_len) 
    mimic_results = eval_testset(args, "mimic", exp2id, mimic_test, loaded_model, tokenizer, tok_len) 
    results = []
    for i in range(args.bootstrap_runs):
        correct_casi = int(np.random.choice(casi_results[0], size=casi_results[1], replace=True).sum())
        total_casi = casi_results[1]
        micro_acc_casi = correct_casi/total_casi

        correct_mimic = int(np.random.choice(mimic_results[0], size=mimic_results[1], replace=True).sum())       
        total_mimic = mimic_results[1]
        micro_acc_mimic = correct_mimic/total_mimic

        results.append([correct_casi, total_casi, micro_acc_casi, correct_mimic, total_mimic, micro_acc_mimic])
    assert len(results) == args.bootstrap_runs
    return results, casi_results, mimic_results

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
    #if not os.path.isdir(results_dir_path):
     #    os.makedirs(results_dir_path)
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
    parser.add_argument('--batch_size', type=int, help="batch_size", default=-1)
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
    parser.add_argument("--pretrain", default=None)
    parser.add_argument("--ctrl", action="store_true")
    parser.add_argument("--custom", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--clinbert", action="store_true")
    parser.add_argument("--sep", action="store_true")
    args = parser.parse_args()
    args.fasttext = "word_embeddings_joined_20190708.bin"

    args.datafile = "/hpf/projects/brudno/marta/mimic_rs_collection/casi_mimic_rs_dataset_20190723/" + args.abbr + "_rs_close_umls_terms_20190723.txt"

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
    with open(args.datafile) as file_handle:
        for line in file_handle:
            content = line[:-1].split("|")
            closest_exp = content[0].replace(",", "")
            if closest_exp == "mall of america moa"  or closest_exp == "diphtheria tetanusus" or closest_exp == "mistake ez pap" or closest_exp == "metarsophalangeal" or (args.abbr == "pr" and closest_exp == "pm"):
                continue
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
    
    all_exps = None

    random_seed = 42
    print(all_exps)     
    print(random_seed)   
    casi_test, exp2id, id2exp = load_data(args, all_exps)
    for key in rel2exp:
        closest_exp = rel2exp[key]
        closest_exp_id = exp2id[closest_exp]
        exp2id[key] = closest_exp_id
    with open(global_file_path, 'a') as f:
        f.write("exp2id:::" + str(exp2id) + '\nid2exp:::' + str(id2exp) + '\n')
        f.write("SEED:::{}\n".format(random_seed))
    results_dir_path = os.path.join(args.output, "val_results")
    if not os.path.isdir(results_dir_path):
         os.makedirs(results_dir_path)
    results_abbr_path = os.path.join(results_dir_path, args.abbr)
    
    if args.ctrl:
        if not os.path.isdir(results_abbr_path):
             os.makedirs(results_abbr_path)
        results_file_path = os.path.join(results_abbr_path, "{}_val_results_{}.txt".format(args.abbr, 1))
        ckpt_dir = '{}/checkpoints'.format(args.output)
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
        save_dir = os.path.join(ckpt_dir, args.abbr)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, '{}_best_validation_{}'.format(args.abbr, 1))
        x = prepare_data.ConceptEmbedModel(args,data_train, None, casi_test, exp2id, id2exp)
        train_data, val_data = x.training_samples, x.val_samples
       
        if args.clinbert:
            #base_model = "emilyalsentzer/Bio_ClinicalBERT"
            base_model = "./pretrained_bert_tf/biobert_pretrain_output_all_notes_150000"
        else:
            base_model = "bert-base-uncased"
        print("BASE_MODEL:::{}".format(base_model))
        if args.custom:
            model = CustomBERTModel(len(id2exp), base_model, args).to(device)
            for name, param in model.named_parameters():
                 if "bert" in name:
                     param.requires_grad = False
                 print(name, param.requires_grad)
            loss_fn = nn.CrossEntropyLoss()
          #  optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        else:
            model = BertForSequenceClassification.from_pretrained(base_model, num_labels=len(id2exp)).to(device)
            if args.finetune:
                for name, param in model.named_parameters():
                    if "bert" in name:
                        param.requires_grad = False
            #for name, param in model.named_parameters():
             #    print(name, param.requires_grad)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        tokenizer = BertTokenizer.from_pretrained(base_model)
        tok_len = args.max_sequence_length + 2
        if args.sep:
            tok_len += 1
        print("Max tokenizer length:::{}".format(tok_len))
        train_encoding = tokenizer(train_data['seq'], return_tensors='pt', padding=True, truncation=True, is_split_into_words=True, max_length=tok_len)
        train_input_ids = train_encoding['input_ids']
        train_attention_mask = train_encoding['attention_mask']
        train_labels = torch.tensor(train_data['label'])
        train_sep_idx = torch.tensor(train_data['sep_idx']) 
        train_dataset = []
        for i in range(train_labels.shape[0]):
            train_dataset.append({'input_ids':train_input_ids[i],
                                  'attention_mask':train_attention_mask[i],
                                  'label':train_labels[i],
                                  'sep_idx':train_sep_idx[i]})
        
        val_encoding = tokenizer(val_data['seq'], return_tensors='pt', padding=True, truncation=True, is_split_into_words=True, max_length=tok_len)

        val_input_ids = val_encoding['input_ids']
        val_attention_mask = val_encoding['attention_mask']
        val_labels = torch.tensor(val_data['label'])
        val_sep_idx = torch.tensor(val_data['sep_idx']) 

        #for i in range(50):
         #   print(tokenizer.decode(train_input_ids[i]), train_sep_idx[i]); 
        #import sys; sys.exit(0)
        param_dir = args.output
        abbr = args.abbr
        if args.batch_size == -1:
            args.batch_size = len(train_data['label'])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        val_loss_array = []
        train_loss_array = []
        best_val_loss, best_epoch = np.inf, 0
        z = open(results_file_path, 'w')
        for epoch in tqdm(range(args.epochs)):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                train_input_ids = batch['input_ids']
                train_attention_mask = batch['attention_mask']
                train_labels = torch.tensor(batch['label'])
                train_sep_idx = torch.tensor(batch['sep_idx']) 

                z.write("Epoch :: " + str(epoch) + '\n')
                model.train()
                outputs = model(train_input_ids.to(device), train_attention_mask.to(device), labels=train_labels.to(device), sep_idx=train_sep_idx.to(device))
                if args.custom:
                    loss = loss_fn(outputs, train_labels.to(device))
                else:
                    loss = outputs.loss
                loss.backward()
                curr_loss_train = loss.item()
                train_loss_array.append(curr_loss_train)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                outputs = model(val_input_ids.to(device), val_attention_mask.to(device), labels=val_labels.to(device), sep_idx=val_sep_idx.to(device))
                if args.custom:
                    loss = loss_fn(outputs, val_labels.to(device))
                    preds = torch.argmax(outputs, dim=1)
                else:
                    loss = outputs.loss
                    preds = torch.argmax(outputs.logits, dim=1)
              #  loss = outputs.loss
                curr_val_loss = loss.item()
                val_loss_array.append(curr_val_loss)
                assert preds.shape == val_labels.shape, (preds.shape, val_labels.shape)
                micro_acc_val = (preds == val_labels.to(device)).float().sum().cpu().detach().numpy()/val_labels.shape[0]      
                macro_acc_val = {}
                

            z.write(str(epoch) + " CORRECT VAL\t" + str(micro_acc_val * val_labels.shape[0]) + '\n')
            z.write(str(epoch) + " TOTAL VAL\t" + str(val_labels.shape[0]) + '\n')
            z.write(str(epoch) + " ACC VAL\t" + str(micro_acc_val) + '\n')
            
            if curr_val_loss < best_val_loss:
                best_val_loss = curr_val_loss
                best_epoch = epoch
                if args.custom:
                    torch.save(model.state_dict(), save_path)
                else:
                    model.save_pretrained(save_path)
                improved_str = "*"
            else:
                improved_str = ""

            z.write("{} Epoch loss: {} {}\n".format(epoch, curr_val_loss, improved_str))

            if epoch == args.epochs-1:
                z.write("TRAIN LOSS ARRAY:\t" + str(train_loss_array) + '\n')
                z.write("VAL LOSS ARRAY:\t" + str(val_loss_array) + '\n')
                if len(val_loss_array) == 0:
                    if args.custom:
                        torch.save(model.state_dict(), save_path)
                    else:
                        model.save_pretrained(save_path)
        z.write("best epoch/val_loss: {}/{}\n".format(best_epoch, best_val_loss))
        test_results, casi_results, mimic_results = run_testset(args, 1, exp2id, id2exp, tokenizer, x.test_mimic, casi_test, base_model, tok_len)
    else:
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
               rstate=np.random.RandomState(random_seed))
        
        best_temp = best['temperature']
        with open(global_file_path, 'a') as f:
            f.write("best_temp:::{}\n".format(best))
        model = conceptEmbedModel_encoder.ConceptEmbedModel(args, data_train, word_model, casi_test, exp2id, id2exp, best_temp)
        test_results, casi_results, mimic_results = run_testset.run_testset(args, exp2id, id2exp, casi_test, model.test_mimic, word_model, temperature=best_temp)
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
