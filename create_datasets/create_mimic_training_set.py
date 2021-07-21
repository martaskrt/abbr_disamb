
import argparse
import pickle
import os
import sys
import random
random.seed(50)

src_dir = '/hpf/projects/brudno/marta/mimic_rs_collection/rs_sorted_alpha_cleaned/'
dest_dir = '/hpf/projects/brudno/marta/mimic_rs_collection/cuis_rs_20190315/'

def load_terms():
#    src_file = '/hpf/projects/brudno/marta/mimic_rs_collection/closest_umls_terms_in_mimic_20190722_b.txt'
    src_file = '/hpf/projects/brudno/marta/mimic_rs_collection/all_allacronym_expansions/closest_medical_concepts_casi_20200602_new.txt'
    related_terms = {}
    # abbr = ''
    # exp = ''
    with open(src_file) as f:
        for line in f:
            if ':::' in line:
                header = line[:-1].split(",")
                if len(header) == 2:
                    flag = True
                    abbr = header[0].split(":::")[1]
                    exp = header[1].split(":::")[1]
                    if abbr == exp or exp == 'remove' or exp == 'in vitro fertilscreeization':
                        flag = False
                        continue
                    if abbr not in related_terms:
                        related_terms[abbr] = {}
                    related_terms[abbr][exp] = []
            else:
                content = line[:-1].split('\t')
                if len(content) > 2 and content[0] == '' and content[1] != 'closest_med_concept_in_mimic' and flag:
                    related_terms[abbr][exp].append([content[1], float(content[2])])
    return related_terms

def load_samples(exp_, args):
    root_dir = "/hpf/projects/brudno/marta/mimic_rs_collection/rs_sorted_alpha_cleaned"
    start_char = exp_[0]
    if start_char == "N":
        start_char = 'num'
    samples = []
    try:
        with open("{}/{}/{}".format(root_dir, start_char, exp_)) as exp_file:
            for line in exp_file:
                content = line.split("|")
                if content[0] != exp_:
                    continue
                samples.append(line[:-1])
    except:
        print("couldn't load file...............{}".format(exp_))
    random.shuffle(samples)
    num_samples = min(len(samples), args.max_samples)
    sampled_sample = samples[:num_samples]
    return sampled_sample

def generate_data(terms_dict, args):
    dest_dir = "/hpf/projects/brudno/marta/mimic_rs_collection/casi_mimic_rs_dataset_20190723"
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)
    for abbr in terms_dict:
        filename = '{}/{}_rs_close_umls_terms_20190723.txt'.format(dest_dir, abbr)
        with open(filename, 'w') as fhandle:
            for expansion in terms_dict[abbr]:
                exp_ = ' '.join(expansion.split("_"))
                terms = load_samples(exp_, args)
                redone_samples = ["{}|{}|{}\n".format(exp_, 0, i) for i in terms]
                for item in redone_samples:
                    fhandle.write(item)
                for relative in terms_dict[abbr][expansion]:
                    rel = ' '.join(relative[0].split("_"))
                    distance = relative[1]
                    terms = load_samples(rel, args)
                    redone_samples = ["{}|{}|{}\n".format(exp_, distance, i) for i in terms]
                    for item in redone_samples:
                        fhandle.write(item)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_samples', default=1000, type=int)

    args = parser.parse_args()
    terms_dict = load_terms()
    generate_data(terms_dict, args)
    print("Done extracting rs samples! \U0001F4AA \U00002600")

if __name__ == "__main__":
    main()
