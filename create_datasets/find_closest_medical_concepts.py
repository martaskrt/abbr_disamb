import argparse
import fastText
import numpy as np
import ast, re, string, os
import pickle

MODEL_PATH = "/hpf/projects/brudno/marta/i2b2/word_embeddings_joined_20190708.bin"
fasttext_model = fastText.load_model(MODEL_PATH)
ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


def clean(word):
    word = word.lower()
    word = re.sub(r'%2[72]', "", word)
    word = re.sub(r'%26', "&", word)
    word = re.sub(r'%\w\w', " ", word)
    word = re.sub(r'[0-9]+', " N ", word)
    reduced_n = re.sub(r'(?<!\w)N( N)+', 'N', word)
    remove_apos = re.sub(r"'", '', reduced_n)
    word = re.sub(r'&', ' and ', remove_apos)
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    word = word.translate(translator)
    word = ' '.join(word.split())
    return word


def load_files(args):
    with open(args.abbr_exp_map, 'rb') as f_handle:
        abbr_exp_dict = pickle.load(f_handle)
    abbr_list = sorted(list(abbr_exp_dict.keys()))
    
    file = "/hpf/projects/brudno/marta/i2b2/medical_concepts_umls_in_mimic_cleaned_20190909.txt"
    with open(file) as file_handle:
        medical_concepts = [line[:-1] for line in file_handle if line[:-1] not in ALPHABET]

    job_id = args.id
    chunk = int(len(abbr_list)//args.num_jobs)
    start = int(chunk*(job_id-1))
    end = int(start + chunk)
    if job_id == args.num_jobs:
        end = len(abbr_list)
    print(start, end)
    abbrs_to_get = abbr_list[start:end]
    expansions = {}
    for abbr in abbrs_to_get:
        expansions[abbr] = abbr_exp_dict[abbr]
    return expansions, medical_concepts


def get_closest_concepts(sense, medical_concepts, max_dist):
    joined_sense = "_".join(sense.split())
    v1 = fasttext_model.get_word_vector(joined_sense)
    dist_to_sense = []
    for medical_concept in medical_concepts:
        medical_concept = "_".join(medical_concept.split())
        if medical_concept == joined_sense:
            continue
        v2 = fasttext_model.get_word_vector(medical_concept)
        dist = np.linalg.norm(v1-v2)
        dist_to_sense.append((dist, medical_concept))
    
    #return sorted([item for item in dist_to_sense if item[0] <= max_dist])
    return sorted(dist_to_sense)[:10]
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', required=True, type=int, help="starting index of mimic sentences to join;"
                        "e.g.: '1'")

    parser.add_argument('--max_dist', default=5, type=int)
    parser.add_argument('--abbr_exp_map', required=True)
    parser.add_argument('--num_jobs', required=True, type=int)
    args = parser.parse_args()

    expansions, medical_concepts = load_files(args)
    target_file = "{}.txt".format("closest_medical_concepts")
    with open(target_file, 'a') as file_handle:
        file_handle.write("\t{}\t{}\n".format("closest_med_concept_in_mimic", "dist_to_medical_concept_in_mimic"))
        for abbr in expansions:
            if abbr != "cN" and abbr != "tN": continue
            file_handle.write("abbr:::{}\n".format(abbr))
            for expansion in expansions[abbr]:
                file_handle.write("abbr:::{},exp:::{}\n".format(abbr,expansion))
                closest_med_concepts = get_closest_concepts(expansion, medical_concepts, args.max_dist)
                for i in range(len(closest_med_concepts)):
                    file_handle.write("\t{}\t{}\n".format(closest_med_concepts[i][1], closest_med_concepts[i][0]))

if __name__ == "__main__":
    main()
