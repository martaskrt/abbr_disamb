import os
import pickle
import argparse

rootdir = "/Volumes/terminator/hpf/mimic_rs_umls_sentences/"
target_dir = "/Volumes/terminator/hpf/rs_sorted_alpha"
rootdir = "/hpf/projects/brudno/marta/mimic_rs_collection/mimic_rs_umls_sentences/"
target_dir = "/hpf/projects/brudno/marta/mimic_rs_collection/rs_sorted_alpha"

#pickle_in = open(os.path.join(os.path.abspath(os.path.join('./', os.pardir)), "umls_name2id_20190307.pickle"), 'rb')
#name2id = pickle.load(pickle_in)
#pickle_in.close()

NUM_ID = 50
#NUM_FILES = 1000
NUM_FILES=50

def sort_rs_by_alphabet(opt):
    job_id = int(opt.id)
    chunk = int(NUM_FILES//NUM_ID)
    start = int(chunk*(job_id-1))
    end = int(start + chunk)
    if job_id == NUM_ID:
#        start= NUM_FILES-1
        end = NUM_FILES

    for i in range(start, end):
        term_dir = {}

        path = rootdir + str(i+1) + ".txt"
        print(path)
        try:
            f = open(path, 'r')
        except:
            continue
        for line in f:
            line = line[:-1]
            tokens = line.split("|")
            term = tokens[0]

            if term in term_dir:
                term_dir[term].add(line)
            else:
                term_dir[term] = set()
                term_dir[term].add(line)

        for key in term_dir:
            start_char = key[0]
            if start_char == "N":
                start_char = "num"
            char_dir = os.path.join(target_dir, start_char)
            term_file = os.path.join(char_dir, key)

            g = open(term_file, 'a')
            lines = term_dir[key]
            for line in lines:
                g.write(line + '\n')
            g.close()

    print("JOB: " + str(job_id) + " | starting file: " + str(start) + " | ending file: " + str(end-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', required=True, help="starting index of files to organize alphabetically;"
                                                    "e.g.: '1'")

    opt = parser.parse_args()
    sort_rs_by_alphabet(opt)
    print("Done writing rs samples to term files! \U0001F925 \U0001F436")

if __name__ == "__main__":
    main()





