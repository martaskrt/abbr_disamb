import os
import pickle
import argparse

#rootdir = "/Volumes/terminator/hpf/mimic_rs_umls_sentences/"
#target_dir = "/Volumes/terminator/hpf/rs_sorted_alpha"

alpha_map = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm',
             14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y',
             26: 'z', 27: 'num'}


def remove_rs_duplicate_sentences(opt):
    job_id = int(opt.id)
    src_dir = "/hpf/projects/brudno/marta/mimic_rs_collection/rs_sorted_alpha/" + str(alpha_map[job_id])
    dest_dir = "/hpf/projects/brudno/marta/mimic_rs_collection/rs_sorted_alpha_cleaned/" + str(alpha_map[job_id])
    #src_dir = "/Volumes/terminator/hpf/rs_sorted_alpha/" + str(alpha_map[job_id]) +'/'
    #dest_dir = "/Volumes/terminator/hpf/rs_sorted_alpha_cleaned/" + str(alpha_map[job_id]) +'/'
    for subdir, dirs, files in os.walk(src_dir):
        for file in files:
            print("Src: " + str(os.path.join(subdir, file)))
            unique_sentences = set()
            try:
                f = open(os.path.join(subdir, file), 'r')
            except:
                continue

            for line in f:
                line = line[:-1]
                unique_sentences.add(line)
            f.close()
            #file = "_".join(file.split())
            print("Dest: " + str(os.path.join(dest_dir, file)))
            g = open(os.path.join(dest_dir, file), 'w')
            for item in unique_sentences:
                g.write(item + '\n')
            g.close()

            print("Finished file: " + str(file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', required=True, help="starting index of umls terms to find in mimic;"
                                                    "e.g.: '1'")

    opt = parser.parse_args()
    remove_rs_duplicate_sentences(opt)
    print("Done deleting duplicate rs sentences! \U0001F436")

if __name__ == "__main__":
    main()
