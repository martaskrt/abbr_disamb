import argparse
import os

NUM_ID = 50
def get_term_array(input):
    terms = []
    f = open(input, 'r')
    for line in f:
        line = line[:-1]
        terms.append(line)
    f.close()
    return terms

def get_sentence(term_to_search, line, window):
    doc = line
    split_on_sense = doc.split(term_to_search)
    try:
        left_of_sense = split_on_sense[0].strip().split()
        right_of_sense = split_on_sense[1].strip().split()

        doc_left = ' '.join(left_of_sense)
        doc_whole = ' '.join(left_of_sense + right_of_sense)

        left_window_start = max(0, len(left_of_sense) - window)
        left_window_end = max(0, len(left_of_sense))
        local_sentence_left = left_of_sense[left_window_start:left_window_end]

        right_window_start = 0
        right_window_end = min(window, len(right_of_sense))
        local_sentence_right = right_of_sense[right_window_start:right_window_end]
        output = str(' '.join(term_to_search.split())) + '|' + str(doc_whole) + '|' + str(doc_left) + '|' +\
                 str(' '.join(local_sentence_left)) + '|' + str(' '.join(local_sentence_right)) + '|'


        return output

    except IndexError:
        return ""

def load_mimic(src):
    mimic_lines = []
    f = open(src, 'r')
    for line in f:
        mimic_lines.append(line[:-1])
    f.close()
    return mimic_lines

def find_lines(opt):
    
    src_file = "umls_terms.txt"
    umls_file_src = os.path.join(os.getcwd(), src_file)
    terms = get_term_array(umls_file_src)

    mimic_file_src = os.path.join(os.path.abspath(os.path.join('./', os.pardir)), "mimicnotes_cleaned.txt")
    mimic_lines = load_mimic(mimic_file_src)

    job_id = int(opt.id)
    chunk = int(len(terms)//NUM_ID)
    start = int(chunk*(job_id-1))
    end = int(start + chunk)
    if job_id == NUM_ID:
        print(chunk)
        end = len(terms)
        print(start, end)

    dest_file = "mimic_rs_umls_sentences/" + str(job_id) + ".txt"
    umls_file_dest = os.path.join(os.getcwd(), dest_file)
    g = open(umls_file_dest, 'a')

    print("JOB: " + str(job_id) + " | starting term: " + terms[start] + " | ending term: " + terms[end-1])
    for i in range(start, end):
        target_word = terms[i]
        try:
            start_char = target_word[0]
            if start_char == "N":
                start_char = "num"
            check_file = '../rs_sorted_alpha_cleaned/{}/{}'.format(start_char,target_word) 
            if os.path.isfile(check_file):
                continue
            ##
            for line in mimic_lines:
                present = False
                term_to_search = " " + target_word + " "
                if term_to_search in line:
                    present = True
                elif len(line) >= len(target_word) + 1:
                    # target word at beginning of sentence
                    if line[:len(target_word) + 1] == target_word + " ":
                        term_to_search = target_word + " "
                        present = True
                    # target word at end of sentence
                    elif line[len(line) - (len(target_word) + 1):] == " " + target_word:
                        term_to_search = " " + target_word
                        present = True
                if present:
                    mimic_sent = get_sentence(term_to_search, line, window=int(opt.window))
                    if mimic_sent != "":
                        g.write(str(mimic_sent) + '\n')

        except:
            print("problem with: {}".format(target_word))
    g.close()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', required=True, help="starting index of umls terms to find in mimic;"
                                                    "e.g.: '1'")
    parser.add_argument('-window', default=20, help="window of local context to get from mimic")
    opt = parser.parse_args()
    find_lines(opt)

if __name__ == "__main__":
    main()
