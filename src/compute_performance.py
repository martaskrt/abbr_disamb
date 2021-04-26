import argparse
import os
import pandas as pd
import numpy as np
import ast

def compute_global_stats(dic):
    macro_acc = np.array([dic["abbr_acc"][key] for key in dic["abbr_acc"]]).mean()
    micro_acc = sum([dic["correct"][key] for key in dic["correct"]])/sum([dic["total"][key] for key in dic['total']])
    return macro_acc, micro_acc

def get_file_list(testdir):
    ext_dict = {"abbr_acc": {}, "correct": {}, "total": {}}
    rootdir = os.path.join(testdir, "test_results")
    for root, dirs, files in os.walk(rootdir):
        for filename in files:
            abbr = filename.split("_")[0]

            df = pd.read_csv(os.path.join(root, filename))
            ext_dict["abbr_acc"][abbr] = df["acc"].mean()
            ext_dict["correct"][abbr] = df["correct"].sum()
            ext_dict["total"][abbr] = df["total"].sum()
            
    return ext_dict

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="path to root dir of models to analyze")

    args = parser.parse_args()
    ext_dict = get_file_list(args.dir)
    print("DATASET:::{}".format(args.dir))
    print(ext_dict["abbr_acc"])
    print("len_dict:::{}".format(len(ext_dict["abbr_acc"])))

    macro_acc_ext, micro_acc_ext = compute_global_stats(ext_dict)

    print("MACRO ACC EXT:::{:.4f}, MICRO ACC EXT:::{:.4f}".format(macro_acc_ext, micro_acc_ext))
if __name__ == "__main__":
    main()
