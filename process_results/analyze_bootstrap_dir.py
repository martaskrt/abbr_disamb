import argparse
import os
import pandas as pd
import numpy as np
#def get_best_temp(filename):
 #   best_temp = None
  #  with open(filename) as f:
   #     for line in f:
    #        if "best_temp" in f:
     #           best_temp = float(line[:-2].split("'temperature': ")[1])
    #return best_temp
#def get_best_temp_files(rootdir):
 #   abbr2temp = {}
  #  rootdir = os.path.join(rootdir, "global_info")
   # for root, dirs, files in os.walk(rootdir):
    #    for filename in files:
     #       if ".txt" in filename:
      #          best_temp = get_best_temp(os.path.join(root, filename))
       #         abbr = filename.split("_")[0]
        #        abbr2temp[abbr] = best_temp

    #results_dir = os.path.join(rootdir, "test_results")
def compute_global_stats(dic):
    macro_acc = np.array([dic["abbr_acc"][key] for key in dic["abbr_acc"]]).mean()
    micro_acc = sum([dic["correct"][key] for key in dic["correct"]])/sum([dic["total"][key] for key in dic['total']])
    return macro_acc, micro_acc

def get_file_list(rootdir):
    mimic_dict = {"abbr_acc": {}, "correct": {}, "total": {}}
    ext_dict = {"abbr_acc": {}, "correct": {}, "total": {}}
    rootdir = os.path.join(rootdir, "test_results")
    for root, dirs, files in os.walk(rootdir):
        for filename in files:
            abbr = filename.split("_")[0]
            if "i2b2" not in rootdir and abbr in ["or", "pm", "bk", "itp"]: continue
            df = pd.read_csv(os.path.join(root, filename))
            # ,correct_casi,total_casi,acc_casi,correct_mimic,total_mimic,acc_mimic,temperature
            ext_dict["abbr_acc"][abbr] = df["acc_casi"].mean()
            ext_dict["correct"][abbr] = df["correct_casi"].sum()
            ext_dict["total"][abbr] = df["total_casi"].sum()
            mimic_dict["abbr_acc"][abbr] = df["acc_mimic"].mean()
            mimic_dict["correct"][abbr] = df["correct_mimic"].sum()
            mimic_dict["total"][abbr] = df["total_mimic"].sum()
    return ext_dict, mimic_dict

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="path to root dir of models to analyze")

    args = parser.parse_args()
    #if "bayesopt" in args.dir:
       # file_list = get_best_temp_files(args.dir)
    ext_dict, mimic_dict = get_file_list(args.dir)
    print("EXTERNAL DATASET")
    print(ext_dict["abbr_acc"])
    print("len_dict:::{}".format(len(ext_dict["abbr_acc"])))
    print("MIMIC DATASET")
    print(mimic_dict["abbr_acc"])
    print("len_dict:::{}".format(len(mimic_dict["abbr_acc"])))

    macro_acc_ext, micro_acc_ext = compute_global_stats(ext_dict)
    macro_acc_mimic, micro_acc_mimic = compute_global_stats(mimic_dict)

    print("MACRO ACC EXT:::{}, MICRO ACC EXT:::{}".format(macro_acc_ext, micro_acc_ext))
    print("MACRO ACC MIMIC:::{}, MICRO ACC MIMIC:::{}".format(macro_acc_mimic, micro_acc_mimic))

if __name__ == "__main__":
    main()
