from argparse import ArgumentParser
from distutils.dir_util import copy_tree
import os, glob

from configparser import ConfigParser

def merge_config_files(infile_2j, infile_3j, outfile):
    config_2j = ConfigParser()
    config_2j.read(infile_2j)

    config_3j = ConfigParser()
    config_3j.read(infile_3j)

    outconfig = ConfigParser()
    outconfig["ModelCollection"] = {"models": "model_2j,model_3j"}
    outconfig["model_2j"] = config_2j["model_2j"]
    outconfig["classifier_2j"] = config_2j["classifier_2j"]
    outconfig["adversary_2j"] = config_2j["adversary_2j"]
    outconfig["training_config_2j"] = config_2j["training_config_2j"]

    outconfig["model_3j"] = config_3j["model_3j"]
    outconfig["classifier_3j"] = config_3j["classifier_3j"]
    outconfig["adversary_3j"] = config_3j["adversary_3j"]
    outconfig["training_config_3j"] = config_3j["training_config_3j"]

    with open(outfile, "w") as config_outfile:
        outconfig.write(config_outfile)

def Merge2j3jRuns(indir_2j, indir_3j, outdir):

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    available_runs_2j = glob.glob(os.path.join(indir_2j, "Master_slice*"))
    
    for cur_run_2j in available_runs_2j:
        
        # find the corresponding 3j run
        run_basename = os.path.basename(os.path.normpath(cur_run_2j))
        cur_run_3j = os.path.join(indir_3j, run_basename)

        if not os.path.exists(cur_run_3j):
            print("3j run does not exist, skipping")
            continue

        # prepare the output directory
        cur_outrun = os.path.join(outdir, run_basename)
        if not os.path.exists(cur_outrun):
            os.makedirs(cur_outrun)

        # copy the training files
        copy_tree(os.path.join(cur_run_3j, "model_3j"), os.path.join(cur_outrun, "model_3j"))
        copy_tree(os.path.join(cur_run_2j, "model_2j"), os.path.join(cur_outrun, "model_2j"))

        # merge the config files and put it in the merged output directory as well
        merge_config_files(infile_3j = os.path.join(cur_run_3j, "meta.conf"), infile_2j = os.path.join(cur_run_2j, "meta.conf"),
                           outfile = os.path.join(cur_outrun, "meta.conf"))

if __name__ == "__main__":
    if not os.environ["ROOTDIR"]:
        raise Exception("Error: 'ROOTDIR' not defined. Please do 'source setup_env.sh'.")

    parser = ArgumentParser()
    parser.add_argument("--outdir", action = "store", dest = "outdir")
    parser.add_argument("--indir_2j", action = "store")
    parser.add_argument("--indir_3j", action = "store")
    args = vars(parser.parse_args())

    Merge2j3jRuns(**args)

