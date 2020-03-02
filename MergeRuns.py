from argparse import ArgumentParser
from distutils.dir_util import copy_tree
import os, glob

def MergeRuns(indirs, outdir):

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    merged_cnt = 0
    for indir in indirs:
        available_rundirs = glob.glob(os.path.join(indir, "Master_slice*"))

        for cur_rundir in available_rundirs:
            cur_sourcedir = cur_rundir
            cur_targetdir = os.path.join(outdir, "Master_slice_{}.0".format(merged_cnt))
            copy_tree(cur_sourcedir, cur_targetdir)

            merged_cnt += 1

if __name__ == "__main__":
    if not os.environ["ROOTDIR"]:
        raise Exception("Error: 'ROOTDIR' not defined. Please do 'source setup_env.sh'.")

    parser = ArgumentParser()
    parser.add_argument("--outdir", action = "store", dest = "outdir")
    parser.add_argument("--indirs", nargs = '+', action = "store")
    args = vars(parser.parse_args())

    MergeRuns(**args)
