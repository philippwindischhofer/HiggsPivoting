import os, glob, uuid
from argparse import ArgumentParser

from utils.CondorJobSubmitter import CondorJobSubmitter

def create_job_script(job_script_dir, indirs, lumi, xsec):
    sceleton = """#!/bin/bash
    source /home/windischhofer/HiggsPivoting/bin/activate
    source /home/windischhofer/HiggsPivoting/setup_env.sh
    
    python3 /home/windischhofer/HiggsPivoting/MakeLumiFile.py --lumi {lumi} --xsec {xsec} {indirs} &> {logfile}
    """
    
    job_id = str(uuid.uuid4())

    job_script_path = os.path.join(job_script_dir, job_id + ".sh")
    logfile_path = os.path.join(job_script_dir, job_id + ".log")

    opts = {"lumi": lumi, "xsec": xsec, "indirs": " ".join(indirs), "logfile": logfile_path}

    with open(job_script_path, 'w') as outfile:
        outfile.write(sceleton.format(**opts))

    return job_script_path

if __name__ == "__main__":
    parser = ArgumentParser(description = "generate lumifile on the batch system")
    parser.add_argument("dirs", nargs = '+', action = "store")
    parser.add_argument("--lumi", action = "store", dest = "lumi")
    parser.add_argument("--xsec", action = "store", dest = "xsec")
    parser.add_argument("--dirsperjob", action = "store", dest = "dirsperjob")
    args = vars(parser.parse_args())

    dirs = args["dirs"]
    lumi = float(args["lumi"])
    xsec = float(args["xsec"])
    dirsperjob = int(args["dirsperjob"])

    # get a list of all subdirectories that contain MC files, launch one job for each
    for cur_dir in dirs:
        cur_sub_dirs = glob.glob(os.path.join(cur_dir, '*/'))
        job_script_dir = cur_dir

        print("found a total of {} directories within '{}'".format(len(cur_sub_dirs), cur_dir))

        def generate_chunks(inlist, chunksize):
            return [inlist[cur:cur + chunksize] for cur in range(0, len(inlist), chunksize)]

        chunked_dirs = generate_chunks(cur_sub_dirs, chunksize = dirsperjob)

        # generate a lumi file for each directory sequentially
        for cur_dirs in chunked_dirs:
            cur_job_script = create_job_script(job_script_dir = job_script_dir, indirs = cur_dirs, 
                                               lumi = lumi, xsec = xsec)
            CondorJobSubmitter.submit_job(cur_job_script)
