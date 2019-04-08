import os, glob, uuid
from argparse import ArgumentParser

from utils.CondorJobSubmitter import CondorJobSubmitter

def create_job_script(indir, lumi):
    sceleton = """#!/bin/bash
    source /home/windischhofer/HiggsPivoting/bin/activate
    source /home/windischhofer/HiggsPivoting/setup_env.sh
    
    python3 /home/windischhofer/HiggsPivoting/MakeLumiFile.py --lumi {lumi} {indir} &> {logfile}
    """
    
    job_id = str(uuid.uuid4())

    job_script_path = os.path.join(indir, job_id + ".sh")
    logfile_path = os.path.join(indir, job_id + ".log")

    opts = {"lumi": lumi, "indir": indir, "logfile": logfile_path}

    with open(job_script_path, 'w') as outfile:
        outfile.write(sceleton.format(**opts))

    return job_script_path

if __name__ == "__main__":
    parser = ArgumentParser(description = "generate lumifile on the batch system")
    parser.add_argument("dirs", nargs = '+', action = "store")
    parser.add_argument("--lumi", action = "store", dest = "lumi")
    args = vars(parser.parse_args())

    dirs = args["dirs"]
    lumi = float(args["lumi"])

    # generate a lumi file for each directory sequentially
    for cur_dir in dirs:
        cur_job_script = create_job_script(indir = cur_dir, lumi = lumi)
        CondorJobSubmitter.submit_job(cur_job_script)
