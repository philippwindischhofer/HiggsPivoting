import os, glob, uuid
from argparse import ArgumentParser

from utils.CondorJobSubmitter import CondorJobSubmitter
from delphes.CrossSectionReader import CrossSectionReader

def createJobScript(outfile, sample_name, input_files, script_dir):
    sceleton = """#!/bin/bash
    source /home/windischhofer/HiggsPivoting/bin/activate
    source /home/windischhofer/HiggsPivoting/setup_env.sh

    python3 /home/windischhofer/HiggsPivoting/DelphesDatasetExtractor.py --outfile {outfile} --sname {sample_name} {input_files}
    """
    
    opts = {"outfile": outfile, "sample_name": sample_name, "input_files": " ".join(input_files)}

    script_path = os.path.join(script_dir, str(uuid.uuid4()) + ".sh")
    with open(script_path, "w") as outfile:
        outfile.write(sceleton.format(**opts))

    return script_path

def launchDelphesDatasetExtractorJobs(input_dir, output_dir, sample_name, nfilesperjob):
    # first, look for the available root files
    print("looking for ROOT files in '{}'".format(input_dir))

    # Note: this semi-automatic way of doing it is faster than simply
    #     event_file_candidates = glob.glob(os.path.join(input_dir, "**/*.root"), recursive = True)
    sub_dirs = glob.glob(os.path.join(indir, '*/'))
    print("looking through {} subdirectories".format(len(sub_dirs)))
    
    event_file_candidates = []
    for sub_dir in sub_dirs:
        eventfile_paths = glob.glob(os.path.join(sub_dir, "*.root"))
        for eventfile_path in eventfile_paths:
            if os.path.isfile(eventfile_path):
                event_file_candidates.append(eventfile_path)

    print("found {} ROOT files for this process".format(len(event_file_candidates)))

    def generate_chunks(inlist, chunksize):
        return [inlist[cur:cur + chunksize] for cur in range(0, len(inlist), chunksize)]

    chunked_inputs = generate_chunks(event_file_candidates, chunksize = nfilesperjob)

    # generate the jobs acting on these chunked file inputs
    for job_input in chunked_inputs:
        cur_outname = str(uuid.uuid4())
        script_path = createJobScript(outfile = os.path.join(output_dir, cur_outname + ".h5"), sample_name = sample_name, input_files = job_input, script_dir = output_dir)
        CondorJobSubmitter.submit_job(script_path)

if __name__ == "__main__":
    parser = ArgumentParser("launch batch jobs to apply the event selection")
    parser.add_argument("--outdir", action = "store", dest = "outdir")
    parser.add_argument("indir", nargs = "+", action = "store")
    parser.add_argument("--nfilesperjob", action = "store", dest = "nfilesperjob")
    args = vars(parser.parse_args())
    
    outdir = args["outdir"]
    nfilesperjob = int(args["nfilesperjob"])
    indir = args["indir"][0]

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    launchDelphesDatasetExtractorJobs(input_dir = indir, output_dir = outdir, sample_name = "generic_process", nfilesperjob = nfilesperjob)
