import os, glob, uuid
from argparse import ArgumentParser

from utils.CondorJobSubmitter import CondorJobSubmitter
from delphes.CrossSectionReader import CrossSectionReader

def createJobScript(outfile, xsec, lumi, sample_name, input_files, script_dir):
    sceleton = """#!/bin/bash
    source /home/windischhofer/HiggsPivoting/bin/activate
    source /home/windischhofer/HiggsPivoting/setup_env.sh

    python3 /home/windischhofer/HiggsPivoting/DelphesDatasetExtractor.py --xsec {xsec_file_path} --outfile {outfile} --lumi {lumi} --sname {sample_name} {input_files}
    """
    
    opts = {"xsec_file_path": xsec, "outfile": outfile, "lumi": lumi, "sample_name": sample_name, "input_files": " ".join(input_files)}

    script_path = os.path.join(script_dir, str(uuid.uuid4()) + ".sh")
    with open(script_path, "w") as outfile:
        outfile.write(sceleton.format(**opts))

    return script_path

def launchDelphesDatasetExtractorJobs(input_dir, output_dir, lumi, sample_name, nfilesperjob):
    # first, look for the available root files
    print("looking for ROOT files in '{}'".format(input_dir))
    event_file_candidates = glob.glob(os.path.join(input_dir, "**/*.root"), recursive = True)
    print("found {} ROOT files for this process".format(len(event_file_candidates)))

    print("looking for xsec file in '{}'".format(input_dir))
    # look for the file containing the cross section
    xsec_file_candidates = glob.glob(os.path.join(input_dir, "cross_section.txt"))
    if len(xsec_file_candidates) != 1:
        raise Exception("Error: found more than one plausible cross-section file. Please check your files!")
    else:
        xsec_file_path = xsec_file_candidates[0]

    def generate_chunks(inlist, chunksize):
        return [inlist[cur:cur + chunksize] for cur in range(0, len(inlist), chunksize)]

    chunked_inputs = generate_chunks(event_file_candidates, chunksize = nfilesperjob)

    # generate the jobs acting on these chunked file inputs
    for job_input in chunked_inputs:
        cur_outname = str(uuid.uuid4())
        script_path = createJobScript(outfile = os.path.join(output_dir, cur_outname + ".h5"), xsec = xsec_file_path, lumi = lumi, sample_name = sample_name, input_files = job_input, script_dir = output_dir)
        CondorJobSubmitter.submit_job(script_path)

if __name__ == "__main__":
    parser = ArgumentParser("launch batch jobs to apply the event selection")
    parser.add_argument("--outdir", action = "store", dest = "outdir")
    parser.add_argument("--lumi", action = "store", dest = "lumi")
    parser.add_argument("--sname", action = "store", dest = "sample_name")
    parser.add_argument("indir", nargs = "+", action = "store")
    parser.add_argument("--nfilesperjob", action = "store", dest = "nfilesperjob")
    args = vars(parser.parse_args())
    
    outdir = args["outdir"]
    lumi = float(args["lumi"])
    sample_name = args["sample_name"]
    nfilesperjob = int(args["nfilesperjob"])
    indir = args["indir"][0]

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    launchDelphesDatasetExtractorJobs(indir, outdir, lumi, sample_name, nfilesperjob)
