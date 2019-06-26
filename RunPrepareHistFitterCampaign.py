import sys, os, glob, uuid
from argparse import ArgumentParser

from utils.CondorJobSubmitter import CondorJobSubmitter
from utils.LocalJobSubmitter import LocalJobSubmitter
from base.Configs import TrainingConfig

def create_job_script(model_dir, script_dir, training_data_path, rootdir, use_test = False):
    script_name = str(uuid.uuid4()) + ".sh"
    script_path = os.path.join(script_dir, script_name)
    plot_dir = model_dir

    with open(script_path, "w") as outfile:
        outfile.write("#!/bin/bash\n")
        outfile.write("source " + os.path.join(rootdir, "bin", "activate") + "\n")
        outfile.write("source " + os.path.join(rootdir, "setup_env.sh") + "\n")

        # first, generate the MC templates
        outfile.write("python " + os.path.join(rootdir, "ExportAnalysisRegionHistograms.py") + " --data " + training_data_path + " --model_dir " + model_dir + " --out_dir " + model_dir + (" --use_test " if use_test else "") + " > " + os.path.join(model_dir, "fitjob.log") + "\n")
        outfile.write("deactivate\n")
        
    return script_path

def RunPrepareHistFitterCampaign(model_dirs, **kwargs):
    training_data_path = TrainingConfig.data_path
    
    for model_dir in model_dirs:
        job_script = create_job_script(model_dir, script_dir = model_dir, training_data_path = training_data_path, rootdir = os.environ["ROOTDIR"], **kwargs)
        TrainingConfig.submitter.submit_job(job_script)

if __name__ == "__main__":
    if not os.environ["ROOTDIR"]:
        raise Exception("Error: 'ROOTDIR' not defined. Please do 'source setup_env.sh'.")

    parser = ArgumentParser()
    parser.add_argument("--use_test", action = "store_const", const = True, default = False, dest = "use_test")
    parser.add_argument("model_dirs", nargs = '+', action = "store")
    args = vars(parser.parse_args())

    RunPrepareHistFitterCampaign(**args)
