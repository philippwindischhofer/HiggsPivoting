import sys, os, glob, uuid
from argparse import ArgumentParser

from utils.CondorJobSubmitter import CondorJobSubmitter

def create_job_script(model_dir, script_dir, training_data_path, num_categories):
    script_name = str(uuid.uuid4()) + ".sh"
    script_path = os.path.join(script_dir, script_name)
    plot_dir = model_dir

    with open(script_path, "w") as outfile:
        outfile.write("#!/bin/bash\n")
        outfile.write("source /home/windischhofer/HiggsPivoting/bin/activate\n")
        outfile.write("source /home/windischhofer/HiggsPivoting/setup_env.sh\n")

        # first, generate the MC templates
        outfile.write("python /home/windischhofer/HiggsPivoting/ExportAnalysisRegionSweepHistograms.py --data " + training_data_path + " --model_dir " + model_dir + " --out_dir " + model_dir + " --test_size 0.5 " + " --num_categories " + num_categories + " > " + os.path.join(model_dir, "fitjob.log") + "\n")
        outfile.write("deactivate\n")
        
    return script_path

def RunPrepareHistFitterCampaign(model_dirs, num_categories = 1):
    training_data_path = "/home/windischhofer/datasmall/Hbb/training-mc16d.h5"
    
    for model_dir in model_dirs:
        job_script = create_job_script(model_dir, script_dir = model_dir, training_data_path = training_data_path, num_categories = num_categories)
        CondorJobSubmitter.submit_job(job_script)

if __name__ == "__main__":
    parser = ArgumentParser(description = "create jobs filling N signal regions")
    parser.add_argument("--num_categories", action = "store", dest = "num_categories")
    parser.add_argument("model_dirs", nargs = '+', action = "store")
    args = vars(parser.parse_args())

    model_dirs = args["model_dirs"]
    num_categories = args["num_categories"]
    RunPrepareHistFitterCampaign(model_dirs, num_categories)
