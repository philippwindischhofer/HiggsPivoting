import sys, os, glob, uuid
from argparse import ArgumentParser

from utils.CondorJobSubmitter import CondorJobSubmitter
from utils.LocalJobSubmitter import LocalJobSubmitter
from base.Configs import TrainingConfig

def create_job_script(model_dir, script_dir, training_data_path, use_test = False):
    script_name = str(uuid.uuid4()) + ".sh"
    script_path = os.path.join(script_dir, script_name)
    plot_dir = model_dir

    with open(script_path, "w") as outfile:
        outfile.write("#!/bin/bash\n")
        outfile.write("source /home/windischhofer/HiggsPivoting/bin/activate\n")
        outfile.write("source /home/windischhofer/HiggsPivoting/setup_env.sh\n")

        outfile.write("python /home/windischhofer/HiggsPivoting/EvaluateModels.py --data " + training_data_path + " --plot_dir " + plot_dir + " " + model_dir + (" --use_test " if use_test else "") + " > " + os.path.join(model_dir, "evaljob.log") + "\n")
        outfile.write("deactivate\n")

    return script_path

def RunEvaluationCampaign(model_dirs, use_test):
    training_data_path = TrainingConfig.data_path

    for model_dir in model_dirs:
        job_script = create_job_script(model_dir, script_dir = model_dir, training_data_path = training_data_path, use_test = use_test)
        CondorJobSubmitter.submit_job(job_script)
        #LocalJobSubmitter.submit_job(job_script)
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--use_test", action = "store_const", const = True, default = False, dest = "use_test")
    parser.add_argument("model_dirs", nargs = '+', action = "store")
    args = vars(parser.parse_args())

    RunEvaluationCampaign(**args)
