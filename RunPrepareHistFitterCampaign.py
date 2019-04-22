import sys, os, glob, uuid

from utils.CondorJobSubmitter import CondorJobSubmitter
from base.Configs import TrainingConfig

def create_job_script(model_dir, script_dir, training_data_path):
    script_name = str(uuid.uuid4()) + ".sh"
    script_path = os.path.join(script_dir, script_name)
    plot_dir = model_dir

    with open(script_path, "w") as outfile:
        outfile.write("#!/bin/bash\n")
        outfile.write("source /home/windischhofer/HiggsPivoting/bin/activate\n")
        outfile.write("source /home/windischhofer/HiggsPivoting/setup_env.sh\n")

        # first, generate the MC templates
        outfile.write("python /home/windischhofer/HiggsPivoting/ExportAnalysisRegionHistograms.py --data " + training_data_path + " --model_dir " + model_dir + " --out_dir " + model_dir + " --test_size {}".format(TrainingConfig.test_size) + " > " + os.path.join(model_dir, "fitjob.log") + "\n")
        outfile.write("deactivate\n")
        
    return script_path

def RunPrepareHistFitterCampaign(model_dirs):
    training_data_path = TrainingConfig.data_path
    
    for model_dir in model_dirs:
        job_script = create_job_script(model_dir, script_dir = model_dir, training_data_path = training_data_path)
        CondorJobSubmitter.submit_job(job_script)

if __name__ == "__main__":
    model_dirs = sys.argv[1:]
    RunPrepareHistFitterCampaign(model_dirs)
