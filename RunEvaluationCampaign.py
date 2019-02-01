import sys, os, glob, uuid

from utils.CondorJobSubmitter import CondorJobSubmitter

def create_job_script(model_dir, script_dir, training_data_path):
    script_name = str(uuid.uuid4()) + ".sh"
    script_path = os.path.join(script_dir, script_name)
    plot_dir = model_dir

    with open(script_path, "w") as outfile:
        outfile.write("#!/bin/bash\n")
        outfile.write("source /home/windischhofer/HiggsPivoting/bin/activate\n")
        outfile.write("source /home/windischhofer/HiggsPivoting/setup_env.sh\n")

        outfile.write("python /home/windischhofer/HiggsPivoting/EvaluateModels.py --data " + training_data_path + " --plot_dir " + plot_dir + " " + model_dir + " > " + os.path.join(model_dir, "evaljob.log") + "\n")
        outfile.write("deactivate\n")

    return script_path

def RunEvaluationCampaign(model_dirs):
    training_data_path = "/home/windischhofer/datasmall/Hbb/training-mc16d.h5"

    for model_dir in model_dirs:
        job_script = create_job_script(model_dir, script_dir = model_dir, training_data_path = training_data_path)
        CondorJobSubmitter.submit_job(job_script)

if __name__ == "__main__":
    model_dirs = sys.argv[1:]
    RunEvaluationCampaign(model_dirs)

