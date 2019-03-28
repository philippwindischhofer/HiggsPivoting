import sys, os, glob, uuid
from argparse import ArgumentParser

from utils.CondorJobSubmitter import CondorJobSubmitter

def create_job_script(model_dir, script_dir, out_dir, training_data_path, num_categories):
    script_name = str(uuid.uuid4()) + ".sh"
    script_path = os.path.join(script_dir, script_name)

    # make sure that the script directory actually exists
    if not os.path.exists(script_dir):
        os.makedirs(script_dir)

    with open(script_path, "w") as outfile:
        outfile.write("#!/bin/bash\n")
        outfile.write("source /home/windischhofer/HiggsPivoting/bin/activate\n")
        outfile.write("source /home/windischhofer/HiggsPivoting/setup_env.sh\n")

        # first, generate the MC templates
        outfile.write("python /home/windischhofer/HiggsPivoting/ExportAnalysisRegionSweepHistograms.py --data " + training_data_path + " --model_dir " + model_dir + " --out_dir " + out_dir + " --test_size 0.5 " + " --num_categories " + str(num_categories) + " > " + os.path.join(model_dir, "fitjob.log") + "\n")
        outfile.write("deactivate\n")
        
    return script_path

def RunPrepareHistFitterCampaign(model_dirs, out_dir, num_categories = 1):
    training_data_path = "/home/windischhofer/datasmall/Hbb/training-mc16d.h5"
    
    for model_dir in model_dirs:
        # need to use a separate folder for each model, following the model naming convention
        cur_sub_outdir = os.path.basename(os.path.normpath(model_dir))
        cur_outdir = os.path.join(out_dir, cur_sub_outdir)
        job_script = create_job_script(model_dir, script_dir = cur_outdir, out_dir = cur_outdir, training_data_path = training_data_path, num_categories = num_categories)
        CondorJobSubmitter.submit_job(job_script)

if __name__ == "__main__":
    parser = ArgumentParser(description = "create jobs filling N signal regions")
    parser.add_argument("--max_categories", action = "store", dest = "max_categories")
    parser.add_argument("--outdir", action = "store", dest = "outdir")
    parser.add_argument("model_dirs", nargs = '+', action = "store")
    args = vars(parser.parse_args())

    model_dirs = args["model_dirs"]
    max_categories = int(args["max_categories"])
    root_out_dir = args["outdir"]

    # first, create the global output directory
    if not os.path.exists(root_out_dir):
        os.makedirs(root_out_dir)

    for cur_num_categories in range(1, max_categories + 1):

        # create a separate output directory for each category setup
        category_outdir = os.path.join(root_out_dir, "Category_slice_{}".format(cur_num_categories))
        if not os.path.exists(category_outdir):
            os.makedirs(category_outdir)

        RunPrepareHistFitterCampaign(model_dirs = model_dirs, out_dir = category_outdir, num_categories = cur_num_categories)
