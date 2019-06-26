import sys, os, uuid
from argparse import ArgumentParser

from utils.CondorJobSubmitter import CondorJobSubmitter
from base.Configs import TrainingConfig

def RunCBAOptimizationCampaign(outdir, nrep):
    training_data_path = TrainingConfig.data_path

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # launch 'nrep' optimization jobs
    for cur in range(int(nrep)):
        cur_outdir = os.path.join(outdir, "slice_{}".format(cur))
        os.makedirs(cur_outdir)

        # create job script
        script_name = str(uuid.uuid4()) + ".sh"
        script_path = os.path.join(cur_outdir, script_name)

        with open(script_path, "w") as outfile:
            outfile.write("#!/bin/bash\n")
            outfile.write("source /home/windischhofer/HiggsPivoting/bin/activate\n")
            outfile.write("source /home/windischhofer/HiggsPivoting/setup_env.sh\n")
            
            outfile.write("python /home/windischhofer/HiggsPivoting/OptimizeCBASensitivity.py --data " + training_data_path + " --outdir " + cur_outdir + " > " + os.path.join(cur_outdir, "job.log\n"))
            
            outfile.write("deactivate\n")

        # submit it
        CondorJobSubmitter.submit_job(script_path)

if __name__ == "__main__":
    parser = ArgumentParser(description = "launch CBA optimization campaign")
    parser.add_argument("--outdir", action = "store", dest = "outdir")
    parser.add_argument("--nrep", action = "store", dest = "nrep")
    args = vars(parser.parse_args())

    RunCBAOptimizationCampaign(**args)

