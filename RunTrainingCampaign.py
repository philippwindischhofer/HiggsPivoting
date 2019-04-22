import sys, os, glob, uuid
sys.path.append("/home/windischhofer/ConfigFileSweeper/ConfigFileSweeper/")
from argparse import ArgumentParser
from shutil import copyfile

from ConfigFileSweeper import ConfigFileSweeper
from utils.CondorJobSubmitter import CondorJobSubmitter
from base.Configs import TrainingConfig

def create_job_script(training_data_path, run_dir, script_dir):
    script_name = str(uuid.uuid4()) + ".sh"
    script_path = os.path.join(script_dir, script_name)

    with open(script_path, "w") as outfile:
        outfile.write("#!/bin/bash\n")
        outfile.write("source /home/windischhofer/HiggsPivoting/bin/activate\n")
        outfile.write("source /home/windischhofer/HiggsPivoting/setup_env.sh\n")

        outfile.write("python /home/windischhofer/HiggsPivoting/TrainAdversarialModel.py --data " + training_data_path + " --outdir " + run_dir + " > " + os.path.join(run_dir, "job.log\n"))

        outfile.write("deactivate\n")

    return script_path
    
def RunTrainingCampaign(master_confpath, nrep = 1):
    # some global settings
    training_data_path = TrainingConfig.data_path
    
    # first, generate the actual configuration files, starting from the master file
    campaign_dir = os.path.dirname(master_confpath)
    ConfigFileSweeper(infile_path = master_confpath, output_dir = campaign_dir)

    # then, create separate subdirectories for each run, and put the respective config files into those directories
    config_files = glob.glob(os.path.join(campaign_dir, "*_slice*.conf"))
    for config_file in config_files:
        config_file_basename, _ = os.path.splitext(config_file)
        config_file_fundname = config_file_basename.replace('.conf', '')

        for rep in range(nrep):
            run_dir = os.path.join(campaign_dir, config_file_fundname + "." + str(rep))

            print("creating run directory '" + run_dir + "'")
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)
            copyfile(config_file, os.path.join(run_dir, "meta.conf"))
            #os.rename(config_file, os.path.join(run_dir, "meta.conf"))
        
            # create the job scripts
            job_script = create_job_script(training_data_path, run_dir, run_dir)

            # submit them to the condor batch system
            CondorJobSubmitter.submit_job(job_script)

        os.remove(config_file)

if __name__ == "__main__":
    parser = ArgumentParser(description = "launch training campaign")
    parser.add_argument("--confpath", action = "store", dest = "master_confpath")
    parser.add_argument("--nrep", action = "store", dest = "nrep")
    args = vars(parser.parse_args())

    master_confpath = args["master_confpath"]
    nrep = int(args["nrep"])
    RunTrainingCampaign(master_confpath, nrep = nrep)
