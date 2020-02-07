from argparse import ArgumentParser
import os, time, glob

from utils.CondorJobSubmitter import CondorJobSubmitter
from utils.LocalJobSubmitter import LocalJobSubmitter
from base.Configs import TrainingConfig

def run_and_wait(func, **args):
    jobs_before = TrainingConfig.submitter.get_running_cluster_IDs()
    func(**args)
    jobs_after = TrainingConfig.submitter.get_running_cluster_IDs()

    need_to_finish = jobs_after.difference(jobs_before)

    while True:
        currently_running = TrainingConfig.submitter.get_running_cluster_IDs()
        wait_to_finish = len(currently_running.intersection(need_to_finish))
        print("waiting for {} jobs to finish".format(wait_to_finish))

        if wait_to_finish > 0:
            time.sleep(5)
        else:
            break

def CampaignPilot(master_confpath, nrep, use_test):

    # get the things that need to be done in the right order
    from RunTrainingCampaign import RunTrainingCampaign
    from RunPrepareHistFitterCampaign import RunPrepareHistFitterCampaign
    from RunHistFitterCampaign import RunHistFitterCampaign
    from MakeGlobalAsimovPlots import MakeGlobalAsimovPlots
    from MakeGlobalAnalysisPlots import MakeAllGlobalAnalysisPlots

    # run the training
    run_and_wait(RunTrainingCampaign, master_confpath = master_confpath, nrep = nrep)

    # these are the directories for the individual runs
    workdir = os.path.dirname(master_confpath)
    run_dir_pattern = os.path.splitext(master_confpath)[0] + "_slice*"
    run_dirs = glob.glob(run_dir_pattern)

    print("After training, found the following model directories:")
    print('\n'.join(run_dirs))

    # export the histograms
    run_and_wait(RunPrepareHistFitterCampaign, model_dirs = run_dirs)

    # run HistFitter
    run_and_wait(RunHistFitterCampaign, model_dirs = run_dirs)

    # and make the plots
    MakeGlobalAsimovPlots(model_dirs = run_dirs, plot_dir = workdir)
    MakeAllGlobalAnalysisPlots({"model_dirs": run_dirs, "plotdir": workdir})

if __name__ == "__main__":
    if not os.environ["ROOTDIR"]:
        raise Exception("Error: 'ROOTDIR' not defined. Please do 'source setup_env.sh'.")

    parser = ArgumentParser()
    parser.add_argument("--confpath", action = "store", dest = "master_confpath")
    parser.add_argument("--nrep", action = "store", dest = "nrep", type = int)
    parser.add_argument("--use_test", action = "store_const", const = True, default = False)
    args = vars(parser.parse_args())

    CampaignPilot(**args)
