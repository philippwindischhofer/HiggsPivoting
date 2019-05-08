import sys, os, glob, uuid

from utils.CondorJobSubmitter import CondorJobSubmitter
from utils.LocalJobSubmitter import LocalJobSubmitter
from base.Configs import TrainingConfig

def create_job_script(model_dir, script_dir, training_data_path):
    script_name = str(uuid.uuid4()) + ".sh"
    script_path = os.path.join(script_dir, script_name)
    plot_dir = model_dir

    with open(script_path, "w") as outfile:
        outfile.write("#!/bin/bash\n")
        
        # then, prepare the HistFitter environment
        outfile.write("export PYTHONPATH=/home/windischhofer/HiggsPivoting/fitting:$PYTHONPATH" + "\n")
        outfile.write("export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase" + "\n")
        outfile.write("export ALRB_rootVersion=6.14.04-x86_64-slc6-gcc62-opt" + "\n")
        outfile.write("source /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh" + "\n")
        outfile.write("lsetup root" + "\n")
        outfile.write("source /home/windischhofer/HistFitter/v0.61.0/setup.sh" + "\n")

        # and launch the Asimov fits
        outfile.write("cd " + model_dir + "\n")
        outfile.write('/home/windischhofer/HistFitter/v0.61.0/scripts/HistFitter.py -w -f -d -D allPlots -F disc -m ALL -a -z --userArg ' + model_dir + ' /home/windischhofer/HiggsPivoting/fitting/ShapeFitNCategoriesBackgroundFixed.py' + " >> " + os.path.join(model_dir, "fitjob_tight_loose_background_fixed.log") + "\n")
        outfile.write('/home/windischhofer/HistFitter/v0.61.0/scripts/HistFitter.py -w -f -d -D allPlots -F disc -m ALL -a -z --userArg ' + model_dir + ' /home/windischhofer/HiggsPivoting/fitting/ShapeFitNCategoriesBackgroundFloating.py' + " >> " + os.path.join(model_dir, "fitjob_tight_loose_background_floating.log") + "\n")

        # also launch Asimov fits in the case of the cut-based analysis
        outfile.write('/home/windischhofer/HistFitter/v0.61.0/scripts/HistFitter.py -w -f -d -D allPlots -F disc -m ALL -a -z --userArg ' + model_dir + ' /home/windischhofer/HiggsPivoting/fitting/ShapeFitHighLowMETBackgroundFixed.py' + " >> " + os.path.join(model_dir, "fitjob_high_low_MET_background_fixed.log") + "\n")
        outfile.write('/home/windischhofer/HistFitter/v0.61.0/scripts/HistFitter.py -w -f -d -D allPlots -F disc -m ALL -a -z --userArg ' + model_dir + ' /home/windischhofer/HiggsPivoting/fitting/ShapeFitHighLowMETBackgroundFloating.py' + " >> " + os.path.join(model_dir, "fitjob_high_low_MET_background_floating.log") + "\n")

        # the Asimov significances that have been determined above
        outfile.write("python /home/windischhofer/HiggsPivoting/fitting/ConvertHistFitterResult.py --mode hypotest --infile " + os.path.join(model_dir, "results", "ShapeFitNCategoriesBackgroundFixed_hypotest.root") + " --outkey asimov_sig_ncat_background_fixed" + " --outfile " + os.path.join(model_dir, "hypodict.pkl") + "\n")
        outfile.write("python /home/windischhofer/HiggsPivoting/fitting/ConvertHistFitterResult.py --mode hypotest --infile " + os.path.join(model_dir, "results", "ShapeFitNCategoriesBackgroundFloating_hypotest.root") + " --outkey asimov_sig_ncat_background_floating" + " --outfile " + os.path.join(model_dir, "hypodict.pkl") + "\n")

        # the equivalent results for the cut-based setup
        outfile.write("python /home/windischhofer/HiggsPivoting/fitting/ConvertHistFitterResult.py --mode hypotest --infile " + os.path.join(model_dir, "results", "ShapeFitHighLowMETBackgroundFixed_hypotest.root") + " --outkey asimov_sig_high_low_MET_background_fixed" + " --outfile " + os.path.join(model_dir, "hypodict.pkl") + "\n")
        outfile.write("python /home/windischhofer/HiggsPivoting/fitting/ConvertHistFitterResult.py --mode hypotest --infile " + os.path.join(model_dir, "results", "ShapeFitHighLowMETBackgroundFloating_hypotest.root") + " --outkey asimov_sig_high_low_MET_background_floating" + " --outfile " + os.path.join(model_dir, "hypodict.pkl") + "\n")

    return script_path

def RunHistFitterCampaign(model_dirs):
    training_data_path = TrainingConfig.data_path
    
    for model_dir in model_dirs:
        job_script = create_job_script(model_dir, script_dir = model_dir, training_data_path = training_data_path)
        CondorJobSubmitter.submit_job(job_script)
        #LocalJobSubmitter.submit_job(job_script)
        
if __name__ == "__main__":
    model_dirs = sys.argv[1:]
    RunHistFitterCampaign(model_dirs)
