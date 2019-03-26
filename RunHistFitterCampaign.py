import sys, os, glob, uuid

from utils.CondorJobSubmitter import CondorJobSubmitter

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

        # and launch the Asimov fit
        outfile.write("cd " + model_dir + "\n")
        outfile.write('/home/windischhofer/HistFitter/v0.61.0/scripts/HistFitter.py -w -f -d -D allPlots -F excl -m ALL -a --userArg ' + model_dir + ' /home/windischhofer/HiggsPivoting/fitting/ShapeFit.py' + " >> " + os.path.join(model_dir, "fitjob.log") + "\n")

        # then, convert the HistFitter output into a pickled file
        outfile.write("python /home/windischhofer/HiggsPivoting/fitting/ConvertHistFitterResult.py --infile " + os.path.join(model_dir, "results", "TemplateAnalysisSimple", 
                                                                                                                             "shape_fit_combined_shape_fit_model_afterFit_asimovData.root") + 
                      " --outfile " + os.path.join(model_dir, "pardict.pkl") + "\n")

    return script_path

def RunHistFitterCampaign(model_dirs):
    training_data_path = "/home/windischhofer/datasmall/Hbb/training-mc16d.h5"
    
    for model_dir in model_dirs:
        job_script = create_job_script(model_dir, script_dir = model_dir, training_data_path = training_data_path)
        CondorJobSubmitter.submit_job(job_script)
        
if __name__ == "__main__":
    model_dirs = sys.argv[1:]
    RunHistFitterCampaign(model_dirs)
