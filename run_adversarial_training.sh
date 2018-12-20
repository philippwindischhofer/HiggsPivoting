#!/bin/bash

# set up the environment
source /home/windischhofer/HiggsPivoting/bin/activate
source /home/windischhofer/HiggsPivoting/setup_env.sh

python /home/windischhofer/HiggsPivoting/TrainAdversarialModel.py --training_data /home/windischhofer/data/Hbb/training-mc16d.h5 --outdir /home/windischhofer/data/test/Hbb_models/ > /home/windischhofer/data/test/job.log

deactivate
