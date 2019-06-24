#!/bin/bash
# keep track of the root path for the codebase
export ROOTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export HISTFITTER_ROOTDIR="/home/windischhofer/HistFitter/v0.61.0"
# for the moment, need to use the most recent nightly build
source /cvmfs/sft-nightlies.cern.ch/lcg/views/dev3python3/Mon/x86_64-centos7-gcc62-opt/setup.sh
#source /cvmfs/sft-nightlies.cern.ch/lcg/views/dev3python3/Fri/x86_64-centos7-gcc62-opt/setup.sh
#export PYTHONPATH="/home/windischhofer/HiggsPivoting/lib64/python3.6/site-packages/:$PYTHONPATH"
export PYTHONPATH="$PYTHONPATH:/home/windischhofer/HiggsPivoting/lib64/python3.6/site-packages"
