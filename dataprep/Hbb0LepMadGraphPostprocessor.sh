#!/bin/bash
source /home/windischhofer/HiggsPivoting/bin/activate
source /home/windischhofer/HiggsPivoting/setup_env_slc6.sh

INDIR=$1
OUTDIR=$2

# generate the SOW file (this will generate $INDIR/lumi.conf)
python3 /home/windischhofer/HiggsPivoting/MakeLumiFile.py --lumi -1 --xsec -1 $INDIR

# apply the event selection
python3 /home/windischhofer/HiggsPivoting/DelphesDatasetExtractor.py --outfile ${INDIR}/events.h5 --sname generic_process ${INDIR}/*.root

# move the results into the output directory
mv ${INDIR}/events.h5 ${INDIR}/lumi.conf $OUTDIR
