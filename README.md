# HiggsPivoting

Code to generate the results presented in "Preserving physically important variables in optimal event selections: A case study in Higgs physics" (ADD ARXIV LINK)

With this repository, you can do the following:
* train a pivotal classifier
* apply it to define signal regions
* evaluate the performance of the analysis

## Setup
Make sure you have a local installation of ROOT (including pyROOT) available before continuing!
As we require `python3` and the `pyROOT` bindings, building ROOT from source is probably the best option.
```
git clone http://github.com/root-project/root.git
cd root
git checkout -b v6-18-00 v6-18-00
mkdir ../rootbuild
cd ../rootbuild
cmake ../root -DPYTHON_EXECUTABLE=$PATH_TO_PYTHON_3
cmake --build . -- -j8
```

Then, clone this repository, create a virtual python3 environment and install dependencies
```
git clone -b master https://github.com/philippwindischhofer/HiggsPivoting.git
cd higgspivoting
python3 -m venv .
source bin/activate
pip install -r requirements.txt
source setup_env.sh
```

Make sure to add `rootbuild/lib` to your `PYTHONPATH` to allow `pyroot` to be imported! You can use e.g. `setup_env.sh` to do that.

## Training

To run a training campaign (assuming you want to locate the output at $TRAIN_DIR):
```
mkdir $TRAIN_DIR
cp examples/Master.conf $TRAIN_DIR
python RunTrainingCampaign.py --confpath $TRAIN_DIR/Master.conf --nrep 1
```
Here, `nrep` is the number of trainings that should be carried out for each parameter point.

This repository contains a small training dataset (100000 events per signal and background process) in `examples/training-MadGraphPy8-ATLAS-small.h5` (used by default). This should be enough to play
with the method, but not enough to reach good performance. A larger dataset is available from the authors upon request.

## Evaluation

To evaluate the classifier and generate the output for an Asimov fit:
```
source bin/activate
source setup_env.sh
python RunPrepareHistFitterCampaign.py $TRAIN_DIR/Master_slice_*
```

To perform an Asimov fit:

Note: for this step, you need to have a local installation of HistFitter available!
Installation instructions are available at http://histfitter.web.cern.ch/histfitter/Software/Install/index.html
Once the installation has completed, you need to correctly set the path to your local HistFitter installation directory
in `setup_env.sh`.

```
source bin/activate
source setup_env.sh
python RunHistFitterCampaign.py $TRAIN_DIR/Master_slice_*
```

## Visualisation / Plotting

To produce the summary plots:
```
source bin/activate
source setup_env.sh
python MakeGlobalAnalysisPlots.py --plotdir $PLOT_DIR $TRAIN_DIR/Master_slice_* 
python MakeGlobalAsimovPlots.py --plotdir $PLOT_DIR $TRAIN_DIR/Master_slice_* 
```
Here, `$PLOT_DIR` is the directory where the plots should be stored.