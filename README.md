# HiggsPivoting

Code to generate the results presented in "Windischhofer, Zgubic, Bortoletto: Preserving physically important variables in optimal event selections: A case study in Higgs physics" (ADD ARXIV LINK)

With this repository, you can do the following:
* train a pivotal classifier
* apply it to define signal regions
* evaluate the performance of the analysis

## Setup

Clone this repository, create a virtual python3 environment and install dependencies
```
git clone -b paper https://github.com/philippwindischhofer/HiggsPivoting.git $SRC_DIR
cd $SRC_DIR
python3 -m venv .
```
`$SRC_DIR` is some local directory where you want to keep the code. [`$SRC_DIR` is meant to reference the source directory you want to use. You can either define an environment variable to keep track of it (`export SRC_DIR="/path/to/source"`), or just keep the path explicit.]

Some of the dependencies require a local installation of ROOT (including pyROOT). As we require `python3` and the `pyROOT` bindings, building ROOT from source is probably the best option.

Download the ROOT source and store it in the local directory `$ROOT_SRC_DIR`:
```
git clone http://github.com/root-project/root.git $ROOT_SRC_DIR
cd $ROOT_SRC_DIR
git checkout -b v6-18-00 v6-18-00
```

We are going to install ROOT in the directory `$ROOT_INSTALL_DIR`. The installation will be completely self-contained and not mess with any other local installation of ROOT you may have. To undo the installation, it is sufficient to just do `rm -rf $ROOT_INSTALL_DIR $ROOT_SRC_DIR`.
```
mkdir $ROOT_INSTALL_DIR
cd $ROOT_INSTALL_DIR
cmake $ROOT_SRC_DIR -DPYTHON_EXECUTABLE=$PATH_TO_PYTHON_3
cmake --build . -- -j8
```
where `$PATH_TO_PYTHON_3` is the path to the Python3 interpreter you want to use (usually `$SRC_DIR/bin/python`).
Make sure to add `$ROOT_SRC_DIR/lib` to your `PYTHONPATH` to allow `pyroot` to be imported! You can use e.g. `$SRC_DIR/setup_env.sh` to do that.

Then, activate the virtual environment and install the remaining dependencies.
```
cd $SRC_DIR
source bin/activate
source setup_env.sh
pip install -r requirements.txt
```

## Training

To run a training campaign (assuming you want to locate the output at `$TRAIN_DIR`):
```
mkdir $TRAIN_DIR
cd $SRC_DIR
cp examples/Master.conf $TRAIN_DIR
python RunTrainingCampaign.py --confpath $TRAIN_DIR/Master.conf --nrep 1
```
Here, `nrep` is the number of trainings that should be carried out for each parameter point. `Master.conf` is a configuration file that specifies the settings that are to be used for the training.
You may want to have a look at it and play with the settings. The default values are such as to guarantee a quick completion of the training, but will not achieve competitive sensitivity.

The config file also allows performing sweeps over an arbitrary number of these parameters (in the example used, the Lagrange multiplier lambda is swept). Refer to `utils/ConfigFileSweeper/README.md` for more details.
Warning: the above command will spin up many processes on your local machine. If you have a Condor batch system available, change `submitter` in `base/Configs.py` accordingly.

This repository contains a small training dataset (100000 events per signal and background process) in `examples/training-MadGraphPy8-ATLAS-small.h5`, which is used by default. This should be enough to play
with the method, but not enough to reach optimum performance. A larger dataset is available from the authors upon request.

## Evaluation

To evaluate the classifier and generate the output for an Asimov fit:
```
python RunPrepareHistFitterCampaign.py $TRAIN_DIR/Master_slice_*
```

To perform an Asimov fit:

```
python RunHistFitterCampaign.py $TRAIN_DIR/Master_slice_*
```

Note: for this step, you need to have a local installation of HistFitter available!
Installation instructions are available at http://histfitter.web.cern.ch/histfitter/Software/Install/index.html
Once the installation has completed, you need to correctly set the path to your local HistFitter installation directory
in `setup_env.sh`.

In case you do not want to evaluate Asimov sensitivities, you can skip this step. You will then be unable to run
`MakeGlobalAsimovPlots.py` below.

## Visualisation / Plotting

To produce the summary plots:
```
python MakeGlobalAnalysisPlots.py --plotdir $PLOT_DIR $TRAIN_DIR/Master_slice_* 
python MakeGlobalAsimovPlots.py --plotdir $PLOT_DIR $TRAIN_DIR/Master_slice_* 
```
Here, `$PLOT_DIR` is the directory where the plots should be stored. `MakeGlobalAnalysisPlots` generates generic performance plots, based on the binned significance. `MakeGlobalAsimovPlots` takes the results from the Asimov fit and visualises them.

## A brief guide to the code
The training is done by the modules `training` and `models`. You may especially want to look at `AdversarialEnvironment` (which builds the `TensorFlow` model) and `AdversarialTrainer` which does the actual training. The `plotting` module does exactly what it is supposed to. `dataprep` and `utils` contain code that is used to prepare and preprocess the MC training dataset, and is included here for completeness. The main configuration file is `base/Configs.py`.