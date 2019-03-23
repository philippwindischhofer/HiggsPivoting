# run it with HistFitter.py -w -f -d -D "after,corrMatrix" -F excl -a --userArg input_dir TemplateAnalysisSimple.py

import ROOT, sys
from argparse import ArgumentParser

# HistFitter imports
from configManager import configMgr
from configWriter import fitConfig,Measurement,Channel,Sample
from systematic import Systematic

# read the path of the directory containing the input template histograms
indir = configMgr.userArg

configMgr.doExclusion = True
configMgr.calculatorType = 2
configMgr.testStatType = 3
configMgr.nPoints = 20

configMgr.analysisName = "TemplateAnalysisSimple"
configMgr.outputFileName = "results/{}.root".format(configMgr.analysisName)

# signal regions that are to be used for this fit
region_names = ["SR1", "SR2"]

# names of the individual signal templates
template_names = ["sig", "bkg"]

configMgr.cutsDict["SR1"] = "1."
configMgr.cutsDict["SR2"] = "1."
configMgr.weights = "1."

# define samples
bkg_sample = Sample("bkg", ROOT.kGreen - 9)
bkg_sample.setNormFactor("mu_bkg", 1, 0, 100)
bkg_sample.buildHisto([5, 5], "SR1", "mBB", 0.5)
bkg_sample.buildHisto([5, 1], "SR2", "mBB", 0.5)

sig_sample = Sample("sig", ROOT.kPink)
sig_sample.setNormFactor("mu_sig", 1, 0, 100)
sig_sample.buildHisto([7, 15], "SR1", "mBB", 0.5)
sig_sample.buildHisto([7, 15], "SR2", "mBB", 0.5)

ana = configMgr.addFitConfig("SPlusB")

meas = ana.addMeasurement(name = "ShapeFit", lumi = 1.0, lumiErr = 0.01)
meas.addPOI("mu_sig")
meas.addPOI("mu_bkg")
meas.addParamSetting("Lumi", True, 1) # no luminosity uncertainty

chan1 = ana.addChannel("mBB", ["SR1"], 2, 0.5, 2.5)
chan1.addSample(bkg_sample)
chan1.addSample(sig_sample)

chan2 = ana.addChannel("mBB", ["SR2"], 2, 0.5, 2.5)
chan2.addSample(bkg_sample)
chan2.addSample(sig_sample)

ana.addSignalChannels([chan1, chan2]) # add the signal regions to the fit configuration

# These lines are needed for the user analysis to run
# Make sure file is re-made when executing HistFactory
if configMgr.executeHistFactory:
    if os.path.isfile("data/%s.root" % configMgr.analysisName):
        os.remove("data/%s.root" % configMgr.analysisName) 

