# run it with HistFitter.py -w -f -d -D "after,corrMatrix" -F excl -a --userArg input_dir TemplateAnalysisSimple.py

import ROOT, sys, glob, string
from ROOT import TColor

# HistFitter imports
from configManager import configMgr
from configWriter import fitConfig,Measurement,Channel,Sample
from systematic import Systematic

from HistogramImporter import HistogramImporter

# read the path of the directory containing the input template histograms
indir = configMgr.userArg

configMgr.doExclusion = False
configMgr.calculatorType = 2
#configMgr.mTOYs = 5000
configMgr.testStatType = 2
configMgr.nPoints = 20

configMgr.analysisName = "ShapeFitNCategoriesBackgroundFixed"
configMgr.outputFileName = "results/{}.root".format(configMgr.analysisName)

# names of the individual input files for the above regions
# Note: these must exist in the directory 'indir' passed above
region_infiles = sorted(glob.glob("region_*.root"))
region_names = ["region" + char for region, char in zip(region_infiles, list(string.ascii_uppercase))]

# names of the individual signal templates available in each region
#sample_names = ["ttbar", "Zjets", "Wjets", "diboson", "singletop", "Hbb"]
sample_names = ["ttbar", "Zjets", "Wjets", "diboson", "Hbb"]
#sample_names = ["ttbar", "Zjets", "Wjets", "Hbb"]
#sample_names = TrainingConfigs.bkg_samples + TrainingConfigs.sig_samples
#normalization_floating = [False, False, False, True]
normalization_floating = [False, False, False, False, True]
#signal_samples = [False, False, False, True]
signal_samples = [False, False, False, False, True]
#template_names = ["ttbar_mBB", "Zjets_mBB", "Wjets_mBB", "diboson_mBB", "singletop_mBB", "Hbb_mBB"]
template_names = [sample_name + "_mBB" for sample_name in sample_names]
template_colors = [TColor.GetColor(255, 204, 0), TColor.GetColor(204, 151, 0), TColor.GetColor(0, 99, 0), 
                   TColor.GetColor(0, 99, 204), TColor.GetColor(204, 204, 204), TColor.GetColor(255, 0, 0)]

# turn off any additional selection cuts
for region_name in region_names:
    configMgr.cutsDict[region_name] = "1."

configMgr.weights = "1."

samples = []
channels = []
POIs = []
signal_sample = None

# prepare the fit configuration
ana = configMgr.addFitConfig("shape_fit")
meas = ana.addMeasurement(name = "shape_fit", lumi = 1.0, lumiErr = 0.01)

# load all MC templates ...
for sample_name, template_name, template_color, is_floating, is_signal in zip(sample_names, template_names, template_colors, normalization_floating, signal_samples):

    cur_sample = Sample(sample_name, template_color)

    if is_floating:
        normalization_name = "mu_" + sample_name
        cur_sample.setNormFactor(normalization_name, 1, 0, 100)

        if is_signal:
            POIs.append(normalization_name)
            signal_sample = cur_sample

    # ... for all regions
    for region_name, region_infile in zip(region_names, region_infiles):
        binvals, edges = HistogramImporter.import_histogram(os.path.join(indir, region_infile), template_name)
        bin_width = edges[1] - edges[0]

        cur_sample.buildHisto(binvals, region_name, "mBB", binLow = edges[0], binWidth = bin_width)

    samples.append(cur_sample)

# also make the (Asimov) data sample
data_sample = Sample("data", ROOT.kBlack)
data_sample.setData()

# in each region, it holds the total event content
for region_name, region_infile in zip(region_names, region_infiles):

    binvals = None
    for sample_name, template_name in zip(sample_names, template_names):
        sample_binvals, edges = HistogramImporter.import_histogram(os.path.join(indir, region_infile), template_name)
        bin_width = edges[1] - edges[0]
        
        if not binvals:
            binvals = sample_binvals
        else:
            binvals = [binval + sample_binval for binval, sample_binval in zip(binvals, sample_binvals)]

    data_sample.buildHisto(binvals, region_name, "mBB", binLow = edges[0], binWidth = bin_width)

# create a Channel object for each analysis region
for region_name in region_names:
    cur_channel = ana.addChannel("mBB", [region_name], nBins = len(binvals), binLow = edges[0], binHigh = edges[-1])

    # add all the samples to it
    for sample in samples:
        cur_channel.addSample(sample)
    cur_channel.addSample(data_sample)

    channels.append(cur_channel)

# add all POIs to the fit configuration
for POI in POIs:
    meas.addPOI(POI)

# ignore luminosity uncertainty
meas.addParamSetting("Lumi", True, 1)

# finally, add all the channels
ana.addSignalChannels(channels)
ana.setSignalSample(signal_sample)

# remove temporary files
if configMgr.executeHistFactory:
    if os.path.isfile("data/{}.root".format(configMgr.analysisName)):
        os.remove("data/{}.root".format(configMgr.analysisName))

