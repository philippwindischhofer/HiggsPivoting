import ROOT
from ROOT import TFile, RooWorkspace, Util
import pickle, os
from argparse import ArgumentParser

# reads the output of a HistFitter run and converts it into
# a pickled dict that can be comfortably read in with a script
# along the lines of MakeGlobalXXXPlots
def ConvertFitResult(infile_path, outfile_path):
    # dictionary with output fit parameters
    # contains data in the format (central_value, uncertainty_down, uncertainty_up)
    pardict = {}

    # first, try to read the RooFit workspace after the fit
    if os.path.isfile(infile_path):
        infile = TFile(infile_path, 'READ')
        w = infile.Get('w')

        # load the fit result
        fit_result_name = "RooExpandedFitResult_afterFit"
        result = w.obj(fit_result_name)

        fpf = result.floatParsFinal() 
        fpi = result.floatParsInit()

        for ind in range(fpf.getSize()):
            parname = fpf[ind].GetName()

            fp = fpf[ind]
            fpv  = fp.getVal()
            fpel = fp.getErrorLo()
            fpeh = fp.getErrorHi()
            
            pardict[parname] = (fpv, fpel, fpeh)

    # pickle the parameter dictionary into a file
    with open(outfile_path, "wb") as outfile:
        pickle.dump(pardict, outfile)

    infile.Close()

def ConvertHypothesisTestResult(infile_path, outfile_path):
    hypodict = {}

    if os.path.isfile(infile_path):
        infile = TFile(infile_path, 'READ')
        res = infile.Get('discovery_htr_Hbb')

        sig = res.Significance()
        hypodict["discovery_sig"] = sig

    # pickle the parameter dictionary into a file
    with open(outfile_path, "wb") as outfile:
        pickle.dump(hypodict, outfile)

    infile.Close()

if __name__ == "__main__":
    parser = ArgumentParser(description = "converts HistFitter output")
    parser.add_argument("--infile", action = "store", dest = "infile_path")
    parser.add_argument("--outfile", action = "store", dest = "outfile_path")
    parser.add_argument("--mode", action = "store", dest = "mode")
    args = vars(parser.parse_args())

    infile_path = args["infile_path"]
    outfile_path = args["outfile_path"]
    mode = args["mode"]

    if mode == "fit":
        ConvertFitResult(infile_path, outfile_path)
    elif mode == "hypotest":
        ConvertHypothesisTestResult(infile_path, outfile_path)
