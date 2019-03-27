import ROOT
from ROOT import TFile, RooWorkspace, Util
import pickle, os
from argparse import ArgumentParser

# reads the output of a HistFitter run and converts it into
# a pickled dict that can be comfortably read in with a script
# along the lines of MakeGlobalXXXPlots
def ConvertFitResult(infile_path, outfile_path, outkey):
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
            
            pardict[parname + outkey] = (fpv, fpel, fpeh)

    # pickle the parameter dictionary into a file
    with open(outfile_path, "wb") as outfile:
        pickle.dump(pardict, outfile)

    infile.Close()

def ConvertHypothesisTestResult(infile_path, outfile_path, outkey):
    hypodict = {}

    if os.path.isfile(infile_path):
        infile = TFile(infile_path, 'READ')
        res = infile.Get('discovery_htr_Hbb')

        sig = res.Significance()
        hypodict[outkey] = sig

        infile.Close()

    # pickle the parameter dictionary into a file
    # first, if the requested file already exists, make sure to append
    # the current data to the dictionary stored there
    try:
        with open(outfile_path, "rb") as outfile:
            hypodict_old = pickle.load(outfile)

        hypodict.update(hypodict_old) # merge the two dictionaries
        print("output file already exists, will append data")
    except IOError:
        print("output file does not yet exist")

    # the output file does not exist, create it
    with open(outfile_path, "wb") as outfile:
        pickle.dump(hypodict, outfile)

if __name__ == "__main__":
    parser = ArgumentParser(description = "converts HistFitter output")
    parser.add_argument("--infile", action = "store", dest = "infile_path")
    parser.add_argument("--outfile", action = "store", dest = "outfile_path")
    parser.add_argument("--outkey", action = "store", dest = "outkey")
    parser.add_argument("--mode", action = "store", dest = "mode")
    args = vars(parser.parse_args())

    infile_path = args["infile_path"]
    outfile_path = args["outfile_path"]
    outkey = args["outkey"]
    mode = args["mode"]

    if mode == "fit":
        ConvertFitResult(infile_path, outfile_path, outkey)
    elif mode == "hypotest":
        ConvertHypothesisTestResult(infile_path, outfile_path, outkey)
