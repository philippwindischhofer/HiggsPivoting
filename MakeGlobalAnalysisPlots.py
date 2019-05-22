import os, pickle
from argparse import ArgumentParser

from MakeGlobalPlots import MakeGlobalComparisonPlot

def MakeGlobalAnalysisPlots(plotdir, model_dirs):
    processes = ["Hbb", "Wjets", "Zjets", "diboson", "ttbar"]
    

    for process in processes:
        MakeGlobalComparisonPlot(model_dirs, outpath = os.path.join(plot_dir, cur + ".pdf"), source_basename = cur + ".pkl", 
                                 overlay_paths = [cur_overlay + ".pkl"] if cur_overlay is not None else None, overlay_colors = ["black"], overlay_labels = ["inclusive"])        

if __name__ == "__main__":
    parser = ArgumentParser(description = "create global comparison plots")
    parser.add_argument("--plotdir")
    parser.add_argument("model_dirs", nargs = '+', action = "store")
    args = vars(parser.parse_args())

    MakeGlobalAnalysisPlots(**args)
