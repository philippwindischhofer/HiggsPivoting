import os, glob
from argparse import ArgumentParser

from plotting.RatioPlotter import RatioPlotter

def MakeDistributionRatioPlots(outdir, indir_a, indir_b):
    # first, need to find the plots that are available in both input directories
    plots_a = [os.path.basename(cur) for cur in glob.glob(os.path.join(indir_a, "*.pkl"))]
    plots_b = [os.path.basename(cur) for cur in glob.glob(os.path.join(indir_b, "*.pkl"))]

    plots_common = list(set(plots_a).intersection(plots_b))

    for cur_plot in plots_common:
        infile_a = os.path.join(indir_a, cur_plot)
        infile_b = os.path.join(indir_b, cur_plot)

        plot_name = os.path.splitext(cur_plot)[0]
        outfile = os.path.join(outdir, plot_name + ".pdf")

        RatioPlotter.histogram_ratio_plot(infile_a, infile_b, outfile, name_a = "ATLAS", name_b = "MadGraph + Delphes", title = plot_name)

if __name__ == "__main__":
    parser = ArgumentParser(description = "compares two sets of plots")
    parser.add_argument("--outdir", action = "store", dest = "outdir")
    parser.add_argument("indirs", nargs = "+", action = "store")
    args = vars(parser.parse_args())

    outdir = args["outdir"]
    indir_a = args["indirs"][0]
    indir_b = args["indirs"][1]

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    MakeDistributionRatioPlots(outdir, indir_a, indir_b)
