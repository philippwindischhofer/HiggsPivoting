import os, pickle
from argparse import ArgumentParser
from plotting.PerformancePlotter import PerformancePlotter

def _load_dict(path):
    retdict = {}

    try:
        with open(path, 'rb') as infile:
            retdict = pickle.load(infile)

    except FileNotFoundError:
        print("file {} not found, ignoring".format(path))

    return retdict

def _load_metadata(path, section):
    from configparser import ConfigParser
    gconfig = ConfigParser()
    gconfig.read(path)
    pars_dict = {key: val for key, val in gconfig[section].items()}

    return pars_dict

def MakeMIEvolutionPlot(plotdir, workdirs):

    plotdata_x = []
    plotdata_y = []
    labels = []

    traces_to_plot = ["binned_MI_tukey", "binned_MI_cellucci", "binned_MI_cellucci_approximated", "binned_MI_bendat_piersol"]#, "neural_MI"]
    trace_labels = ["Tukey", "Cellucci", "Cellucci approx", "Bendat-Piersol"]
    style_library = ["-", "--", ":", "-."]
    style_labels = {}
    styles = []

    color_labels = {}
    color_library = ['orange', 'royalblue', 'green']
    colors = []

    # load the traces as well as the metadata (lambda of the run)
    for workdir, cur_color in zip(workdirs, color_library):
        tracedict = _load_dict(os.path.join(workdir, "training_evolution.pkl"))
        anadict = _load_dict(os.path.join(workdir, "anadict.pkl"))
        if not anadict:
            # did not find it in this format, look into the metadata directly
            anadict = _load_metadata(os.path.join(workdir, "meta.conf"), section = "AdversarialEnvironment")

        x_data = tracedict["batch"]
        for trace, trace_label, style in zip(traces_to_plot, trace_labels, style_library):
            y_data = tracedict[trace]
            plotdata_x.append(x_data)
            plotdata_y.append(y_data)
            labels.append(trace)
            styles.append(style)
            colors.append(cur_color)

            style_labels[style] = trace_label

        cur_lambda = anadict["lambda"]
        color_labels[cur_color] = '$\lambda = {}$'.format(cur_lambda)

    outfile_path = os.path.join(plotdir, "MI_evolution.pdf")
    PerformancePlotter._simple_plot(plotdata_x, plotdata_y, colors, styles, style_labels, color_labels, outfile_path, xlabel = "minibatch", ylabel = r'$\hat{MI}$')

if __name__ == "__main__":
    parser = ArgumentParser(description = "show evolution of MI as the training progresses")
    parser.add_argument("--plotdir", action = "store")
    parser.add_argument("--workdirs", nargs = '+', action = "store")
    args = vars(parser.parse_args())

    MakeMIEvolutionPlot(**args)
