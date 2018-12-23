import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

class PerformancePlotter:

    @staticmethod
    def _II_plot(perfdicts, outpath):
        PerformancePlotter._perfdict_plot(perfdicts, xquant = "I(f,label)", yquant = "I(f,nu)", xlabel = r'$I(f, \mathrm{label})$', ylabel = r'$I(f, \nu)$', legquant = "lambda", legquantlabel = r'$\lambda$', outfile = os.path.join(outpath, "II_plot.pdf"))

    @staticmethod
    def _AUC_sig_sq_plot(perfdicts, outpath):
        PerformancePlotter._perfdict_plot(perfdicts, xquant = "ROCAUC", yquant = "sig_sq_diff", xlabel = "AUC", ylabel = "sig_sq_diff_05", legquant = "lambda", legquantlabel = r'$\lambda$', outfile = os.path.join(outpath, "AUC_sig_sq_plot.pdf"))

    @staticmethod
    def _AUC_bkg_sq_plot(perfdicts, outpath):
        PerformancePlotter._perfdict_plot(perfdicts, xquant = "ROCAUC", yquant = "bkg_sq_diff", xlabel = "AUC", ylabel = "bkg_sq_diff_05", legquant = "lambda", legquantlabel = r'$\lambda$', outfile = os.path.join(outpath, "AUC_bkg_sq_plot.pdf"))

    @staticmethod
    def _perfdict_plot(perfdicts, xquant, yquant, xlabel, ylabel, legquant, legquantlabel, outfile):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for ind, perfdict in enumerate(perfdicts):
            ax.scatter(perfdict[xquant], perfdict[yquant], color = plt.cm.viridis(float(ind) / len(perfdict)), label = r'{}$ = {:.2f}$'.format(legquantlabel, float(perfdict[legquant])))

        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.tight_layout()
        
        fig.savefig(outfile)
        plt.close()        

    @staticmethod
    def plot(perfdicts, outpath):
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        PerformancePlotter._II_plot(perfdicts, outpath)
        PerformancePlotter._AUC_sig_sq_plot(perfdicts, outpath)
        PerformancePlotter._AUC_bkg_sq_plot(perfdicts, outpath)
