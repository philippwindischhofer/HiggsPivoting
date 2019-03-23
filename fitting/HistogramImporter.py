import ROOT
from ROOT import TFile

class HistogramImporter:

    @staticmethod
    def import_histogram(infile, histogram_name):
        need_to_close = False
        if isinstance(infile, str):
            infile = TFile(infile, 'READ')
            need_to_close = True

        hist = infile.Get(histogram_name)
        hist.SetDirectory(0)

        binvals = []
        edges = []
        
        # exclude over- and underflow bins
        for cur_bin in range(1, hist.GetSize() - 1):
            binvals.append(hist.GetBinContent(cur_bin))
            edges.append(hist.GetBinLowEdge(cur_bin))
        edges.append(hist.GetBinLowEdge(cur_bin + 1))

        if need_to_close:
            infile.Close()

        print(edges)

        # return the bin contents as well as the bin edges
        return binvals, edges
