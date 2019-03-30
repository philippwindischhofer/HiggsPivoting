from delphes.DelphesPreprocessor import DelphesPreprocessor

class Hbb0LepDelphesPreprocessor(DelphesPreprocessor):

    def __init__(self):
        # the original branches needed for all subsequent processing steps
        self.input_branches = ["Event.Weight", "Jet.PT", "Jet.Flavor"]

        # the branches that will finally be exported
        self.output_branches = ["weight"]

    def load(self, infile_path):
        super(Hbb0LepDelphesPreprocessor, self).load(infile_path = infile_path, branches = self.input_branches)

    def process(self, lumi, xsec):
        print("running with lumi = {} fb^-1".format(lumi))
        print("running with xsec = {} pb".format(xsec))

        # get the sum-of-weights of the loaded events
        sow = float(sum(self._extract_column("Event.Weight")))
        print("found SOW = {}".format(sow))

        weight_modifier = lumi * xsec * 1000 / sow # the factor of 1000 converts between fb and pb

        # ensure the correct normalization of these events
        self._add_column("weight", lambda row: row["Event.Weight"][0] * weight_modifier)

        # first, try to flatten the dataframe
        

        return self.df[self.output_branches]
