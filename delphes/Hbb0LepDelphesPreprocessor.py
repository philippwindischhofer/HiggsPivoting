from delphes.DelphesPreprocessor import DelphesPreprocessor

class Hbb0LepDelphesPreprocessor(DelphesPreprocessor):

    def __init__(self):
        # the original branches needed for all subsequent processing steps
        self.input_branches = ["Event.Weight", "Jet.PT", "Jet.Flavor"]

    def load(self, infile_path):
        super(Hbb0LepDelphesPreprocessor, self).load(infile_path = infile_path, branches = self.input_branches)

    def process(self):
        return self.df
