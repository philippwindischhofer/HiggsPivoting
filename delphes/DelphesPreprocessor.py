import uproot as ur

class DelphesPreprocessor:
    
    def __init__(self):
        self.df = None

    def load(self, infile_path, branches):
        print("loading the following branches:")
        for branch in branches:
            print(branch)

        tree = ur.open(infile_path)["Delphes"]
        self.df = tree.pandas.df(branches, flatten = False)

    def process(self):
        pass

    def _add_column(self, column_name, fill_lambda):
        pass

    def _drop_column(self):
        pass
