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
        self.df.reset_index(drop = True)

    def process(self):
        """ do nothing by default; all the action happens in the overriding methods """
        return self.df

    def _add_column(self, column_name, fill_lambda):
        self.df[column_name] = self.df.apply(fill_lambda, axis = 1)

    def _drop_columns(self, to_drop):
        self.df.drop(columns = to_drop)

    def _select(self, selection_lambda):
        self.df = self.df[self.df.apply(selection_lambda, axis = 1)]

    def _extract_column(self, column_name):
        return self.df[column_name]
