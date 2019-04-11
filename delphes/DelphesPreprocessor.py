import uproot as ur

class DelphesPreprocessor:
    
    def __init__(self):
        self.df = None

    def load(self, infile_path, branches):
        try:
            tree = ur.open(infile_path)["Delphes"]
            self.df = tree.pandas.df(branches, flatten = False)
            self.df.reset_index(drop = True)
        except:
            # the Delphes tree is not available for some reason, do nothing
            print("file '{}' not found or problem reading it!".format(infile_path))
            self.df = None

    def process(self):
        """ do nothing by default; all the action happens in the overriding methods """
        return self.df

    def _add_column(self, column_name, fill_lambda):
        if self.df is not None:
            if len(self.df) >= 1:
                self.df[column_name] = self.df.apply(fill_lambda, axis = 1)
            else:
                self.df = None

    def _drop_columns(self, to_drop):
        if self.df is not None:
            self.df.drop(columns = to_drop)

    def _select(self, selection_lambda):
        if self.df is not None:
            if len(self.df) > 0:
                self.df = self.df[self.df.apply(selection_lambda, axis = 1)]

    def _extract_column(self, column_name):
        if self.df is not None:
            if len(self.df) > 0:
                return self.df[column_name]
        else:
            return None
