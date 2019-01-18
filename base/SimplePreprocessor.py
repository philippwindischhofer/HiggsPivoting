import pandas as pd
import numpy as np

from base.Preprocessor import Preprocessor

class SimplePreprocessor(Preprocessor):
    
    def __init__(self, data_branches, cuts):
        self.data_branches = data_branches
        self.cuts = cuts

    def process_generator(self, gen, rettype = 'np'):
        cut_samples = []
        for cut in self.cuts:
            cut_samples.append(pd.DataFrame())

        for chunk in gen:
            pre_chunks = self.process(chunk)
            
            for ind, pre_chunk in enumerate(pre_chunks):
                if len(pre_chunk) > 0:
                    cut_samples[ind] = pd.concat([cut_samples[ind], pre_chunk])
        
        if rettype == 'pd':
            return cut_samples
        elif rettype == 'np':
            return [self._as_matrix(cur) for cur in cut_samples]

    def process(self, chunk):
        cut_chunks = []

        for cut in self.cuts:
            cut_chunks.append(self._rowcol_cut(chunk, cut, self.data_branches))

        return cut_chunks

    def _rowcol_cut(self, chunk, row_cut, cols):
        chunk = chunk.loc[chunk.apply(row_cut, axis = 1)]
        chunk = chunk.loc[:, cols]
        return chunk

    def _as_matrix(self, df):
        return np.array(df.values)
