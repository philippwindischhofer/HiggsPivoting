from Preprocessor import Preprocessor
import pandas as pd
import numpy as np

class SimplePreprocessor(Preprocessor):
    
    def __init__(self, data_branches, sig_cut, bkg_cut):
        self.data_branches = data_branches
        self.sig_cut = sig_cut
        self.bkg_cut = bkg_cut

    def process_generator(self, gen):
        sig_data = pd.DataFrame()
        bkg_data = pd.DataFrame()

        cnt = 0
        for chunk in gen:
            sig_chunk, bkg_chunk = self.process(chunk)

            if len(sig_chunk) > 0:
                sig_data = pd.concat([sig_data, sig_chunk])
            if len(bkg_chunk) > 0:
                bkg_data = pd.concat([bkg_data, bkg_chunk])

            cnt += 1
            if cnt > 4:
                break
        
        return self._as_matrix(sig_data), self._as_matrix(bkg_data)

    def process(self, chunk):
        #print(chunk)
        sig_chunk = self._rowcol_cut(chunk, self.sig_cut, self.data_branches)
        bkg_chunk = self._rowcol_cut(chunk, self.bkg_cut, self.data_branches)

        return sig_chunk, bkg_chunk

    def _rowcol_cut(self, chunk, row_cut, cols):
        chunk = chunk.loc[chunk.apply(row_cut, axis = 1)]
        chunk = chunk.loc[:, cols]
        return chunk

    def _as_matrix(self, df):
        return np.array(df.values)
