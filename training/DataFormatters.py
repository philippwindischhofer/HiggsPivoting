class TrainingSample:    
    def __init__(self, data, nuis, weights, labels):
        self.data = data
        self.nuis = nuis
        self.weights = weights
        self.labels = labels

    @classmethod
    def fromTable(cls, table, is_signal = False):
        from DatasetExtractor import TrainNuisAuxSplit
        import numpy as np
        cur_data, cur_nuis, cur_weights = TrainNuisAuxSplit(table)

        if is_signal:
            cur_labels = np.ones(len(cur_data))
        else:
            cur_labels = np.zeros(len(cur_data))
            
        return cls(cur_data, cur_nuis, cur_weights, cur_labels)

class only_nJ:
    
    def __init__(self, nJ):
        self.nJ = nJ

    def _extract_nJ(self, sample, nJ):
        sample_nJ = sample.loc[sample["nJ"] == self.nJ]
        return sample_nJ

    def format_as_TrainingSample(self, data, is_signal = False):
        data_nJ = self._extract_nJ(data, self.nJ)
        return TrainingSample.fromTable(data_nJ, is_signal)

    def get_formatted_indices(self, data):
        return self._extract_nJ(data, self.nJ).index

class only_2j(only_nJ):

    def __init__(self):
        super().__init__(nJ = 2)

class only_3j(only_nJ):

    def __init__(self):
        super().__init__(nJ = 3)


