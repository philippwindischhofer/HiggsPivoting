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

def only_nJ(data, nJ, is_signal = False):
    data_nJ = _extract_nJ(data, nJ)
    return TrainingSample.fromTable(data_nJ, is_signal)

def only_2j(data, is_signal = False):
    return only_nJ(data, nJ = 2, is_signal = is_signal)

def only_3j(data, is_signal = False):
    return only_nJ(data, nJ = 3, is_signal = is_signal)

def _extract_nJ(sample, nJ):
    sample_nJ = sample.loc[sample["nJ"] == nJ]
    return sample_nJ

