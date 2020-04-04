import numpy as np
import training.BatchSamplers as BatchSamplers

class AdversarialModelTrainer:

    def __init__(self, model, data_formatter, batch_sampler):
        self.model = model
        self.data_formatter = data_formatter
        self.batch_sampler = batch_sampler

    def train(self, trainsamples_sig, trainsamples_bkg, valsamples_sig, valsamples_bkg):
        
        # first, format the training and validation datasets in the way required by the model
        trainsamples_sig_formatted = [self.data_formatter(cur_sample, is_signal = True) for cur_sample in trainsamples_sig]
        trainsamples_bkg_formatted = [self.data_formatter(cur_sample, is_signal = False) for cur_sample in trainsamples_bkg]

        valsamples_sig_formatted = [self.data_formatter(cur_sample, is_signal = True) for cur_sample in valsamples_sig]
        valsamples_bkg_formatted = [self.data_formatter(cur_sample, is_signal = False) for cur_sample in valsamples_bkg]

        (data_all, nuis_all, labels_all), weights_all = BatchSamplers.all(trainsamples_sig_formatted + trainsamples_bkg_formatted)
        
        self.model.init(data_all, nuis_all)

        sampled_sig, weights_sig = self.batch_sampler(trainsamples_sig_formatted, size = 10)
        sampled_bkg, weights_bkg = self.batch_sampler(trainsamples_bkg_formatted, size = 10)

        (data_batch, nuis_batch, labels_batch), weights_batch = self._combine_samples(sampled_sig, weights_sig, sampled_bkg, weights_bkg)

        self.model.train_adversary(data_batch, nuis_batch, labels_batch, weights_batch, batchnum = 1)

        res = self.model.predict(data_batch)
        print(res)

    def _combine_samples(self, samples_sig, weights_sig, samples_bkg, weights_bkg):

        data_combined = [np.concatenate([cur_sig_sampled, cur_bkg_sampled], axis = 0) for cur_sig_sampled, cur_bkg_sampled in zip(samples_sig, samples_bkg)]
        weights_combined = np.concatenate([weights_sig, weights_bkg], axis = 0)
        
        return data_combined, weights_combined
