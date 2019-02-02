import pandas as pd
import numpy as np
import pickle

from training.Trainer import Trainer

class AdversarialTrainer(Trainer):

    def __init__(self, training_pars):
        super(AdversarialTrainer, self).__init__(training_pars)
        self.statistics_dict = {}

    def sample_from(self, sources_sig, weights_sig, sources_bkg, weights_bkg, initial_req, sow_target = 1.0, target_tol = 0.1, batch_limit = 10000):
        # Note: the length of the individual consituents of sources_sig and sources_bkg must have the
        # same length! (will usually be the case since they correspond to the same events anyways)

        # need to sample from signal and background in such a way that the sum of weights
        # of either source is very similar (or ideally, identical)
        inds_sig = np.random.choice(len(weights_sig), initial_req)
        inds_bkg = np.random.choice(len(weights_bkg), initial_req)

        # resample as long as the sums-of-weights of signal and background events are equal
        # to each other within some tolerance
        while True:
            sampled_weights_sig = weights_sig[inds_sig]
            sampled_weights_bkg = weights_bkg[inds_bkg]

            sow_sig = np.sum(sampled_weights_sig)
            sow_bkg = np.sum(sampled_weights_bkg)

            # have reached the SOW target for both signal and background, stop
            if (sow_sig > sow_target - target_tol and sow_bkg > sow_target - target_tol) or (len(inds_sig) + len(inds_bkg) > batch_limit):
                break

            # get a good guess for how many more samples will be needed separately for signal and background
            sample_request_bkg = int(min(0.1 * max(sow_target - sow_bkg, 0.0) / abs(sow_bkg) * len(inds_bkg), batch_limit / 4))
            sample_request_sig = int(min(0.1 * max(sow_target - sow_sig, 0.0) / abs(sow_sig) * len(inds_sig), batch_limit / 4))
            
            # get the new samples and append them
            inds_sampled_bkg = np.random.choice(len(weights_bkg), sample_request_bkg)
            inds_sampled_sig = np.random.choice(len(weights_sig), sample_request_sig)

            inds_sig = np.concatenate([inds_sig, inds_sampled_sig], axis = 0)
            inds_bkg = np.concatenate([inds_bkg, inds_sampled_bkg], axis = 0)

        sampled_weights_sig = weights_sig[inds_sig]
        sampled_weights_bkg = weights_bkg[inds_bkg]

        sampled_sig = [cur_source[inds_sig] for cur_source in sources_sig]
        sampled_bkg = [cur_source[inds_bkg] for cur_source in sources_bkg]

        sampled = [np.concatenate([sample_sig, sample_bkg], axis = 0) for sample_sig, sample_bkg in zip(sampled_sig, sampled_bkg)]
        sampled_weights = np.concatenate([sampled_weights_sig, sampled_weights_bkg], axis = 0)

        return sampled, sampled_weights

    # overload the 'train' method here
    def train(self, env, number_batches, traindat_sig, traindat_bkg, nuisances_sig, nuisances_bkg, weights_sig, weights_bkg):
        data_sig = traindat_sig
        data_bkg = traindat_bkg

        labels_sig = np.ones(len(data_sig))
        labels_bkg = np.zeros(len(data_bkg))

        # also prepare arrays with the full training dataset
        data_train = np.concatenate([data_sig, data_bkg], axis = 0)
        nuisances_train = np.concatenate([nuisances_sig, nuisances_bkg], axis = 0)

        # initialize the environment
        env.init(data_train = data_train, data_nuisance = nuisances_train)

        # pre-train the adversary
        print("pretraining adversarial network for {} batches".format(self.training_pars["pretrain_batches"]))
        for batch in range(self.training_pars["pretrain_batches"]):
            # sample coherently from (data, nuisance, label) tuples
            (data_batch, nuisances_batch, labels_batch), weights_batch = self.sample_from([data_sig, nuisances_sig, labels_sig], weights_sig, [data_bkg, nuisances_bkg, labels_bkg], weights_bkg, 
                                                                                          initial_req = 100, sow_target = self.training_pars["sow_target"])

            env.train_adversary(data_step = data_batch, nuisances_step = nuisances_batch, labels_step = labels_batch, weights_step = weights_batch, batchnum = batch)
            env.dump_loss_information(data = data_batch, nuisances = nuisances_batch, labels = labels_batch, weights = weights_batch)

        print("pretraining complete!")

        print("starting training:")
        for batch in range(number_batches):
            # sample coherently from (data, nuisance, label) tuples
            (data_batch, nuisances_batch, labels_batch), weights_batch = self.sample_from([data_sig, nuisances_sig, labels_sig], weights_sig, [data_bkg, nuisances_bkg, labels_bkg], weights_bkg, 
                                                                                          initial_req = 100, sow_target = self.training_pars["sow_target"])

            env.train_adversary(data_step = data_batch, nuisances_step = nuisances_batch, labels_step = labels_batch, weights_step = weights_batch, batchnum = batch)
            env.train_step(data_step = data_batch, nuisances_step = nuisances_batch, labels_step = labels_batch, weights_step = weights_batch, batchnum = batch)

            # callbacks to keep track of the parameter evolution during training
            stat_dict_cur = env.get_model_statistics(data = data_batch, nuisances = nuisances_batch, labels = labels_batch, weights = weights_batch)
            stat_dict_cur["batch"] = batch
            
            for key, val in stat_dict_cur.items():
                if not key in self.statistics_dict:
                    self.statistics_dict[key] = []
                self.statistics_dict[key].append(val)

            # some status printouts
            if not batch % self.training_pars["printout_interval"]:
                print("batch {}:".format(batch))
                print("dynamic batch size = " + str(len(data_batch)))
                env.dump_loss_information(data = data_batch, nuisances = nuisances_batch, labels = labels_batch, weights = weights_batch)
                print("stat_dict = " + str(stat_dict_cur))

    def save_training_statistics(self, filepath):
        with open(filepath, "wb") as outfile:
            pickle.dump(self.statistics_dict, outfile)
