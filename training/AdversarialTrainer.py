import pandas as pd
import numpy as np
import pickle

from training.Trainer import Trainer
from base.Configs import TrainingConfig

class AdversarialTrainer(Trainer):

    def __init__(self, training_pars):
        super(AdversarialTrainer, self).__init__(training_pars)
        self.statistics_dict = {}

    # sample a fixed number of events from 'sources'
    def sample_from(self, sources, weights, req = 100):
        inds = np.random.choice(len(weights), req)

        sampled_weights = weights[inds]
        sampled_data = [cur_source[inds] for cur_source in sources]

        return sampled_data, sampled_weights        

    # draw a certain number of events from 'components'. If 'equalize_fraction' is set to 'False', the
    # original proportions of the individual components are kept.
    def sample_from_components(self, components, weights, batch_size = 1000, sampling_pars = {}):
        sampling_pars.setdefault("sampling_fractions", None)

        # Note: this effectively transposes the nested lists such that the iterations become easier
        sources = list(map(list, zip(*components)))        

        # first, compute the number of events available for each signal / background component ...
        nevents = [len(cur) for cur in weights]
        total_nevents = np.sum(nevents)

        # ... and also the SOW for each
        if sampling_pars["sampling_fractions"] is not None:
            # make sure they sum up to one
            sampling_pars["sampling_fractions"] /= np.sum(sampling_pars["sampling_fractions"])

            # pretend that each component came with an equal SOW to start with
            # SOWs = [1 / len(weights) for cur in weights]
            SOWs = np.array(sampling_pars["sampling_fractions"])
            print("using explicit sampling fractions: {}".format(SOWs))
        else:
            # keep the original proportions
            SOWs = [np.sum(cur) for cur in weights]
            total_SOW = np.sum(SOWs)
            SOWs /= total_SOW # normalize total SOW to 1
            print("using original sampling fractions: {}".format(SOWs))

        total_SOW = 0

        # now, compute the number of events that should be sampled from each signal / background component:
        # sample them in the same proportions with which they appear in the training dataset ...
        sampled_data = []
        sampled_weights = []
        for cur_source, cur_weights, cur_nevents in zip(sources, weights, nevents):
            #cur_sampled_data, cur_sampled_weights = self.sample_from(cur_source, cur_weights, req = int(batch_size * cur_nevents / total_nevents))
            cur_sampled_data, cur_sampled_weights = self.sample_from(cur_source, cur_weights, req = int(batch_size / len(sources)))
            sampled_data.append(cur_sampled_data)
            sampled_weights.append(cur_sampled_weights)
            total_SOW += np.sum(cur_sampled_weights)
        
        print("---")
        print("fractions before rescaling")
        print([np.sum(cur) / total_SOW for cur in sampled_weights])

        # # ... and normalize them such that their SOWs are in the correct relation to each other
        # for cur, cur_SOW in enumerate(SOWs):
        #     sampled_weights[cur] *= cur_SOW / np.sum(sampled_weights[cur]) # each batch will have a total SOW of 1

        # just normalize them to have the same SOW as the signal
        for cur, cur_SOW in enumerate(SOWs):
            sampled_weights[cur] /= total_SOW

        print("fractions after rescaling")
        print([np.sum(cur) for cur in sampled_weights])
        print("---")

        # transpose them back for easy concatenation
        sampled_sources = list(map(list, zip(*sampled_data)))

        # perform the concatenation ...
        sampled = [np.concatenate(cur, axis = 0) for cur in sampled_sources]
        sampled_weights = np.concatenate(sampled_weights, axis = 0)

        # ... and return
        return sampled, sampled_weights

    def sample_batch(self, sources_sig, weights_sig, sources_bkg, weights_bkg, size = 1000, sig_sampling_pars = {}, bkg_sampling_pars = {}):
        # sample a halfbatch size full of events from signal, and from background, individually normalized to unit SOW
        # and keeping their relative proportions fixed
        sig_sampled, sig_weights = self.sample_from_components(sources_sig, weights_sig, batch_size = size // 2, sampling_pars = sig_sampling_pars)
        print("sampled {} events from signal components with total SOW = {}".format(len(sig_weights), np.sum(sig_weights)))

        bkg_sampled, bkg_weights = self.sample_from_components(sources_bkg, weights_bkg, batch_size = size // 2, sampling_pars = bkg_sampling_pars)
        print("sampled {} events from background components with total SOW = {}".format(len(bkg_weights), np.sum(bkg_weights)))

        # concatenate the individual components
        data_combined = [np.concatenate([cur_sig_sampled, cur_bkg_sampled], axis = 0) for cur_sig_sampled, cur_bkg_sampled in zip(sig_sampled, bkg_sampled)]
        weights_combined = np.concatenate([sig_weights, bkg_weights], axis = 0) * 100

        return data_combined, weights_combined

    def sample_batch_SOW(self, sources_sig, weights_sig, sources_bkg, weights_bkg, size = 1.0, sig_sampling_pars = {}, bkg_sampling_pars = {}):
        # this function (at least for now) only support one global set of sampling parameters
        sampling_pars = sig_sampling_pars
        sampling_pars.setdefault("initial_req", 100)
        sampling_pars.setdefault("target_tol", 0.1)
        sampling_pars.setdefault("batch_limit", 10000)

        sow_target = size # in this method, interpret the 'size' parameter as the total SOW of the requested batch

        # Note: the length of the individual consituents of sources_sig and sources_bkg must have the
        # same length! (will usually be the case since they correspond to the same events anyways)
        sources_sig = [np.concatenate(cur, axis = 0) for cur in sources_sig]
        sources_bkg = [np.concatenate(cur, axis = 0) for cur in sources_bkg]
        weights_sig = np.concatenate(weights_sig, axis = 0)
        weights_bkg = np.concatenate(weights_bkg, axis = 0)

        # need to sample from signal and background in such a way that the sum of weights
        # of either source is very similar (or ideally, identical)
        inds_sig = np.random.choice(len(weights_sig), sampling_pars["initial_req"])
        inds_bkg = np.random.choice(len(weights_bkg), sampling_pars["initial_req"])

        # resample as long as the sums-of-weights of signal and background events are equal
        # to each other within some tolerance
        while True:
            sampled_weights_sig = weights_sig[inds_sig]
            sampled_weights_bkg = weights_bkg[inds_bkg]

            sow_sig = np.sum(sampled_weights_sig)
            sow_bkg = np.sum(sampled_weights_bkg)

            # have reached the SOW target for both signal and background, stop
            if (sow_sig > sow_target - sampling_pars["target_tol"] and sow_bkg > sow_target - sampling_pars["target_tol"]) or (len(inds_sig) + len(inds_bkg) > sampling_pars["batch_limit"]):
                break

            # get a good guess for how many more samples will be needed separately for signal and background
            sample_request_bkg = int(min(0.1 * max(sow_target - sow_bkg, 0.0) / abs(sow_bkg) * len(inds_bkg), sampling_pars["batch_limit"] / 4))
            sample_request_sig = int(min(0.1 * max(sow_target - sow_sig, 0.0) / abs(sow_sig) * len(inds_sig), sampling_pars["batch_limit"] / 4))
            
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

    def _get_nJ_component(self, inlist, auxlist, nJ = 2):
        outlist = []
        for sample, aux_sample in zip(inlist, auxlist):
            nJ_cut = (aux_sample[:, TrainingConfig.other_branches.index("nJ")] == nJ)
            outlist.append(sample[nJ_cut])

        return outlist

    # overload the 'train' method here
    def train(self, env, number_batches, traindat_sig, traindat_bkg, nuisances_sig, nuisances_bkg, weights_sig, weights_bkg, auxdat_sig, auxdat_bkg, sig_sampling_pars = {}, bkg_sampling_pars = {}):
        data_sig = traindat_sig
        data_bkg = traindat_bkg

        # prepare the labels for each signal / background component
        labels_sig = [np.ones(len(cur_data_sig)) for cur_data_sig in data_sig]
        labels_bkg = [np.zeros(len(cur_data_bkg)) for cur_data_bkg in data_bkg]

        # separate them into their 2j/3j components
        data_sig_2j = self._get_nJ_component(data_sig, auxdat_sig, nJ = 2)
        data_sig_3j = self._get_nJ_component(data_sig, auxdat_sig, nJ = 3)

        nuisances_sig_2j = self._get_nJ_component(nuisances_sig, auxdat_sig, nJ = 2)
        nuisances_sig_3j = self._get_nJ_component(nuisances_sig, auxdat_sig, nJ = 3)

        labels_sig_2j = self._get_nJ_component(labels_sig, auxdat_sig, nJ = 2)
        labels_sig_3j = self._get_nJ_component(labels_sig, auxdat_sig, nJ = 3)

        auxdat_sig_2j = self._get_nJ_component(auxdat_sig, auxdat_sig, nJ = 2)
        auxdat_sig_3j = self._get_nJ_component(auxdat_sig, auxdat_sig, nJ = 3)

        weights_sig_2j = self._get_nJ_component(weights_sig, auxdat_sig, nJ = 2)
        weights_sig_3j = self._get_nJ_component(weights_sig, auxdat_sig, nJ = 3)

        data_bkg_2j = self._get_nJ_component(data_bkg, auxdat_bkg, nJ = 2)
        data_bkg_3j = self._get_nJ_component(data_bkg, auxdat_bkg, nJ = 3)

        nuisances_bkg_2j = self._get_nJ_component(nuisances_bkg, auxdat_bkg, nJ = 2)
        nuisances_bkg_3j = self._get_nJ_component(nuisances_bkg, auxdat_bkg, nJ = 3)

        labels_bkg_2j = self._get_nJ_component(labels_bkg, auxdat_bkg, nJ = 2)
        labels_bkg_3j = self._get_nJ_component(labels_bkg, auxdat_bkg, nJ = 3)

        auxdat_bkg_2j = self._get_nJ_component(auxdat_bkg, auxdat_bkg, nJ = 2)
        auxdat_bkg_3j = self._get_nJ_component(auxdat_bkg, auxdat_bkg, nJ = 3)

        weights_bkg_2j = self._get_nJ_component(weights_bkg, auxdat_bkg, nJ = 2)
        weights_bkg_3j = self._get_nJ_component(weights_bkg, auxdat_bkg, nJ = 3)

        # also prepare arrays with the full training dataset
        comb_data_sig = np.concatenate(data_sig, axis = 0)
        comb_data_bkg = np.concatenate(data_sig, axis = 0)
        comb_auxdata_sig = np.concatenate(auxdat_sig, axis = 0)
        comb_auxdata_bkg = np.concatenate(auxdat_bkg, axis = 0)
        comb_data_train = np.concatenate([comb_data_sig, comb_data_bkg], axis = 0)
        comb_nuisances_sig = np.concatenate(nuisances_sig, axis = 0)
        comb_nuisances_bkg = np.concatenate(nuisances_bkg, axis = 0)
        nuisances_train = np.concatenate([comb_nuisances_sig, comb_nuisances_bkg], axis = 0)

        # initialize the environment
        env.init(data_train = comb_data_train, data_nuisance = nuisances_train)

        print(self.training_pars)

        # check which sampling mode is prescribed in the config file
        if "sow_target" in self.training_pars:
            sampling_callback = self.sample_batch_SOW
            size = self.training_pars["sow_target"]
            print("using SOW based batch sampling")
        elif "batchsize" in self.training_pars:
            sampling_callback = self.sample_batch
            size = self.training_pars["batchsize"]
            print("using fixed-size batch sampling")

        # pre-train the adversary
        print("pretraining adversarial network for {} batches".format(self.training_pars["adversary_pretrain_batches"]))
        for batch in range(int(self.training_pars["pretrain_batches"])):
            # sample coherently from (data, nuisance, label) tuples
            (data_batch_2j, nuisances_batch_2j, labels_batch_2j, auxdata_batch_2j), weights_batch_2j = sampling_callback([data_sig_2j, nuisances_sig_2j, labels_sig_2j, auxdat_sig_2j], weights_sig_2j, 
                                                                                                                         [data_bkg_2j, nuisances_bkg_2j, labels_bkg_2j, auxdat_bkg_2j], weights_bkg_2j,
                                                                                                                         size = size // 2, sig_sampling_pars = sig_sampling_pars, bkg_sampling_pars = bkg_sampling_pars)
            (data_batch_3j, nuisances_batch_3j, labels_batch_3j, auxdata_batch_3j), weights_batch_3j = sampling_callback([data_sig_3j, nuisances_sig_3j, labels_sig_3j, auxdat_sig_3j], weights_sig_3j, 
                                                                                                                         [data_bkg_3j, nuisances_bkg_3j, labels_bkg_3j, auxdat_bkg_3j], weights_bkg_3j,
                                                                                                                         size = size // 2, sig_sampling_pars = sig_sampling_pars, bkg_sampling_pars = bkg_sampling_pars)            
            data_batch = np.concatenate([data_batch_2j, data_batch_3j])
            nuisances_batch = np.concatenate([nuisances_batch_2j, nuisances_batch_3j])
            labels_batch = np.concatenate([labels_batch_2j, labels_batch_3j])
            weights_batch = np.concatenate([weights_batch_2j, weights_batch_3j])
            auxdata_batch = np.concatenate([auxdata_batch_2j, auxdata_batch_3j])

            env.train_adversary(data_step = data_batch, nuisances_step = nuisances_batch, labels_step = labels_batch, weights_step = weights_batch, batchnum = batch, auxdat_step = auxdata_batch)
            env.dump_loss_information(data = data_batch, nuisances = nuisances_batch, labels = labels_batch, weights = weights_batch, auxdat_step = auxdata_batch)
        print("adversary pretraining complete!")

        # pre-train the classifier
        print("pretraining classifier for {} batches".format(self.training_pars["classifier_pretrain_batches"]))
        for batch in range(int(self.training_pars["classifier_pretrain_batches"])):
            # sample coherently from (data, nuisance, label) tuples
            (data_batch_2j, nuisances_batch_2j, labels_batch_2j, auxdata_batch_2j), weights_batch_2j = sampling_callback([data_sig_2j, nuisances_sig_2j, labels_sig_2j, auxdat_sig_2j], weights_sig_2j, 
                                                                                                                         [data_bkg_2j, nuisances_bkg_2j, labels_bkg_2j, auxdat_bkg_2j], weights_bkg_2j, 
                                                                                                                         size = size // 2, sig_sampling_pars = sig_sampling_pars, bkg_sampling_pars = bkg_sampling_pars)
            (data_batch_3j, nuisances_batch_3j, labels_batch_3j, auxdata_batch_3j), weights_batch_3j = sampling_callback([data_sig_3j, nuisances_sig_3j, labels_sig_3j, auxdat_sig_3j], weights_sig_3j, 
                                                                                                                         [data_bkg_3j, nuisances_bkg_3j, labels_bkg_3j, auxdat_bkg_3j], weights_bkg_3j, 
                                                                                                                         size = size // 2, sig_sampling_pars = sig_sampling_pars, bkg_sampling_pars = bkg_sampling_pars)
            
            data_batch = np.concatenate([data_batch_2j, data_batch_3j])
            nuisances_batch = np.concatenate([nuisances_batch_2j, nuisances_batch_3j])
            labels_batch = np.concatenate([labels_batch_2j, labels_batch_3j])
            weights_batch = np.concatenate([weights_batch_2j, weights_batch_3j])
            auxdata_batch = np.concatenate([auxdata_batch_2j, auxdata_batch_3j])

            env.train_classifier(data_step = data_batch, labels_step = labels_batch, weights_step = weights_batch, batchnum = batch, auxdat_step = auxdata_batch)
            env.dump_loss_information(data = data_batch, nuisances = nuisances_batch, labels = labels_batch, weights = weights_batch, auxdat_step = auxdata_batch)            
        print("classifier pretraining complete!")

        # start the actual adversarial training
        print("starting adversarial training:")
        for batch in range(int(number_batches)):
            # sample coherently from (data, nuisance, label) tuples
            (data_batch_2j, nuisances_batch_2j, labels_batch_2j, auxdata_batch_2j), weights_batch_2j = sampling_callback([data_sig_2j, nuisances_sig_2j, labels_sig_2j, auxdat_sig_2j], weights_sig_2j, 
                                                                                                                         [data_bkg_2j, nuisances_bkg_2j, labels_bkg_2j, auxdat_bkg_2j], weights_bkg_2j, 
                                                                                                                         size = size // 2, sig_sampling_pars = sig_sampling_pars, bkg_sampling_pars = bkg_sampling_pars)
            (data_batch_3j, nuisances_batch_3j, labels_batch_3j, auxdata_batch_3j), weights_batch_3j = sampling_callback([data_sig_3j, nuisances_sig_3j, labels_sig_3j, auxdat_sig_3j], weights_sig_3j, 
                                                                                                                         [data_bkg_3j, nuisances_bkg_3j, labels_bkg_3j, auxdat_bkg_3j], weights_bkg_3j, 
                                                                                                                         size = size // 2, sig_sampling_pars = sig_sampling_pars, bkg_sampling_pars = bkg_sampling_pars)

            data_batch = np.concatenate([data_batch_2j, data_batch_3j])
            nuisances_batch = np.concatenate([nuisances_batch_2j, nuisances_batch_3j])
            labels_batch = np.concatenate([labels_batch_2j, labels_batch_3j])
            weights_batch = np.concatenate([weights_batch_2j, weights_batch_3j])
            auxdata_batch = np.concatenate([auxdata_batch_2j, auxdata_batch_3j])
            
            env.train_adversary(data_step = data_batch, nuisances_step = nuisances_batch, labels_step = labels_batch, weights_step = weights_batch, batchnum = batch, auxdat_step = auxdata_batch)
            env.train_step(data_step = data_batch, nuisances_step = nuisances_batch, labels_step = labels_batch, weights_step = weights_batch, batchnum = batch, auxdat_step = auxdata_batch)
            env.dump_loss_information(data = data_batch, nuisances = nuisances_batch, labels = labels_batch, weights = weights_batch, auxdat_step = auxdata_batch)

            # callbacks to keep track of the parameter evolution during training
            stat_dict_cur = env.get_model_statistics(data = data_batch, nuisances = nuisances_batch, labels = labels_batch, weights = weights_batch, auxdat_step = auxdata_batch)
            stat_dict_cur["batch"] = batch
            
            for key, val in stat_dict_cur.items():
                if not key in self.statistics_dict:
                    self.statistics_dict[key] = []
                self.statistics_dict[key].append(val)

            # some status printouts
            if not batch % int(self.training_pars["printout_interval"]):
                print("batch {}:".format(batch))
                print("dynamic batch size = " + str(len(weights_batch)))
                print("SOW per batch = " + str(np.sum(weights_batch)))
                env.dump_loss_information(data = data_batch, nuisances = nuisances_batch, labels = labels_batch, weights = weights_batch, auxdat_step = auxdata_batch)
                print("stat_dict = " + str(stat_dict_cur))

    def save_training_statistics(self, filepath):
        with open(filepath, "wb") as outfile:
            pickle.dump(self.statistics_dict, outfile)
