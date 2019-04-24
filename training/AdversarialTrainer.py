import pandas as pd
import numpy as np
import pickle

from training.Trainer import Trainer

class AdversarialTrainer(Trainer):

    def __init__(self, training_pars):
        super(AdversarialTrainer, self).__init__(training_pars)
        self.statistics_dict = {}

    # coherently draw samples from 'sources' (a list of datasets, e.g. the training features, the nuisances and the weights)
    # such that the final sample comes as close as possible to 'sow_target'
    def sample_from_by_sow(self, sources, weights, initial_req = 100, sow_target = 1.0, target_tol = 0.1, batch_limit = 10000):
        inds = np.random.choice(len(weights), initial_req)
        
        while True:
            sampled_weights = weights[inds]
            sow = np.sum(sampled_weights)

            # check if have already reached the SOW target
            if sow > sow_target - target_tol or len(inds_sig) > batch_limit:
                break

            # estimate how many more samples should be requested
            sample_request = int(min(0.1 * max(sow_target - sow, 0.0) / abs(sow) * len(inds), batch_limit / 4))

            # get new samples and append them to the old ones
            inds_sampled = np.random.choice(len(weights), sample_request)
            inds = np.concatenate([inds, inds_sampled], axis = 0)

        sampled_weights = weights[inds]
        sampled_data = [cur_source[inds] for cur_source in sources]

        return sampled_data, sampled_weights

    # sample a fixed number of events from 'sources'
    def sample_from(self, sources, weights, req = 100):
        inds = np.random.choice(len(weights), req)

        sampled_weights = weights[inds]
        sampled_data = [cur_source[inds] for cur_source in sources]

        return sampled_data, sampled_weights        

    # draw samples from 'signal_components' and 'background_components' such that both are equally represented
    # in the resulting sample (i.e. appear with equal SOW of 'sow_target')
    def sample_from_components(self, components, weights, batch_size = 1000):        
        # Note: this effectively transposes the nested lists such that the iterations become easier
        sources = list(map(list, zip(*components)))        

        # first, compute the number of events available for each signal / background component ...
        nevents = [len(cur) for cur in weights]
        total_nevents = np.sum(nevents)

        # ... and also the SOW for each
        SOWs = [np.sum(cur) for cur in weights]
        total_SOW = np.sum(SOWs)
        SOWs /= total_SOW # normalize total SOW to 1

        # now, compute the number of events that should be sampled from each signal / background component:
        # sample them in the same proportions with which they appear in the training dataset ...
        sampled_data = []
        sampled_weights = []
        for cur_source, cur_weights, cur_nevents in zip(sources, weights, nevents):
            cur_sampled_data, cur_sampled_weights = self.sample_from(cur_source, cur_weights, req = int(batch_size * cur_nevents / total_nevents))
            sampled_data.append(cur_sampled_data)
            sampled_weights.append(cur_sampled_weights)

        # ... and normalize them such that their SOWs are in the correct relation to each other
        for cur, cur_SOW in enumerate(SOWs):
            sampled_weights[cur] *= cur_SOW / np.sum(sampled_weights[cur]) # each batch will have a SOW of 1

        # transpose them back for easy concatenation
        sampled_sources = list(map(list, zip(*sampled_data)))

        # perform the concatenation ...
        sampled = [np.concatenate(cur, axis = 0) for cur in sampled_sources]
        sampled_weights = np.concatenate(sampled_weights, axis = 0)

        # ... and return
        return sampled, sampled_weights

    def sample_batch(self, sources_sig, weights_sig, sources_bkg, weights_bkg, batch_size = 1000):
        # sample a halfbatch size full of events from signal, and from background, individually normalized to unit SOW
        # and keeping their relative proportions fixed
        sig_sampled, sig_weights = self.sample_from_components(sources_sig, weights_sig, batch_size = batch_size // 2)
        print("sampled {} events from signal components with total SOW = {}".format(len(sig_weights), np.sum(sig_weights)))

        bkg_sampled, bkg_weights = self.sample_from_components(sources_bkg, weights_bkg, batch_size = batch_size // 2)
        print("sampled {} events from background components with total SOW = {}".format(len(bkg_weights), np.sum(bkg_weights)))

        # concatenate the individual components
        data_combined = [np.concatenate([cur_sig_sampled, cur_bkg_sampled], axis = 0) for cur_sig_sampled, cur_bkg_sampled in zip(sig_sampled, bkg_sampled)]
        weights_combined = np.concatenate([sig_weights, bkg_weights], axis = 0)

        return data_combined, weights_combined

    # def sample_from(self, sources_sig, weights_sig, sources_bkg, weights_bkg, initial_req, sow_target = 1.0, target_tol = 0.1, batch_limit = 10000):
    #     # Note: the length of the individual consituents of sources_sig and sources_bkg must have the
    #     # same length! (will usually be the case since they correspond to the same events anyways)

    #     # need to sample from signal and background in such a way that the sum of weights
    #     # of either source is very similar (or ideally, identical)
    #     inds_sig = np.random.choice(len(weights_sig), initial_req)
    #     inds_bkg = np.random.choice(len(weights_bkg), initial_req)

    #     # resample as long as the sums-of-weights of signal and background events are equal
    #     # to each other within some tolerance
    #     while True:
    #         sampled_weights_sig = weights_sig[inds_sig]
    #         sampled_weights_bkg = weights_bkg[inds_bkg]

    #         sow_sig = np.sum(sampled_weights_sig)
    #         sow_bkg = np.sum(sampled_weights_bkg)

    #         # have reached the SOW target for both signal and background, stop
    #         if (sow_sig > sow_target - target_tol and sow_bkg > sow_target - target_tol) or (len(inds_sig) + len(inds_bkg) > batch_limit):
    #             break

    #         # get a good guess for how many more samples will be needed separately for signal and background
    #         sample_request_bkg = int(min(0.1 * max(sow_target - sow_bkg, 0.0) / abs(sow_bkg) * len(inds_bkg), batch_limit / 4))
    #         sample_request_sig = int(min(0.1 * max(sow_target - sow_sig, 0.0) / abs(sow_sig) * len(inds_sig), batch_limit / 4))
            
    #         # get the new samples and append them
    #         inds_sampled_bkg = np.random.choice(len(weights_bkg), sample_request_bkg)
    #         inds_sampled_sig = np.random.choice(len(weights_sig), sample_request_sig)

    #         inds_sig = np.concatenate([inds_sig, inds_sampled_sig], axis = 0)
    #         inds_bkg = np.concatenate([inds_bkg, inds_sampled_bkg], axis = 0)

    #     sampled_weights_sig = weights_sig[inds_sig]
    #     sampled_weights_bkg = weights_bkg[inds_bkg]

    #     sampled_sig = [cur_source[inds_sig] for cur_source in sources_sig]
    #     sampled_bkg = [cur_source[inds_bkg] for cur_source in sources_bkg]

    #     sampled = [np.concatenate([sample_sig, sample_bkg], axis = 0) for sample_sig, sample_bkg in zip(sampled_sig, sampled_bkg)]
    #     sampled_weights = np.concatenate([sampled_weights_sig, sampled_weights_bkg], axis = 0)

    #     return sampled, sampled_weights

    # overload the 'train' method here
    def train(self, env, number_batches, traindat_sig, traindat_bkg, nuisances_sig, nuisances_bkg, weights_sig, weights_bkg):
        data_sig = traindat_sig
        data_bkg = traindat_bkg

        # first, compute the SOW of the individual signal / background components as well as the total SOW for the combined
        # signal and the combined background
        print("got training dataset with {} signal components".format(len(data_sig)))
        print("got training dataset with {} background components".format(len(data_bkg)))

        sig_SOWs = [np.sum(cur_weights) for cur_weights in weights_sig]
        bkg_SOWs = [np.sum(cur_weights) for cur_weights in weights_bkg]
        total_sig_SOW = np.sum(sig_SOWs)
        total_bkg_SOW = np.sum(bkg_SOWs)

        print("signal SOWs = {}".format(sig_SOWs))
        print("total signal SOW = {}".format(total_sig_SOW))
        print("background SOWs = {}".format(bkg_SOWs))
        print("total background SOW = {}".format(total_bkg_SOW))

        labels_sig = [np.ones(len(cur_data_sig)) for cur_data_sig in data_sig]
        labels_bkg = [np.zeros(len(cur_data_bkg)) for cur_data_bkg in data_bkg]

        # also prepare arrays with the full training dataset
        comb_data_sig = np.concatenate(data_sig, axis = 0)
        comb_data_bkg = np.concatenate(data_sig, axis = 0)
        comb_data_train = np.concatenate([comb_data_sig, comb_data_bkg], axis = 0)
        comb_nuisances_sig = np.concatenate(nuisances_sig, axis = 0)
        comb_nuisances_bkg = np.concatenate(nuisances_bkg, axis = 0)
        nuisances_train = np.concatenate([comb_nuisances_sig, comb_nuisances_bkg], axis = 0)

        # initialize the environment
        env.init(data_train = comb_data_train, data_nuisance = nuisances_train)

        # pre-train the adversary
        print("pretraining adversarial network for {} batches".format(self.training_pars["adversary_pretrain_batches"]))
        for batch in range(int(self.training_pars["pretrain_batches"])):
            # sample coherently from (data, nuisance, label) tuples
            (data_batch, nuisances_batch, labels_batch), weights_batch = self.sample_batch([data_sig, nuisances_sig, labels_sig], weights_sig, [data_bkg, nuisances_bkg, labels_bkg], weights_bkg, 
                                                                                           batch_size = self.training_pars["batchsize"])

            env.train_adversary(data_step = data_batch, nuisances_step = nuisances_batch, labels_step = labels_batch, weights_step = weights_batch, batchnum = batch)
            env.dump_loss_information(data = data_batch, nuisances = nuisances_batch, labels = labels_batch, weights = weights_batch)
        print("adversary pretraining complete!")

        # pre-train the classifier
        print("pretraining classifier for {} batches".format(self.training_pars["classifier_pretrain_batches"]))
        for batch in range(int(self.training_pars["classifier_pretrain_batches"])):
            # sample coherently from (data, nuisance, label) tuples
            (data_batch, nuisances_batch, labels_batch), weights_batch = self.sample_batch([data_sig, nuisances_sig, labels_sig], weights_sig, [data_bkg, nuisances_bkg, labels_bkg], weights_bkg, 
                                                                                           batch_size = self.training_pars["batchsize"])

            env.train_classifier(data_step = data_batch, labels_step = labels_batch, weights_step = weights_batch, batchnum = batch)
            env.dump_loss_information(data = data_batch, nuisances = nuisances_batch, labels = labels_batch, weights = weights_batch)            
        print("classifier pretraining complete!")

        # start the actual adversarial training
        print("starting adversarial training:")
        for batch in range(int(number_batches)):
            # sample coherently from (data, nuisance, label) tuples
            (data_batch, nuisances_batch, labels_batch), weights_batch = self.sample_batch([data_sig, nuisances_sig, labels_sig], weights_sig, [data_bkg, nuisances_bkg, labels_bkg], weights_bkg, 
                                                                                           batch_size = self.training_pars["batchsize"])

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
            if not batch % int(self.training_pars["printout_interval"]):
                print("batch {}:".format(batch))
                print("dynamic batch size = " + str(len(data_batch)))
                env.dump_loss_information(data = data_batch, nuisances = nuisances_batch, labels = labels_batch, weights = weights_batch)
                print("stat_dict = " + str(stat_dict_cur))

    def save_training_statistics(self, filepath):
        with open(filepath, "wb") as outfile:
            pickle.dump(self.statistics_dict, outfile)
