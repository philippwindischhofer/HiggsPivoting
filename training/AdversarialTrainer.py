import pandas as pd
import numpy as np
import pickle

from training.Trainer import Trainer

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

    # overload the 'train' method here
    def train(self, env, number_batches, traindat_sig, traindat_bkg, nuisances_sig, nuisances_bkg, weights_sig, weights_bkg):
        data_sig = traindat_sig
        data_bkg = traindat_bkg

        # prepare the labels for each signal / background component
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
