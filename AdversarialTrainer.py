import pandas as pd
import numpy as np

from Trainer import Trainer

class AdversarialTrainer(Trainer):

    def __init__(self, training_pars):
        super(AdversarialTrainer, self).__init__(training_pars)

    # overload the 'train' method here
    def train(self, env, number_batches, df_sig, df_bkg, nuisances):
        # this assumes that the nuisances are indeed part of the training data,
        # and furthermore that the 'nuisances' argument is a list of the corresponding
        # column names
        nuisances_sig = df_sig[nuisances].values
        nuisances_bkg = df_bkg[nuisances].values

        data_sig = df_sig.values                
        data_bkg = df_bkg.values

        labels_sig = np.ones(len(data_sig))
        labels_bkg = np.zeros(len(data_bkg))

        # also prepare arrays with the full training dataset
        data_train = np.concatenate([data_sig, data_bkg], axis = 0)
        nuisances_train = np.concatenate([nuisances_sig, nuisances_bkg], axis = 0)
        labels_train = np.concatenate([labels_sig, labels_bkg], axis = 0)

        # initialize the environment
        env.init(data_train = data_train, data_nuisance = nuisances_train)

        # pre-train the adversary
        print("pretraining MINE network for {} batches".format(self.training_pars["pretrain_batches"]))
        for batch in range(self.training_pars["pretrain_batches"]):
            env.train_adversary(data_step = data_train, nuisances_step = nuisances_train, labels_step = labels_train)
            env.dump_loss_information(data = data_train, nuisances = nuisances_train, labels = labels_train)
        print("pretraining complete!")

        print("starting training:")
        for batch in range(number_batches):
            # sample separately from signal and background samples, try to have a balanced sig/bkg ratio in every batch
            inds_sig = np.random.choice(len(data_sig), int(self.training_pars["batch_size"] / 2))
            inds_bkg = np.random.choice(len(data_bkg), int(self.training_pars["batch_size"] / 2))

            data_batch_sig = data_sig[inds_sig]
            data_batch_bkg = data_bkg[inds_bkg]

            nuisances_batch_sig = nuisances_sig[inds_sig]
            nuisances_batch_bkg = nuisances_bkg[inds_bkg]

            labels_batch_sig = labels_sig[inds_sig]
            labels_batch_bkg = labels_bkg[inds_bkg]

            data_batch = np.concatenate([data_batch_sig, data_batch_bkg], axis = 0)
            nuisances_batch = np.concatenate([nuisances_batch_sig, nuisances_batch_bkg], axis = 0)
            labels_batch = np.concatenate([labels_batch_sig, labels_batch_bkg], axis = 0)

            env.train_adversary(data_step = data_train, nuisances_step = nuisances_train, labels_step = labels_train)
            env.train_step(data_step = data_batch, nuisances_step = nuisances_batch, labels_step = labels_batch)

            if not batch % self.training_pars["printout_interval"]:
                print("batch {}:".format(batch))
                env.dump_loss_information(data = data_train, nuisances = nuisances_train, labels = labels_train)


