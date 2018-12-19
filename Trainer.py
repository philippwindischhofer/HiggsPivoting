import numpy as np

class Trainer:

    def __init__(self, training_pars):
        self.training_pars = training_pars

        # give default values
        self.training_pars.setdefault("printout_interval", 1000)
        self.training_pars.setdefault("batch_size", 256)

    def train(self, env, number_batches, data_sig, data_bkg):
        # initialize the environment
        env.init()

        for batch in range(number_batches):
            # sample separately from signal and background samples, try to have a balanced sig/bkg ratio in every batch
            inds_sig = np.random.choice(len(data_sig), int(self.training_pars["batch_size"] / 2))
            inds_bkg = np.random.choice(len(data_bkg), int(self.training_pars["batch_size"] / 2))

            data_batch_sig = data_sig[inds_sig]
            data_batch_bkg = data_bkg[inds_bkg]

            data_batch = np.concatenate([data_batch_sig, data_batch_bkg], axis = 0)
            labels_batch = np.concatenate([np.ones(len(data_batch_sig)), np.zeros(len(data_batch_bkg))], axis = 0)
            
            env.train_step(data_step = data_batch, labels_step = labels_batch)

            if not batch % self.training_pars["printout_interval"]:
                loss = env.loss(data = data_batch, labels = labels_batch)
                print("batch {}: loss = {}".format(batch, loss))

