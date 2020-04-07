import os, pickle
import numpy as np
import training.BatchSamplers as BatchSamplers

class AdversarialModelTrainer:

    def __init__(self, model, batch_sampler, training_pars):
        self.model = model
        self.data_formatter = self.model.data_formatter
        self.batch_sampler = batch_sampler
        self.training_pars = training_pars

        if not isinstance(self.training_pars, dict):
            self.training_pars = {key: float(val) for key, val in self.training_pars.items()}

        self.validation_check_interval = 500
        self.validation_check_batchsize = 20000

        self.statistics_dict = {} # to hold the model statistics

    def train(self, trainsamples_sig, trainsamples_bkg, valsamples_sig, valsamples_bkg):

        best_val_loss = 1e6
        
        # first, format the training and validation datasets in the way required by the model
        trainsamples_sig_formatted = [self.data_formatter.format_as_TrainingSample(cur_sample, is_signal = True) for cur_sample in trainsamples_sig]
        trainsamples_bkg_formatted = [self.data_formatter.format_as_TrainingSample(cur_sample, is_signal = False) for cur_sample in trainsamples_bkg]

        valsamples_sig_formatted = [self.data_formatter.format_as_TrainingSample(cur_sample, is_signal = True) for cur_sample in valsamples_sig]
        valsamples_bkg_formatted = [self.data_formatter.format_as_TrainingSample(cur_sample, is_signal = False) for cur_sample in valsamples_bkg]

        (data_all, nuis_all, labels_all), weights_all = BatchSamplers.all(trainsamples_sig_formatted + trainsamples_bkg_formatted)
        
        self.model.init(data_all, nuis_all)

        batchsize = self.training_pars["batchsize"]

        # pre-train the classifier
        clf_pretrain_batches = int(self.training_pars["classifier_pretrain_batches"])
        print("pretraining the classifier for {} batches".format(clf_pretrain_batches))
        for batch in range(clf_pretrain_batches):
            sampled_sig, weights_sig = self.batch_sampler(trainsamples_sig_formatted, size = batchsize // 2)
            sampled_bkg, weights_bkg = self.batch_sampler(trainsamples_bkg_formatted, size = batchsize // 2)
            (data_batch, nuis_batch, labels_batch), weights_batch = self._combine_samples(sampled_sig, weights_sig, sampled_bkg, weights_bkg)
            weights_batch = np.abs(weights_batch)  # train on absolute weights

            self.model.train_classifier(data_batch, labels_batch, weights_batch, batch)

        # pre-train the adversary
        adv_pretrain_batches = int(self.training_pars["adversary_pretrain_batches"])
        print("pretraining the adversarial network for {} batches".format(adv_pretrain_batches))
        for batch in range(adv_pretrain_batches):
            (data_batch_bkg, nuis_batch_bkg, labels_batch_bkg), weights_bkg = self.batch_sampler(trainsamples_bkg_formatted, size = batchsize)
            weights_bkg = np.abs(weights_bkg)

            self.model.train_adversary(data_batch_bkg, nuis_batch_bkg, labels_batch_bkg, weights_bkg, batch)

            if batch % self.validation_check_interval == 0:
                
                adv_loss = self.model.evaluate_adversary_loss(data_batch_bkg, nuis_batch_bkg, labels_batch_bkg, weights_bkg)
                print("batch {}: adv_loss = {}".format(batch, adv_loss))

        # start the actual adversarial training
        adv_training_batches = int(self.training_pars["training_batches"])
        print("performing adversarial training for {} batches".format(adv_training_batches))
        for batch in range(adv_training_batches):
            
            # update adversary
            for adv_update in range(5):
                (data_batch_bkg, nuis_batch_bkg, labels_batch_bkg), weights_bkg = self.batch_sampler(trainsamples_bkg_formatted, size = batchsize)
                weights_bkg = np.abs(weights_bkg)
                self.model.train_adversary(data_batch_bkg, nuis_batch_bkg, labels_batch_bkg, weights_bkg, batch)

            # update classifier
            sampled_sig, weights_sig = self.batch_sampler(trainsamples_sig_formatted, size = batchsize // 2)
            sampled_bkg, weights_bkg = self.batch_sampler(trainsamples_bkg_formatted, size = batchsize // 2)
            (data_batch, nuis_batch, labels_batch), weights_batch = self._combine_samples(sampled_sig, weights_sig, sampled_bkg, weights_bkg)
            weights_batch = np.abs(weights_batch)

            self.model.train_step(data_batch, nuis_batch, labels_batch, weights_batch, batch)

            if batch % self.validation_check_interval == 0:

                # gather some statistics on the evolution of the losses on the training dataset
                cur_statdict = self.model.get_model_statistics(data_batch, nuis_batch, labels_batch, weights_batch, postfix = "_train")
                cur_statdict["batch"] = batch

                # evaluate the total training loss
                train_loss = self.model.evaluate_loss(data_batch, nuis_batch, labels_batch, weights_batch)

                # compute the average loss on the validation dataset to check when to stop training
                validation_sampled_sig, validation_weights_sig = self.batch_sampler(valsamples_sig_formatted, size = self.validation_check_batchsize // 2)
                validation_sampled_bkg, validation_weights_bkg = self.batch_sampler(valsamples_bkg_formatted, size = self.validation_check_batchsize // 2)
                (data_validation_batch, nuis_validation_batch, labels_validation_batch), validation_weights_batch = self._combine_samples(validation_sampled_sig, validation_weights_sig, validation_sampled_bkg, validation_weights_bkg)
                validation_weights_batch = np.abs(validation_weights_batch)
                
                cur_statdict_validation = self.model.get_model_statistics(data_validation_batch, nuis_validation_batch, labels_validation_batch, validation_weights_batch, postfix = "_validation")
                cur_statdict.update(cur_statdict_validation)

                stat_dict_text = ["{} = {:.6g}".format(key, val) for key, val in cur_statdict.items() if key is not "batch"]
                print("batch {}: {}".format(batch, " | ".join(stat_dict_text)))

                self._append_to_statistics_dict(cur_statdict) # register it in the central location

                validation_loss = cur_statdict_validation["total_loss_validation"]
                if validation_loss < best_val_loss:
                    print("have new best validation loss; triggering checkpoint saver")
                    best_val_loss = validation_loss
                
                    self.model.save(self.model.path)

        # save training statistics at the end
        model_stat_path = os.path.join(self.model.path, "training_evolution.pkl")
        print("saving statistics dict to {}".format(model_stat_path))
        with open(model_stat_path, "wb") as stat_outfile:
            pickle.dump(self.statistics_dict, stat_outfile)

    def _combine_samples(self, samples_sig, weights_sig, samples_bkg, weights_bkg):

        data_combined = [np.concatenate([cur_sig_sampled, cur_bkg_sampled], axis = 0) for cur_sig_sampled, cur_bkg_sampled in zip(samples_sig, samples_bkg)]
        weights_combined = np.concatenate([weights_sig, weights_bkg], axis = 0)
        
        return data_combined, weights_combined

    def _append_to_statistics_dict(self, to_append):
        for key, val in to_append.items():
            if not key in self.statistics_dict:
                self.statistics_dict[key] = []
            self.statistics_dict[key].append(val)
