import numpy as np
import training.BatchSamplers as BatchSamplers

class AdversarialModelTrainer:

    def __init__(self, model, batch_sampler, training_pars):
        self.model = model
        self.data_formatter = self.model.data_formatter
        self.batch_sampler = batch_sampler
        self.training_pars = training_pars

        self.validation_check_interval = 100
        self.validation_check_batchsize = 10000

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

                # evaluate the total training loss
                train_loss = self.model.evaluate_loss(data_batch, nuis_batch, labels_batch, weights_batch)

                # compute the average loss on the validation dataset to check when to stop training
                validation_sampled_sig, validation_weights_sig = self.batch_sampler(valsamples_sig_formatted, size = self.validation_check_batchsize // 2)
                validation_sampled_bkg, validation_weights_bkg = self.batch_sampler(valsamples_bkg_formatted, size = self.validation_check_batchsize // 2)
                (data_validation_batch, nuis_validation_batch, labels_validation_batch), validation_weights_batch = self._combine_samples(validation_sampled_sig, validation_weights_sig, validation_sampled_bkg, validation_weights_bkg)
                validation_weights_batch = np.abs(validation_weights_batch)
                validation_loss = self.model.evaluate_loss(data_validation_batch, nuis_validation_batch, labels_validation_batch, validation_weights_batch)

                print("batch {}: train_loss = {}  val_loss = {}".format(batch, train_loss, validation_loss))

                if validation_loss < best_val_loss:
                    print("have new best validation loss; triggering checkpoint saver")
                    best_val_loss = validation_loss
                
                    self.model.save(self.model.path)

    def _combine_samples(self, samples_sig, weights_sig, samples_bkg, weights_bkg):

        data_combined = [np.concatenate([cur_sig_sampled, cur_bkg_sampled], axis = 0) for cur_sig_sampled, cur_bkg_sampled in zip(samples_sig, samples_bkg)]
        weights_combined = np.concatenate([weights_sig, weights_bkg], axis = 0)
        
        return data_combined, weights_combined
