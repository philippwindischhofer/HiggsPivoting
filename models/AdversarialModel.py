import os
import numpy as np
from configparser import ConfigParser

import tensorflow as tf

from base.PCAWhiteningPreprocessor import PCAWhiteningPreprocessor

# possible concrete models that are supported
from models.GMMAdversary import GMMAdversary
from models.MINEAdversary import MINEAdversary
from models.DisCoAdversary import DisCoAdversary
from models.SimpleClassifier import SimpleClassifier
from models.SimpleProbabilisticClassifier import SimpleProbabilisticClassifier

# available data formatters that are supported
from training.DataFormatters import only_2j, only_3j

from base.Configs import TrainingConfig

class AdversarialModel:

    def __init__(self, name, classifier_model, adversary_model, global_pars, path = None, training_config = {}):
        self.classifier_model = classifier_model
        self.adversary_model = adversary_model
        self.training_config = training_config

        self.pre = None
        self.pre_nuisance = None

        self.global_pars = global_pars
        self.data_formatter = eval(global_pars["data_formatter"])()

        self.lambda_final = float(self.global_pars["lambda"])
        self.graph = tf.Graph()

        self.path = path
        self.name = name
        self.sess = self.sess = tf.Session(graph = self.graph, 
                                           config = TrainingConfig.session_config)

        self.private_DisCo_adversary = DisCoAdversary("private_DisCo", hyperpars = {})

    @staticmethod
    def extract_config(model_name, config):
        gconfig = ConfigParser()

        main_config = config[model_name]
        gconfig["AdversarialModel"] = main_config

        classifier_model_name = main_config["classifier_model"]
        adversary_model_name = main_config["adversary_model"]
        training_config_name = main_config["training_config"]

        gconfig[classifier_model_name] = config[classifier_model_name]
        gconfig[adversary_model_name] = config[adversary_model_name]
        gconfig[training_config_name] = config[training_config_name]

        return gconfig

    @classmethod
    def from_config(cls, config_dir):
        gconfig = ConfigParser()
        gconfig.read(os.path.join(config_dir, "meta.conf"))

        global_pars = gconfig["AdversarialModel"]

        model_name = global_pars["model_name"]
        classifier_model_name = global_pars["classifier_model"]
        adversary_model_name = global_pars["adversary_model"]

        classifier_model_type = gconfig[classifier_model_name]["model_type"]
        adversary_model_type = gconfig[adversary_model_name]["model_type"]

        classifier_model = eval(classifier_model_type)
        adversary_model = eval(adversary_model_type)

        classifier_hyperpars = gconfig[global_pars["classifier_model"]]
        adversary_hyperpars = gconfig[global_pars["adversary_model"]]
        training_config = gconfig[global_pars["training_config"]]

        mod = classifier_model(classifier_model_name, hyperpars = classifier_hyperpars)
        adv = adversary_model(adversary_model_name, hyperpars = adversary_hyperpars)
        obj = cls(model_name, mod, adv, global_pars, path = config_dir, training_config = training_config)

        # then re-build the graph using these settings
        obj.build()

        # finally, load back the weights and preprocessors, if available
        obj.load(config_dir)
        return obj

    def build(self):
        
        num_inputs = int(float(self.global_pars["num_inputs"]))
        num_nuisances = int(float(self.global_pars["num_nuisances"]))

        with self.graph.as_default():
            self.pre = PCAWhiteningPreprocessor(num_inputs)
            self.pre_nuisance = PCAWhiteningPreprocessor(num_nuisances)
            
            # prepare the inputs
            self.labels_in = tf.placeholder(tf.int32, [None, ], name = 'labels_in')
            self.data_in = tf.placeholder(tf.float32, [None, num_inputs], name = 'data_in')
            self.nuisances_in = tf.placeholder(tf.float32, [None, num_nuisances], name = 'nuisances_in')
            self.weights_in = tf.placeholder(tf.float32, [None, ], name = 'weights_in')
            self.is_training = tf.placeholder(tf.bool, name = 'is_training')
            self.lambdaval = tf.placeholder(tf.float32, [1], name = 'lambdaval')
            self.lambdaval_private_DisCo = tf.placeholder(tf.float32, [1], name = 'lambdaval_private_DisCo')

            self.classifier_lr = tf.placeholder(tf.float32, [], name = "classifier_lr")
            self.adversary_lr = tf.placeholder(tf.float32, [], name = "adversary_lr")
                        
            self.labels_one_hot = tf.one_hot(self.labels_in, depth = 2)
            self.weights_bkg = tf.where(tf.math.equal(self.labels_in, 0), self.weights_in, tf.zeros_like(self.weights_in))

            # set up the classifier
            self.classifier_out, self.classifier_vars = self.classifier_model.build_model(self.data_in, is_training = self.is_training)
            self.classification_loss = self.classifier_model.build_loss(self.classifier_out, self.labels_one_hot, weights = self.weights_in)

            self.classifier_out_single = tf.expand_dims(self.classifier_out[:,0], axis = 1)

            # set up the adversary
            self.adv_loss, self.adversary_vars = self.adversary_model.build_loss(self.classifier_out_single, self.nuisances_in, weights = self.weights_bkg, is_training = self.is_training)

            self.private_DisCo_adv_loss, _ = self.private_DisCo_adversary.build_loss(self.classifier_out_single, self.nuisances_in, weights = self.weights_bkg, is_training = self.is_training)

            # total loss
            self.total_loss = self.classification_loss + self.lambdaval * (-self.adv_loss)

            self.private_DisCo_total_loss = self.classification_loss + self.lambdaval_private_DisCo * (-self.private_DisCo_adv_loss)

            # set up the optimisers
            self.train_classifier_standalone = tf.train.AdamOptimizer(learning_rate = self.classifier_lr, 
                                                                      beta1 = float(self.global_pars["adam_clf_beta1"]), 
                                                                      beta2 = float(self.global_pars["adam_clf_beta2"]), 
                                                                      epsilon = float(self.global_pars["adam_clf_eps"])).minimize(self.classification_loss, var_list = self.classifier_vars)

            self.train_adversary_standalone = tf.train.AdamOptimizer(learning_rate = self.adversary_lr,
                                                                     beta1 = float(self.global_pars["adam_adv_beta1"]), 
                                                                     beta2 = float(self.global_pars["adam_adv_beta2"]), 
                                                                     epsilon = float(self.global_pars["adam_adv_eps"])).minimize(self.adv_loss, var_list = self.adversary_vars)

            self.train_classifier_adv = tf.train.AdamOptimizer(learning_rate = self.classifier_lr, 
                                                               beta1 = float(self.global_pars["adam_clf_adv_beta1"]), 
                                                               beta2 = float(self.global_pars["adam_clf_adv_beta2"]), 
                                                               epsilon = float(self.global_pars["adam_clf_adv_eps"])).minimize(self.total_loss, var_list = self.classifier_vars)

            self.saver = tf.train.Saver(var_list = self.classifier_vars + self.adversary_vars)

    def init(self, data_train, data_nuisance):
        self.pre.setup(data_train)
        self.pre_nuisance.setup(data_nuisance)

        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

    def _lr_scheduler(self, lr_start, lr_decay, batchnum):
        return lr_start * np.exp(-lr_decay * batchnum)

    def train_step(self, data_step, nuisances_step, labels_step, weights_step, batchnum):
        data_pre = self.pre.process(data_step)
        nuisances_pre = self.pre_nuisance.process(nuisances_step)
        weights_step = weights_step.flatten()

        classifier_lr = self._lr_scheduler(lr_start = float(self.global_pars["adam_clf_adv_lr"]),
                                           lr_decay = float(self.global_pars["adam_clf_adv_lr_decay"]),
                                           batchnum = batchnum)

        with self.graph.as_default():
            self.sess.run(self.train_classifier_adv, feed_dict = {self.data_in: data_pre, self.nuisances_in: nuisances_pre, self.labels_in: labels_step, self.weights_in: weights_step, self.lambdaval: [self.lambda_final], self.is_training: True, self.classifier_lr: classifier_lr})

    def train_classifier(self, data_step, labels_step, weights_step, batchnum):
        data_pre = self.pre.process(data_step)
        weights_step = weights_step.flatten()

        # determine the current learning rate as per the scheduling
        classifier_lr = self._lr_scheduler(lr_start = float(self.global_pars["adam_clf_lr"]),
                                          lr_decay = float(self.global_pars["adam_clf_lr_decay"]),
                                          batchnum = batchnum)

        with self.graph.as_default():
            self.sess.run(self.train_classifier_standalone, feed_dict = {self.data_in: data_pre, self.labels_in: labels_step, self.weights_in: weights_step, self.is_training: True, self.classifier_lr: classifier_lr})

    def train_adversary(self, data_step, nuisances_step, labels_step, weights_step, batchnum):
        data_pre = self.pre.process(data_step)
        nuisances_pre = self.pre_nuisance.process(nuisances_step)
        weights_step = weights_step.flatten()

        # determine the current learning rate as per the scheduling
        adversary_lr = self._lr_scheduler(lr_start = float(self.global_pars["adam_adv_lr"]),
                                          lr_decay = float(self.global_pars["adam_adv_lr_decay"]),
                                          batchnum = batchnum)

        with self.graph.as_default():
            self.sess.run(self.train_adversary_standalone, feed_dict = {self.data_in: data_pre, self.nuisances_in: nuisances_pre, self.labels_in: labels_step, self.weights_in: weights_step, self.is_training: True, self.adversary_lr: adversary_lr})

    def evaluate_classifier_loss(self, data, labels, weights_step):
        data_pre = self.pre.process(data)
        weights_step = weights_step.flatten()

        with self.graph.as_default():
            classifier_lossval = self.sess.run(self.classification_loss, feed_dict = {self.data_in: data_pre, self.labels_in: labels, self.weights_in: weights_step, self.is_training: True})

        return classifier_lossval

    def evaluate_adversary_loss(self, data, nuisances, labels, weights_step):
        data_pre = self.pre.process(data)
        nuisances_pre = self.pre_nuisance.process(nuisances)
        weights_step = weights_step.flatten()

        with self.graph.as_default():
            adv_loss = self.sess.run(self.adv_loss, feed_dict = {self.data_in: data_pre, self.nuisances_in: nuisances_pre, self.labels_in: labels, self.weights_in: weights_step, self.is_training: True})

        return adv_loss

    def evaluate_private_DisCo_adversary_loss(self, data, nuisances, labels, weights_step):
        data_pre = self.pre.process(data)
        nuisances_pre = self.pre_nuisance.process(nuisances)
        weights_step = weights_step.flatten()

        with self.graph.as_default():
            adv_loss = self.sess.run(self.private_DisCo_adv_loss, feed_dict = {self.data_in: data_pre, self.nuisances_in: nuisances_pre, self.labels_in: labels, self.weights_in: weights_step, self.is_training: True})

        return adv_loss

    def evaluate_loss(self, data, nuisances, labels, weights_step):
        data_pre = self.pre.process(data)
        nuisances_pre = self.pre_nuisance.process(nuisances)
        weights_step = weights_step.flatten()

        with self.graph.as_default():
            total_loss = self.sess.run(self.total_loss, feed_dict = {self.data_in: data_pre, self.nuisances_in: nuisances_pre, self.labels_in: labels, self.weights_in: weights_step, self.lambdaval: [self.lambda_final], self.is_training: True})

        return total_loss

    def evaluate_all_losses(self, data, nuisances, labels, weights_step, DisCo_lambda):
        data_pre = self.pre.process(data)
        nuisances_pre = self.pre_nuisance.process(nuisances)
        weights_step = weights_step.flatten()

        with self.graph.as_default():
            (clf_loss, adv_loss, total_loss, private_DisCo_adv_loss, private_DisCo_total_loss) = self.sess.run([self.classification_loss, self.adv_loss, self.total_loss, self.private_DisCo_adv_loss, self.private_DisCo_total_loss], 
                                                                                                               feed_dict = {self.data_in: data_pre, self.nuisances_in: nuisances_pre, self.labels_in: labels, self.weights_in: weights_step, self.lambdaval: [self.lambda_final], self.lambdaval_private_DisCo: [DisCo_lambda], self.is_training: True})

        return clf_loss, adv_loss, total_loss, private_DisCo_adv_loss, private_DisCo_total_loss

    def evaluate_private_DisCo_total_loss(self, data, nuisances, labels, weights_step, DisCo_lambda):
        data_pre = self.pre.process(data)
        nuisances_pre = self.pre_nuisance.process(nuisances)
        weights_step = weights_step.flatten()

        with self.graph.as_default():
            total_loss = self.sess.run(self.private_DisCo_total_loss, feed_dict = {self.data_in: data_pre, self.nuisances_in: nuisances_pre, self.labels_in: labels, self.weights_in: weights_step, self.lambdaval: [self.lambda_final], self.lambdaval_private_DisCo: [DisCo_lambda], self.is_training: True})

        return total_loss

    def predict(self, data, pred_size = 256):
        data_pre = self.pre.process(data)
        datlen = len(data_pre)

        print("datlen = {}".format(datlen))

        chunks = np.array_split(data_pre, max(datlen / pred_size, 1), axis = 0)
        retvals = []
        for chunk in chunks:
            retval_cur = self.sess.run(self.classifier_out, feed_dict = {self.data_in: chunk, self.is_training: False})
            retvals.append(retval_cur)

        return np.concatenate(retvals, axis = 0)

    def load(self, indir):
        
        # load the weights
        with self.graph.as_default():
            try:
                self.saver.restore(self.sess, os.path.join(indir, "model.dat"))
                print("weights successfully loaded from " + indir)
            except:
                print("no weights found, continuing with uninitialized graph!")

        # load the preprocessors
        try:
            self.pre = PCAWhiteningPreprocessor.from_file(os.path.join(indir, "pre.pkl"))
            self.pre_nuisance = PCAWhiteningPreprocessor.from_file(os.path.join(indir, "pre_nuis.pkl"))
            print("preprocessors successfully loaded from " + indir)
        except FileNotFoundError:
            print("no preprocessors found")

    def save(self, outdir):
        
        # save the weights in the graph
        with self.graph.as_default():
            self.saver.save(self.sess, os.path.join(outdir, "model.dat"))

        # save the preprocessors
        self.pre.save(os.path.join(outdir, "pre.pkl"))
        self.pre_nuisance.save(os.path.join(outdir, "pre_nuis.pkl"))

        config_path = os.path.join(outdir, "meta.conf")
        gconfig = ConfigParser()
        gconfig.read(config_path) # start from the current version of the config file and add changes on top
        gconfig["AdversarialModel"] = self.global_pars
        gconfig[self.classifier_model.name] = self.classifier_model.hyperpars
        gconfig[self.adversary_model.name] = self.adversary_model.hyperpars
        with open(config_path, 'w') as metafile:
            gconfig.write(metafile)

    def get_model_statistics(self, data, nuisances, labels, weights_step, postfix = "", DisCo_lambda = 5.0):

        stat_dict = {}

        clf_loss, adv_loss, total_loss, private_DisCo_adv_loss, private_DisCo_total_loss = self.evaluate_all_losses(data, nuisances, labels, weights_step, DisCo_lambda)

        stat_dict["clf_loss" + postfix] = clf_loss
        stat_dict["adv_loss" + postfix] = adv_loss
        stat_dict["total_loss" + postfix] = total_loss[0]
        stat_dict["total_loss_private_DisCo" + postfix] = private_DisCo_total_loss[0]
        stat_dict["adv_loss_private_DisCo" + postfix] = private_DisCo_adv_loss

        return stat_dict
        
    def create_paramdict(self):
        paramdict = {}

        for key, val in self.global_pars.items():
            paramdict[key] = val

        for key, val in self.classifier_model.hyperpars.items():
            paramdict[self.classifier_model.name + "_" + key] = val

        for key, val in self.adversary_model.hyperpars.items():
            paramdict[self.adversary_model.name + "_" + key] = val

        return paramdict

