import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from configparser import ConfigParser

from training.TFEnvironment import TFEnvironment
from models.SimpleClassifier import SimpleClassifier
from models.SimpleProbabilisticClassifier import SimpleProbabilisticClassifier
from models.GMMAdversary import GMMAdversary
from models.MINEAdversary import MINEAdversary
from base.PCAWhiteningPreprocessor import PCAWhiteningPreprocessor

class AdversarialEnvironment(TFEnvironment):
    
    def __init__(self, classifier_model, adversary_model, global_pars):
        super(AdversarialEnvironment, self).__init__()
        self.classifier_model = classifier_model
        self.adversary_model = adversary_model

        self.pre = None
        self.pre_nuisance = None

        self.global_pars = global_pars

    # attempt to reconstruct a previously built graph, including loading back its weights
    @classmethod
    def from_file(cls, config_dir):
        # first, load back the meta configuration variables of the graph
        gconfig = ConfigParser()
        gconfig.read(os.path.join(config_dir, "meta.conf"))

        global_pars = gconfig["AdversarialEnvironment"]
        classifier_model_type = global_pars["classifier_model"]
        adversary_model_type = global_pars["adversary_model"]
        classifier_model = eval(classifier_model_type)
        adversary_model = eval(adversary_model_type)

        classifier_hyperpars = gconfig[classifier_model_type]
        adversary_hyperpars = gconfig[adversary_model_type]

        mod = classifier_model("class", hyperpars = classifier_hyperpars)
        adv = adversary_model("adv", hyperpars = adversary_hyperpars)
        obj = cls(mod, adv, global_pars)

        # then re-build the graph using these settings
        obj.build()

        # finally, load back the weights and preprocessors, if available
        obj.load(config_dir)        

        return obj

    def build(self):

        lambda_val = float(self.global_pars["lambda"])
        num_inputs = int(float(self.global_pars["num_inputs"]))
        num_nuisances = int(float(self.global_pars["num_nuisances"]))

        print("building AdversarialEnvironment using lamba = {}".format(lambda_val))
        with self.graph.as_default():
            self.pre = PCAWhiteningPreprocessor(num_inputs)
            self.pre_nuisance = PCAWhiteningPreprocessor(num_nuisances)

            # build the remaining graph
            self.labels_in = tf.placeholder(tf.int32, [None, ], name = 'labels_in')
            self.data_in = tf.placeholder(tf.float32, [None, num_inputs], name = 'data_in')
            self.nuisances_in = tf.placeholder(tf.float32, [None, num_nuisances], name = 'nuisances_in')

            # set up the classifier model
            self.classifier_out, self.classifier_vars = self.classifier_model.build_model(self.data_in)
            self.labels_one_hot = tf.one_hot(self.labels_in, depth = 2)
            self.classification_loss = self.classifier_model.build_loss(self.classifier_out, self.labels_one_hot)

            # set up the model for the adversary
            self.classifier_out_single = tf.expand_dims(self.classifier_out[:,0], axis = 1)
            self.adv_loss, self.adversary_vars = self.adversary_model.build_loss(self.classifier_out_single, self.nuisances_in)

            self.total_loss = self.classification_loss + lambda_val * (-self.adv_loss)

            # set up the optimizers for both classifier and adversary
            self.train_classifier_standalone = tf.train.AdamOptimizer(learning_rate = 0.003, beta1 = 0.9, beta2 = 0.999).minimize(self.classification_loss, var_list = self.classifier_vars)
            self.train_adversary_standalone = tf.train.AdamOptimizer(learning_rate = 0.005, beta1 = 0.9, beta2 = 0.999).minimize(self.adv_loss, var_list = self.adversary_vars)
            self.train_classifier_adv = tf.train.AdamOptimizer(learning_rate = 0.003, beta1 = 0.3, beta2 = 0.5).minimize(self.total_loss, var_list = self.classifier_vars)

            self.saver = tf.train.Saver()

    def init(self, data_train, data_nuisance):
        self.pre.setup(data_train)
        self.pre_nuisance.setup(data_nuisance)

        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
    
    def train_step(self, data_step, nuisances_step, labels_step):
        data_pre = self.pre.process(data_step)
        nuisances_pre = self.pre_nuisance.process(nuisances_step)

        with self.graph.as_default():
            self.sess.run(self.train_classifier_adv, feed_dict = {self.data_in: data_pre, self.nuisances_in: nuisances_pre, self.labels_in: labels_step})

    def train_adversary(self, data_step, nuisances_step, labels_step):
        data_pre = self.pre.process(data_step)
        nuisances_pre = self.pre_nuisance.process(nuisances_step)

        with self.graph.as_default():
            self.sess.run(self.train_adversary_standalone, feed_dict = {self.data_in: data_pre, self.nuisances_in: nuisances_pre, self.labels_in: labels_step})

    def evaluate_classifier_loss(self, data, labels):
        data_pre = self.pre.process(data)
        with self.graph.as_default():
            classifier_lossval = self.sess.run(self.classification_loss, feed_dict = {self.data_in: data_pre, self.labels_in: labels})
        return classifier_lossval

    def evaluate_adversary_loss(self, data, nuisances, labels):
        data_pre = self.pre.process(data)
        nuisances_pre = self.pre_nuisance.process(nuisances)

        with self.graph.as_default():
            retval = self.sess.run(self.adv_loss, feed_dict = {self.data_in: data_pre, self.nuisances_in: nuisances_pre, self.labels_in: labels})

        return retval

    def dump_loss_information(self, data, nuisances, labels):
        classifier_lossval = self.evaluate_classifier_loss(data, labels)
        adversary_lossval = self.evaluate_adversary_loss(data, nuisances, labels)
        print("classifier loss: {:.4f}, adv. loss = {:.4f}".format(classifier_lossval, adversary_lossval))

    def predict(self, data):
        data_pre = self.pre.process(data)

        datlen = len(data_pre)
        pred_size = 256

        chunks = np.split(data_pre, datlen / pred_size, axis = 0)

        retvals = []
        for chunk in chunks:
            with self.graph.as_default():
                retval_cur = self.sess.run(self.classifier_out, feed_dict = {self.data_in: chunk})
                retvals.append(retval_cur)

        return np.concatenate(retvals, axis = 0)

    def get_model_statistics(self, data, nuisances, labels):
        retdict = {}
        retdict["class. loss"] = self.evaluate_classifier_loss(data, labels)
        retdict["adv. loss"] = self.evaluate_adversary_loss(data, nuisances, labels)
        return retdict

    # try to load back the environment
    def load(self, indir):
        file_path = os.path.join(indir, "model.dat")

        with self.graph.as_default():
            try:
                self.saver.restore(self.sess, file_path)
                print("weights successfully loaded for " + indir)
            except:
                print("no model checkpoint found, continuing with uninitialized graph!")
        try:
            self.pre = PCAWhiteningPreprocessor.from_file(os.path.join(os.path.dirname(file_path), 'pre.pkl'))
            self.pre_nuisance = PCAWhiteningPreprocessor.from_file(os.path.join(os.path.dirname(file_path), 'pre_nuis.pkl'))
            print("preprocessors successfully loaded for " + indir)
        except FileNotFoundError:
            print("no preprocessors found")

    # save the entire environment such that it can be set up again from here
    def save(self, outdir):
        file_path = os.path.join(outdir, "model.dat")

        # save all the variables in the graph
        with self.graph.as_default():
            self.saver.save(self.sess, file_path)

        # save the preprocessors
        self.pre.save(os.path.join(outdir, 'pre.pkl'))
        self.pre_nuisance.save(os.path.join(outdir, 'pre_nuis.pkl'))

        # save some meta-information about the graph, such that it can be fully reconstructed
        gconfig = ConfigParser()
        gconfig["AdversarialEnvironment"] = self.global_pars
        gconfig[self.classifier_model.__class__.__name__] = self.classifier_model.hyperpars
        gconfig[self.adversary_model.__class__.__name__] = self.adversary_model.hyperpars
        with open(os.path.join(outdir, "meta.conf"), 'w') as metafile:
            gconfig.write(metafile)
