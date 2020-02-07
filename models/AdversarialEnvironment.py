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
from models.JSAdversary import JSAdversary
from models.DisCoAdversary import DisCoAdversary
from models.PtEstAdversary import PtEstAdversary
from base.PCAWhiteningPreprocessor import PCAWhiteningPreprocessor
from base.Configs import TrainingConfig

class AdversarialEnvironment(TFEnvironment):
    
    def __init__(self, classifier_model_2j, classifier_model_3j, adversary_model_2j, adversary_model_3j, global_pars):
        super(AdversarialEnvironment, self).__init__()
        self.classifier_model_2j = classifier_model_2j
        self.classifier_model_3j = classifier_model_3j
        self.adversary_model_2j = adversary_model_2j
        self.adversary_model_3j = adversary_model_3j

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

        mod_2j = classifier_model("class_2j", hyperpars = classifier_hyperpars)
        mod_3j = classifier_model("class_3j", hyperpars = classifier_hyperpars)
        adv_2j = adversary_model("adv_2j", hyperpars = adversary_hyperpars)
        adv_3j = adversary_model("adv_3j", hyperpars = adversary_hyperpars)
        obj = cls(mod_2j, mod_3j, adv_2j, adv_3j, global_pars)

        # then re-build the graph using these settings
        obj.build()

        # finally, load back the weights and preprocessors, if available
        obj.load(config_dir)        

        return obj

    def build(self):

        self.lambda_final = float(self.global_pars["lambda"])
        num_inputs = int(float(self.global_pars["num_inputs"]))
        num_nuisances = int(float(self.global_pars["num_nuisances"]))

        print("building AdversarialEnvironment using lambda = {}".format(self.lambda_final))
        print("global_pars:")
        for key, val in self.global_pars.items():
            print(key + ": " + str(val))

        with self.graph.as_default():
            self.pre = PCAWhiteningPreprocessor(num_inputs)
            self.pre_nuisance = PCAWhiteningPreprocessor(num_nuisances)

            # build the remaining graph
            self.labels_in = tf.placeholder(tf.int32, [None, ], name = 'labels_in')
            self.data_in = tf.placeholder(tf.float32, [None, num_inputs], name = 'data_in')
            self.nuisances_in = tf.placeholder(tf.float32, [None, num_nuisances], name = 'nuisances_in')
            self.weights_in = tf.placeholder(tf.float32, [None, ], name = 'weights_in')
            self.batchnum = tf.placeholder(tf.float32, [1], name = 'batchnum')
            self.is_training = tf.placeholder(tf.bool, name = 'is_training')
            self.lambdaval = tf.placeholder(tf.float32, [1], name = 'lambdaval')
            self.nJ_in = tf.placeholder(tf.float32, [None, ], name = 'nJ_in')
            self.classifier_2j_lr = tf.placeholder(tf.float32, [], name = "classifier_2j_lr")
            self.classifier_3j_lr = tf.placeholder(tf.float32, [], name = "classifier_3j_lr")
            self.adversary_2j_lr = tf.placeholder(tf.float32, [], name = "adversary_2j_lr")
            self.adversary_3j_lr = tf.placeholder(tf.float32, [], name = "adversary_3j_lr")

            self.labels_one_hot = tf.one_hot(self.labels_in, depth = 2)

            # set the weights separately for 2j / 3j to effectively route the events into the two separate adversaries
            self.weights_2j = tf.where(tf.math.less(self.nJ_in, 2.5), self.weights_in, tf.zeros_like(self.weights_in))
            self.weights_3j = tf.where(tf.math.greater(self.nJ_in, 2.5), self.weights_in, tf.zeros_like(self.weights_in))

            # set up the classifier models, separately for 2j and 3j
            self.classifier_out_2j, self.classifier_vars_2j = self.classifier_model_2j.build_model(self.data_in, is_training = self.is_training)
            self.classification_loss_2j = self.classifier_model_2j.build_loss(self.classifier_out_2j, self.labels_one_hot, weights = self.weights_2j, batchnum = self.batchnum)

            self.classifier_out_3j, self.classifier_vars_3j = self.classifier_model_3j.build_model(self.data_in, is_training = self.is_training)
            self.classification_loss_3j = self.classifier_model_3j.build_loss(self.classifier_out_3j, self.labels_one_hot, weights = self.weights_3j, batchnum = self.batchnum)

            self.classifier_out_single_2j = tf.expand_dims(self.classifier_out_2j[:,0], axis = 1)
            self.classifier_out_single_3j = tf.expand_dims(self.classifier_out_3j[:,0], axis = 1)

            self.classifier_out = tf.where(tf.math.less(self.nJ_in, 2.5), self.classifier_out_2j, self.classifier_out_3j)

            # set up the model for the adversary
            self.adv_loss_2j, self.adversary_vars_2j = self.adversary_model_2j.build_loss(self.classifier_out_single_2j, self.nuisances_in, weights = self.weights_2j, batchnum = self.batchnum, is_training = self.is_training)
            self.adv_loss_3j, self.adversary_vars_3j = self.adversary_model_3j.build_loss(self.classifier_out_single_3j, self.nuisances_in, weights = self.weights_3j, batchnum = self.batchnum, is_training = self.is_training)

            self.print_0 = tf.print("nJ", self.nJ_in)
            self.print_1 = tf.print("weights (2j)", self.weights_2j)
            self.print_2 = tf.print("weights (3j)", self.weights_3j)

            # collect the regularization losses
            self.regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

            self.adv_loss = self.adv_loss_2j + self.adv_loss_3j
            self.classification_loss = self.classification_loss_2j + self.classification_loss_3j

            self.total_loss_2j = self.classification_loss_2j + self.lambdaval * (-self.adv_loss_2j)
            self.total_loss_3j = self.classification_loss_3j + self.lambdaval * (-self.adv_loss_3j)

            # set up the optimizers for both classifier and adversary
            self.train_classifier_standalone_2j = tf.train.AdamOptimizer(learning_rate = self.classifier_2j_lr, 
                                                                         beta1 = float(self.global_pars["adam_clf_beta1"]), 
                                                                         beta2 = float(self.global_pars["adam_clf_beta2"]), 
                                                                         epsilon = float(self.global_pars["adam_clf_eps"])).minimize(self.classification_loss_2j, var_list = self.classifier_vars_2j)

            self.train_classifier_standalone_3j = tf.train.AdamOptimizer(learning_rate = self.classifier_3j_lr,
                                                                         beta1 = float(self.global_pars["adam_clf_beta1"]), 
                                                                         beta2 = float(self.global_pars["adam_clf_beta2"]), 
                                                                         epsilon = float(self.global_pars["adam_clf_eps"])).minimize(self.classification_loss_3j, var_list = self.classifier_vars_3j)

            self.train_adversary_standalone_2j = tf.train.AdamOptimizer(learning_rate = self.adversary_2j_lr,
                                                                        beta1 = float(self.global_pars["adam_adv_beta1"]), 
                                                                        beta2 = float(self.global_pars["adam_adv_beta2"]), 
                                                                        epsilon = float(self.global_pars["adam_adv_eps"])).minimize(self.adv_loss_2j, var_list = self.adversary_vars_2j)

            self.train_adversary_standalone_3j = tf.train.AdamOptimizer(learning_rate = self.adversary_3j_lr,
                                                                        beta1 = float(self.global_pars["adam_adv_beta1"]), 
                                                                        beta2 = float(self.global_pars["adam_adv_beta2"]), 
                                                                        epsilon = float(self.global_pars["adam_adv_eps"])).minimize(self.adv_loss_3j, var_list = self.adversary_vars_3j)

            self.train_classifier_adv_2j = tf.train.AdamOptimizer(learning_rate = self.classifier_2j_lr, 
                                                                  beta1 = float(self.global_pars["adam_clf_adv_beta1"]), 
                                                                  beta2 = float(self.global_pars["adam_clf_adv_beta2"]), 
                                                                  epsilon = float(self.global_pars["adam_clf_adv_eps"])).minimize(self.total_loss_2j, var_list = self.classifier_vars_2j)            

            self.train_classifier_adv_3j = tf.train.AdamOptimizer(learning_rate = self.classifier_3j_lr, 
                                                                  beta1 = float(self.global_pars["adam_clf_adv_beta1"]), 
                                                                  beta2 = float(self.global_pars["adam_clf_adv_beta2"]), 
                                                                  epsilon = float(self.global_pars["adam_clf_adv_eps"])).minimize(self.total_loss_3j, var_list = self.classifier_vars_3j)            

            self.saver = tf.train.Saver()

    def init(self, data_train, data_nuisance):
        self.pre.setup(data_train)
        self.pre_nuisance.setup(data_nuisance)

        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
    
    def _lr_scheduler(self, lr_start, lr_decay, batchnum):
        return lr_start * np.exp(-lr_decay * batchnum)

    def train_step(self, data_step, nuisances_step, labels_step, weights_step, batchnum, auxdat_step):
        data_pre = self.pre.process(data_step)
        nuisances_pre = self.pre_nuisance.process(nuisances_step)
        weights_step = weights_step.flatten()

        # determine the current learning rate as per the scheduling
        classifier_lr = self._lr_scheduler(lr_start = float(self.global_pars["adam_clf_adv_lr"]),
                                           lr_decay = float(self.global_pars["adam_clf_adv_lr_decay"]),
                                           batchnum = batchnum)

        print("cur LR = {}".format(classifier_lr))

        # use a constant lambda for the time being
        lambda_cur = self.lambda_final

        with self.graph.as_default():
            self.sess.run(self.train_classifier_adv_2j, feed_dict = {self.data_in: data_pre, self.nuisances_in: nuisances_pre, self.labels_in: labels_step, self.weights_in: weights_step, self.lambdaval: [lambda_cur], self.batchnum: [batchnum], self.is_training: True, self.nJ_in: auxdat_step[:, TrainingConfig.auxiliary_branches.index("nJ")], self.classifier_2j_lr: classifier_lr, self.classifier_3j_lr: classifier_lr})
            self.sess.run(self.train_classifier_adv_3j, feed_dict = {self.data_in: data_pre, self.nuisances_in: nuisances_pre, self.labels_in: labels_step, self.weights_in: weights_step, self.lambdaval: [lambda_cur], self.batchnum: [batchnum], self.is_training: True, self.nJ_in: auxdat_step[:, TrainingConfig.auxiliary_branches.index("nJ")], self.classifier_2j_lr: classifier_lr, self.classifier_3j_lr: classifier_lr})

    def train_adversary(self, data_step, nuisances_step, labels_step, weights_step, batchnum, auxdat_step):
        data_pre = self.pre.process(data_step)
        nuisances_pre = self.pre_nuisance.process(nuisances_step)
        weights_step = weights_step.flatten()

        # determine the current learning rate as per the scheduling
        adversary_lr = self._lr_scheduler(lr_start = float(self.global_pars["adam_adv_lr"]),
                                          lr_decay = float(self.global_pars["adam_adv_lr_decay"]),
                                          batchnum = batchnum)

        with self.graph.as_default():
            print("train 2j")
            self.sess.run([self.train_adversary_standalone_2j], feed_dict = {self.data_in: data_pre, self.nuisances_in: nuisances_pre, self.labels_in: labels_step, self.weights_in: weights_step, self.batchnum: [batchnum], self.is_training: True, self.nJ_in: auxdat_step[:, TrainingConfig.auxiliary_branches.index("nJ")], self.adversary_2j_lr: adversary_lr, self.adversary_3j_lr: adversary_lr})
            print("train 3j")
            self.sess.run([self.train_adversary_standalone_3j], feed_dict = {self.data_in: data_pre, self.nuisances_in: nuisances_pre, self.labels_in: labels_step, self.weights_in: weights_step, self.batchnum: [batchnum], self.is_training: True, self.nJ_in: auxdat_step[:, TrainingConfig.auxiliary_branches.index("nJ")], self.adversary_2j_lr: adversary_lr, self.adversary_3j_lr: adversary_lr})

    def train_classifier(self, data_step, labels_step, weights_step, batchnum, auxdat_step):
        data_pre = self.pre.process(data_step)
        weights_step = weights_step.flatten()

        # determine the current learning rate as per the scheduling
        adversary_lr = self._lr_scheduler(lr_start = float(self.global_pars["adam_clf_lr"]),
                                          lr_decay = float(self.global_pars["adam_clf_lr_decay"]),
                                          batchnum = batchnum)

        with self.graph.as_default():
            self.sess.run(self.train_classifier_standalone_2j, feed_dict = {self.data_in: data_pre, self.labels_in: labels_step, self.weights_in: weights_step, self.batchnum: [batchnum], self.is_training: True, self.nJ_in: auxdat_step[:, TrainingConfig.auxiliary_branches.index("nJ")], self.classifier_2j_lr: classifier_lr, self.classifier_3j_lr: classifier_lr})
            self.sess.run(self.train_classifier_standalone_3j, feed_dict = {self.data_in: data_pre, self.labels_in: labels_step, self.weights_in: weights_step, self.batchnum: [batchnum], self.is_training: True, self.nJ_in: auxdat_step[:, TrainingConfig.auxiliary_branches.index("nJ")], self.classifier_2j_lr: classifier_lr, self.classifier_3j_lr: classifier_lr})

    def evaluate_classifier_loss(self, data, labels, weights_step, auxdat_step):
        data_pre = self.pre.process(data)
        weights_step = weights_step.flatten()

        with self.graph.as_default():
            classifier_lossval_2j = self.sess.run(self.classification_loss_2j, feed_dict = {self.data_in: data_pre, self.labels_in: labels, self.weights_in: weights_step, self.is_training: True, self.nJ_in: auxdat_step[:, TrainingConfig.auxiliary_branches.index("nJ")]})
            classifier_lossval_3j = self.sess.run(self.classification_loss_3j, feed_dict = {self.data_in: data_pre, self.labels_in: labels, self.weights_in: weights_step, self.is_training: True, self.nJ_in: auxdat_step[:, TrainingConfig.auxiliary_branches.index("nJ")]})
        return classifier_lossval_2j, classifier_lossval_3j

    def evaluate_adversary_loss(self, data, nuisances, labels, weights_step, batchnum, auxdat_step):
        data_pre = self.pre.process(data)
        nuisances_pre = self.pre_nuisance.process(nuisances)
        weights_step = weights_step.flatten()

        with self.graph.as_default():
            adv_loss_2j = self.sess.run(self.adv_loss_2j, feed_dict = {self.data_in: data_pre, self.nuisances_in: nuisances_pre, self.labels_in: labels, self.weights_in: weights_step, self.batchnum: [batchnum], self.is_training: True, self.nJ_in: auxdat_step[:, TrainingConfig.auxiliary_branches.index("nJ")]})
            adv_loss_3j = self.sess.run(self.adv_loss_3j, feed_dict = {self.data_in: data_pre, self.nuisances_in: nuisances_pre, self.labels_in: labels, self.weights_in: weights_step, self.batchnum: [batchnum], self.is_training: True, self.nJ_in: auxdat_step[:, TrainingConfig.auxiliary_branches.index("nJ")]})

        return adv_loss_2j, adv_loss_3j

    def dump_loss_information(self, data, nuisances, labels, weights, auxdat_step):
        classifier_lossval_2j, classifier_lossval_3j = self.evaluate_classifier_loss(data, labels, weights, auxdat_step)
        adversary_lossval_2j, adversary_lossval_3j = self.evaluate_adversary_loss(data, nuisances, labels, weights, 0, auxdat_step)
        weights = weights.flatten()
        print("classifier loss (2j): {:.4e}, classifier loss (3j): {:.4e}, adv. loss (2j) = {:.4e}, adv. loss (3j) = {:.4e}".format(classifier_lossval_2j, classifier_lossval_3j, adversary_lossval_2j, adversary_lossval_3j))

    # use the model to make predictions on 'data', adhering to a certain batch size for evolution
    def predict(self, data, auxdat = None, pred_size = 256, use_dropout = True):
        if auxdat is None:
            nJ = data[:, TrainingConfig.training_branches.index("nJ")]
        else:
            nJ = auxdat[:, TrainingConfig.auxiliary_branches.index("nJ")]
        data_pre = self.pre.process(data)

        datlen = len(data_pre)
        chunks = np.split(data_pre, datlen / pred_size, axis = 0)
        nJ_chunks = np.split(nJ, datlen / pred_size, axis = 0)

        retvals = []
        for chunk, nJ_chunk in zip(chunks, nJ_chunks):
            with self.graph.as_default():
                retval_cur = self.sess.run(self.classifier_out, feed_dict = {self.data_in: chunk, self.is_training: use_dropout, self.nJ_in: nJ_chunk})
                retvals.append(retval_cur)

        return np.concatenate(retvals, axis = 0)

    def get_model_statistics(self, data, nuisances, labels, weights, auxdat_step):
        retdict = {}
        retdict["class. loss (2j)"], retdict["class. loss (3j)"] = self.evaluate_classifier_loss(data, labels, weights, auxdat_step)
        retdict["adv. loss (2j)"], retdict["adv. loss (3j)"] = self.evaluate_adversary_loss(data, nuisances, labels, weights, 0, auxdat_step)
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
        config_path = os.path.join(outdir, "meta.conf")
        gconfig = ConfigParser()
        gconfig.read(config_path) # start from the current version of the config file and add changes on top
        gconfig["AdversarialEnvironment"] = self.global_pars
        gconfig[self.classifier_model_2j.__class__.__name__] = self.classifier_model_2j.hyperpars
        gconfig[self.adversary_model_2j.__class__.__name__] = self.adversary_model_2j.hyperpars
        with open(config_path, 'w') as metafile:
            gconfig.write(metafile)

    # return a dictionary with the list of all parameters that characterize this adversarial model
    def create_paramdict(self):
        paramdict = {}

        for key, val in self.global_pars.items():
            paramdict[key] = val

        for key, val in self.classifier_model_2j.hyperpars.items():
            paramdict[self.classifier_model_2j.name + "_" + key] = val

        for key, val in self.adversary_model_2j.hyperpars.items():
            paramdict[self.adversary_model_2j.name + "_" + key] = val

        return paramdict
