import os
import numpy as np
import tensorflow as tf

from TFEnvironment import TFEnvironment
from PCAWhiteningPreprocessor import PCAWhiteningPreprocessor
from MINEModel import MINEModel

class MINEClassifierEnvironment(TFEnvironment):

    def __init__(self, classifier_model):
        super(MINEClassifierEnvironment, self).__init__()
        self.classifier_model = classifier_model
        self.pre = None
        self.pre_nuisance = None

    def build(self, num_inputs, num_nuisances, lambda_val):
        print("building MINEClassifierEnvironment using lambda = {}".format(lambda_val))
        self.pre = PCAWhiteningPreprocessor(num_inputs)
        self.pre_nuisance = PCAWhiteningPreprocessor(num_nuisances)
        
        # build the remaining graph
        self.labels_in = tf.placeholder(tf.int32, [None, ], name = 'labels_in')
        self.data_in = tf.placeholder(tf.float32, [None, num_inputs], name = 'data_in')
        self.nuisances_in = tf.placeholder(tf.float32, [None, num_nuisances], name = 'nuisances_in')

        # set up classifier and its loss
        self.classifier_out, self.classifier_vars = self.classifier_model.classifier(self.data_in)
        self.labels_one_hot = tf.one_hot(self.labels_in, depth = 2)
        self.classification_loss = tf.losses.softmax_cross_entropy(onehot_labels = self.labels_one_hot,
                                                                   logits = self.classifier_out)

        # mutual information between the classifier output and the nuisance parameters
        self.classifier_out_single = tf.expand_dims(self.classifier_out[:,0], axis = 1)
        self.MI_nuisances = MINEModel(name = "MINE_nuisances", hyperpars = {"num_hidden_layers": 3, "num_units": 30})
        self.MINE_loss, self.MINE_vars = self.MI_nuisances.MINE_loss(self.classifier_out_single, self.nuisances_in)

        # total adversarial loss
        self.adv_loss = self.classification_loss + lambda_val * (-self.MINE_loss)

        # optimizers for the classifier and MINE
        self.train_classifier = tf.train.AdamOptimizer(learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999).minimize(self.classification_loss, var_list = self.classifier_vars)
        self.train_MINE = tf.train.AdamOptimizer(learning_rate = 0.01, beta1 = 0.3, beta2 = 0.5).minimize(self.MINE_loss, var_list = self.MINE_vars)
        self.train_adv = tf.train.AdamOptimizer(learning_rate = 0.01, beta1 = 0.3, beta2 = 0.5).minimize(self.adv_loss, var_list = self.classifier_vars)
        
        self.saver = tf.train.Saver()

    def init(self, data_train, data_nuisance):
        self.pre.setup(data_train)
        self.pre_nuisance.setup(data_nuisance)
        self.sess.run(tf.global_variables_initializer())

    def train_step(self, data_step, nuisances_step, labels_step):
        data_pre = self.pre.process(data_step)
        nuisances_pre = self.pre_nuisance.process(nuisances_step)
        self.sess.run(self.train_adv, feed_dict = {self.data_in: data_pre, self.nuisances_in: nuisances_pre, self.labels_in: labels_step})

    def train_adversary(self, data_step, nuisances_step, labels_step):
        data_pre = self.pre.process(data_step)
        nuisances_pre = self.pre_nuisance.process(nuisances_step)
        self.sess.run(self.train_MINE, feed_dict = {self.data_in: data_pre, self.nuisances_in: nuisances_pre, self.labels_in: labels_step})

    def dump_loss_information(self, data, nuisances, labels):
        data_pre = self.pre.process(data)
        nuisances_pre = self.pre_nuisance.process(nuisances)
        classifier_lossval = self.sess.run(self.classification_loss, feed_dict = {self.data_in: data_pre, self.nuisances_in: nuisances_pre, self.labels_in: labels})
        mutual_information = self.sess.run(-self.MINE_loss, feed_dict = {self.data_in: data_pre, self.nuisances_in: nuisances_pre, self.labels_in: labels})

        print("classifier loss: {:.4f}, MI = {:.4f}".format(classifier_lossval, mutual_information))

    def predict(self, data):
        data_pre = self.pre.process(data)
        retval = self.sess.run(self.classifier_out, feed_dict = {self.data_in: data_pre})
        return retval

    def load(self, file_path):
        self.saver.restore(self.sess, file_path)
        self.pre = PCAWhiteningPreprocessor.from_file(os.path.join(os.path.dirname(file_path), 'pre.pkl'))
        self.pre_nuisance = PCAWhiteningPreprocessor.from_file(os.path.join(os.path.dirname(file_path), 'pre_nuis.pkl'))

    def save(self, file_path):
        self.saver.save(self.sess, file_path)
        self.pre.save(os.path.join(os.path.dirname(file_path), 'pre.pkl'))
        self.pre_nuisance.save(os.path.join(os.path.dirname(file_path), 'pre_nuis.pkl'))

    # def test_MINE(self):
    #     self.build(num_inputs = 1, num_nuisances = 1)
    #     self.init(data_train = None)

    #     # prepare some test data
    #     num_samples = 50000
    #     x = np.random.normal(loc = 0, scale = 1, size = [num_samples])
    #     y = np.random.normal(loc = 0, scale = 0.9, size = [num_samples]) 

    #     x = np.expand_dims(x, axis = 1)
    #     y = np.expand_dims(y, axis = 1)

    #     for batch in range(100):
    #         self.sess.run(self.train_MINE, feed_dict = {self.data_in: x, self.nuisances_in: y})
    #         lossval = self.sess.run(self.MINE_loss, feed_dict = {self.data_in: x, self.nuisances_in: y})
    #         print("MI = {}".format(-lossval))
