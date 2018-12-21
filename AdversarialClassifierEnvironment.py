import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from TFEnvironment import TFEnvironment
from PCAWhiteningPreprocessor import PCAWhiteningPreprocessor

class AdversarialClassifierEnvironment(TFEnvironment):
    
    def __init__(self, classifier_model):
        super(AdversarialClassifierEnvironment, self).__init__()
        self.classifier_model = classifier_model
        self.pre = None
        self.pre_nuisance = None

    def build(self, num_inputs, num_nuisances, lambda_val):
        print("building AdversarialClassifierEnvironment using lambda = {}".format(lambda_val))
        with self.graph.as_default():
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

            # set up the adversary
            self.classifier_out_single = tf.expand_dims(self.classifier_out[:,0], axis = 1)
            self.mu, self.sigma, self.frac, self.adversary_vars = self._build_parameter_adversary(in_tensor = self.classifier_out_single,
                                                                                                  hyperpars = {"num_hidden_layers": 3, "num_units": 30, "num_components": 5})

            self.GMM_loss = -self._GMM_loglik(self.mu, self.sigma, self.frac, num_components = 5, data = self.nuisances_in)
            self.adv_loss = self.classification_loss + lambda_val * (-self.GMM_loss)
            
            # optimizers for the classifier and the adversary
            self.train_classifier_op = tf.train.AdamOptimizer(learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999).minimize(self.classification_loss, var_list = self.classifier_vars)
            self.train_adversary_op = tf.train.AdamOptimizer(learning_rate = 0.01, beta1 = 0.3, beta2 = 0.5).minimize(self.GMM_loss, var_list = self.adversary_vars)
            self.train_adv_op = tf.train.AdamOptimizer(learning_rate = 0.01, beta1 = 0.3, beta2 = 0.5).minimize(self.adv_loss, var_list = self.classifier_vars)

            self.saver = tf.train.Saver()

    def _build_parameter_adversary(self, in_tensor, hyperpars, name = "parest_net"):
        with tf.variable_scope(name):
            lay = in_tensor

            for layer in range(hyperpars["num_hidden_layers"]):
                lay = layers.relu(lay, hyperpars["num_units"])

            nc = hyperpars["num_components"]
            pre_output = layers.linear(lay, 3 * nc)
            
            mu_val = pre_output[:,:nc] # no sign restriction on the mean values
            sigma_val = tf.exp(pre_output[:,nc:2*nc]) # standard deviations need to be positive
            frac_val = tf.nn.softmax(pre_output[:,2*nc:3*nc]) # the mixture fractions need to add up to unity

        these_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = name)

        return mu_val, sigma_val, frac_val, these_vars

    # returns the log(L) of 'data' under the Gaussian mixture model given by (mu, sigma, frac)
    def _GMM_loglik(self, mu, sigma, frac, num_components, data):
        # mu, sigma and frac are of shape (batch_size, num_components)
        comps_val = 1.0 / (np.sqrt(2.0 * np.pi)) * frac / sigma * tf.math.exp(-0.5 * tf.square(mu - data) / tf.square(sigma))

        pdfval = tf.reduce_sum(comps_val, axis = 1) # sum over components
        logpdf = tf.math.log(pdfval + 1e-5)
        loglik = tf.reduce_mean(logpdf, axis = 0) # sum over batch

        return loglik

    def init(self, data_train, data_nuisance):
        self.pre.setup(data_train)
        self.pre_nuisance.setup(data_nuisance)

        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

    def train_step(self, data_step, nuisances_step, labels_step):
        data_pre = self.pre.process(data_step)
        nuisances_pre = self.pre_nuisance.process(nuisances_step)

        with self.graph.as_default():
            self.sess.run(self.train_adv_op, feed_dict = {self.data_in: data_pre, self.nuisances_in: nuisances_pre, self.labels_in: labels_step})

    def train_adversary(self, data_step, nuisances_step, labels_step):
        data_pre = self.pre.process(data_step)
        nuisances_pre = self.pre_nuisance.process(nuisances_step)

        with self.graph.as_default():
            self.sess.run(self.train_adversary_op, feed_dict = {self.data_in: data_pre, self.nuisances_in: nuisances_pre, self.labels_in: labels_step})

    def evaluate_classifier_loss(self, data, labels):
        data_pre = self.pre.process(data)
        with self.graph.as_default():
            classifier_lossval = self.sess.run(self.classification_loss, feed_dict = {self.data_in: data_pre, self.labels_in: labels})
        return classifier_lossval

    def evaluate_adversary_loss(self, data, nuisances, labels):
        data_pre = self.pre.process(data)
        nuisances_pre = self.pre_nuisance.process(nuisances)

        with self.graph.as_default():
            retval = self.sess.run(self.GMM_loss, feed_dict = {self.data_in: data_pre, self.nuisances_in: nuisances_pre, self.labels_in: labels})

        return retval

    def dump_loss_information(self, data, nuisances, labels):
        classifier_lossval = self.evaluate_classifier_loss(data, labels)
        adversary_lossval = self.evaluate_adversary_loss(data, nuisances, labels)
        print("classifier loss: {:.4f}, adv. loss = {:.4f}".format(classifier_lossval, adversary_lossval))

    def predict(self, data):
        data_pre = self.pre.process(data)

        with self.graph.as_default():
            retval = self.sess.run(self.classifier_out, feed_dict = {self.data_in: data_pre})

        return retval

    def get_model_statistics(self, data, nuisances, labels):
        retdict = {}
        retdict["class. loss"] = self.evaluate_classifier_loss(data, labels)
        retdict["adv. loss"] = self.evaluate_adversary_loss(data, nuisances, labels)
        return retdict

    def load(self, file_path):
        with self.graph.as_default():
            self.saver.restore(self.sess, file_path)

        self.pre = PCAWhiteningPreprocessor.from_file(os.path.join(os.path.dirname(file_path), 'pre.pkl'))
        self.pre_nuisance = PCAWhiteningPreprocessor.from_file(os.path.join(os.path.dirname(file_path), 'pre_nuis.pkl'))

    def save(self, file_path):
        with self.graph.as_default():
            self.saver.save(self.sess, file_path)

        self.pre.save(os.path.join(os.path.dirname(file_path), 'pre.pkl'))
        self.pre_nuisance.save(os.path.join(os.path.dirname(file_path), 'pre_nuis.pkl'))

