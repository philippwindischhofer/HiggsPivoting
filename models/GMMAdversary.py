import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from models.BaseModels import AdversaryModel

class GMMAdversary(AdversaryModel):

    def __init__(self, name, hyperpars):
        print("initializing a GMMAdversary")
        self.name = name
        self.hyperpars = hyperpars

    def build_loss(self, pred, nuisance, is_training, weights = 1.0, batchnum = 0):
        mu, sigma, frac, these_vars = self._adversary_model(pred)

        # mu, sigma and frac are of shape (batch_size, num_components)
        comps_val = 1.0 / (np.sqrt(2.0 * np.pi)) * frac / sigma * tf.math.exp(-0.5 * tf.square(mu - nuisance) / tf.square(sigma))

        pdfval = tf.reduce_sum(comps_val, axis = 1) # sum over components
        logpdf = tf.math.log(pdfval + 1e-5)
        loglik = tf.reduce_mean(logpdf * weights, axis = 0) # sum over batch

        return -loglik, these_vars

    def _adversary_model(self, in_tensor):
        with tf.variable_scope(self.name):
            lay = in_tensor

            for layer in range(int(float(self.hyperpars["num_hidden_layers"]))):
                lay = layers.relu(lay, int(float(self.hyperpars["num_units"])))

            nc = int(float(self.hyperpars["num_components"]))
            pre_output = layers.linear(lay, 3 * nc)
            
            mu_val = pre_output[:,:nc] # no sign restriction on the mean values
            sigma_val = tf.exp(pre_output[:,nc:2*nc]) # standard deviations need to be positive
            frac_val = tf.nn.softmax(pre_output[:,2*nc:3*nc]) # the mixture fractions need to add up to unity

        these_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self.name)

        return mu_val, sigma_val, frac_val, these_vars

