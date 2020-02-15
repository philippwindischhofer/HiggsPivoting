import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from models.BaseModels import AdversaryModel

class PtEstAdversary(AdversaryModel):

    def __init__(self, name, hyperpars):
        self.name = name
        self.hyperpars = hyperpars

    def build_loss(self, pred, nuisance, is_training, weights = 1.0, eps = 1e-6, batchnum = 0):
        with tf.variable_scope(self.name):
            self.regressed_nuisance, self.these_vars = self._adversary_model(pred, is_training)
            self.loss = tf.reduce_mean(weights * tf.squeeze(tf.math.square(self.regressed_nuisance - nuisance), axis = 1))

        return self.loss, self.these_vars

    def _adversary_model(self, in_tensor, is_training):
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            lay = in_tensor

            for layer in range(int(float(self.hyperpars["num_hidden_layers"]))):
                lay = layers.relu(lay, int(float(self.hyperpars["num_units"])), weights_initializer = layers.xavier_initializer(seed = 12345),
                                  weights_regularizer = layers.l2_regularizer(scale = 0.0))
                lay = layers.dropout(lay, keep_prob = 1 - float(self.hyperpars["dropout_rate"]), is_training = is_training)

            outputs = layers.linear(lay, 1)

        these_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self.name)
        return outputs, these_vars

