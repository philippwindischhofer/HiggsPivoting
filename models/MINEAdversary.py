import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from models.BaseModels import AdversaryModel

class MINEAdversary(AdversaryModel):

    def __init__(self, name, hyperpars):
        self.name = name
        self.hyperpars = hyperpars

    def build_loss(self, pred, nuisance, is_training, weights = 1.0, eps = 1e-6, batchnum = 0):
        with tf.variable_scope(self.name):
            self.idx = tf.range(tf.squeeze(tf.shape(weights)))
            self.idx_shuffled = tf.random_shuffle(self.idx)

            self.nuisance_shuffled = tf.gather(nuisance, self.idx_shuffled)
            self.weights_shuffled = tf.gather(weights, self.idx_shuffled)

            self.data_xy = tf.concat([pred, nuisance], axis = 1)
            self.data_x_y = tf.concat([pred, self.nuisance_shuffled], axis = 1)
            
            self.T_xy, self.these_vars = self._adversary_model(self.data_xy, is_training)
            self.T_x_y, self.these_vars_cc = self._adversary_model(self.data_x_y, is_training)
            
            self.T_xy = tf.squeeze(self.T_xy)
            self.T_x_y = tf.squeeze(self.T_x_y)

            self.MINE_lossval = -(tf.reduce_mean(self.T_xy * weights * weights, axis = 0) - tf.reduce_mean(tf.math.exp(self.T_x_y - 1.0) * weights * self.weights_shuffled, axis = 0))
            self.MINE_lossval = self.MINE_lossval / tf.reduce_mean(weights)
            #self.MINE_lossval_unrolled = tf.math.multiply(-(self.T_xy - tf.math.exp(self.T_x_y - 1)), weights)
            #self.MINE_lossval = tf.reduce_mean(self.MINE_lossval_unrolled)
            #self.MINE_lossval = -(tf.reduce_mean(self.T_xy * weights, axis = 0) - tf.reduce_mean(tf.math.exp(self.T_x_y - 1) * weights, axis = 0))
            #self.MINE_lossval = self.MINE_lossval[0]

        return self.MINE_lossval, self.these_vars

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

