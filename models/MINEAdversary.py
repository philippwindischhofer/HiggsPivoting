import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from models.BaseModels import AdversaryModel

class MINEAdversary(AdversaryModel):

    def __init__(self, name, hyperpars):
        self.name = name
        self.hyperpars = hyperpars

    def build_loss(self, pred, nuisance, weights = 1.0):
        nuisance_shuffled = tf.random_shuffle(nuisance)

        data_xy = tf.concat([pred, nuisance], axis = 1)
        data_x_y = tf.concat([pred, nuisance_shuffled], axis = 1)

        T_xy, these_vars = self._adversary_model(data_xy)
        T_x_y, these_vars_cc = self._adversary_model(data_x_y)

        #MINE_lossval = -(tf.reduce_mean(T_xy * weights, axis = 0) - tf.math.log(tf.reduce_mean(tf.math.exp(T_x_y) * weights, axis = 0)))
        MINE_lossval = -(tf.reduce_mean(T_xy * weights, axis = 0) - tf.reduce_mean(tf.math.exp(T_x_y - 1) * weights, axis = 0))
        MINE_lossval = MINE_lossval[0]

        return MINE_lossval, these_vars

    def _adversary_model(self, in_tensor):
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            lay = in_tensor

            for layer in range(int(float(self.hyperpars["num_hidden_layers"]))):
                lay = layers.relu(lay, int(float(self.hyperpars["num_units"])))

            outputs = layers.linear(lay, 1)

        these_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self.name)
        return outputs, these_vars

