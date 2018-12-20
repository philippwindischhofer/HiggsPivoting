import tensorflow as tf
import tensorflow.contrib.layers as layers

class MINEModel:
    def __init__(self, name, hyperpars):
        self.name = name
        self.num_hidden_layers = hyperpars["num_hidden_layers"]
        self.num_units = hyperpars["num_units"]

    def MINE_network(self, data_input):
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            lay = data_input

            for layer in range(self.num_hidden_layers):
                lay = layers.relu(lay, self.num_units)

            outputs = layers.linear(lay, 1)

        these_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self.name)
        return outputs, these_vars

    def MINE_loss(self, data_X, data_Y):
        data_Y_shuffled = tf.random_shuffle(data_Y)

        data_xy = tf.concat([data_X, data_Y], axis = 1)
        data_x_y = tf.concat([data_X, data_Y_shuffled], axis = 1)

        T_xy, MINE_vars = self.MINE_network(data_xy)
        T_x_y, MINE_vars_cc = self.MINE_network(data_x_y)

        print(MINE_vars)
        print(MINE_vars_cc)

        MINE_lossval = -(tf.reduce_mean(T_xy, axis = 0) - tf.math.log(tf.reduce_mean(tf.math.exp(T_x_y), axis = 0)))
        MINE_lossval = MINE_lossval[0]

        return MINE_lossval, MINE_vars
