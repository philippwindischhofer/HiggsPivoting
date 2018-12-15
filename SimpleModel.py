import tensorflow as tf
import tensorflow.contrib.layers as layers

class SimpleModel:
    def __init__(self, name, hyperpars):
        self.name = name
        self.num_hidden_layers = hyperpars["num_hidden_layers"]
        self.num_units = hyperpars["num_units"]

    def classfier(self, classifier_input):
        with tf.variable_scope(self.name):
            lay = classifier_input

            for layer in range(self.num_hidden_layers):
                lay = layers.relu(lay, self.num_units)

            lay = layers.relu(lay, 2)
            outputs = layers.softmax(lay)

        these_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self.name)
    
        return outputs, these_vars

