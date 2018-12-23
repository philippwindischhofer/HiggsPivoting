import tensorflow as tf
import tensorflow.contrib.layers as layers

class SimpleModel:
    def __init__(self, name, hyperpars):
        self.name = name
        self.hyperpars = hyperpars

    def classifier(self, classifier_input):
        with tf.variable_scope(self.name):
            lay = classifier_input

            for layer in range(int(self.hyperpars["num_hidden_layers"])):
                lay = layers.relu(lay, int(self.hyperpars["num_units"]))

            lay = layers.relu(lay, 2)
            outputs = layers.softmax(lay)

        these_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self.name)
    
        return outputs, these_vars
