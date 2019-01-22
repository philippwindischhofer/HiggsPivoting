import tensorflow as tf
import tensorflow.contrib.layers as layers

from models.BaseModels import ClassifierModel

class SimpleClassifier(ClassifierModel):
    def __init__(self, name, hyperpars):
        print("initializing a SimpleClassifier")
        self.name = name
        self.hyperpars = hyperpars

    def build_model(self, in_tensor):
        with tf.variable_scope(self.name):
            lay = in_tensor

            for layer in range(int(float(self.hyperpars["num_hidden_layers"]))):
                lay = layers.relu(lay, int(float(self.hyperpars["num_units"])))

            lay = layers.relu(lay, 2)
            outputs = layers.softmax(lay)

        these_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self.name)
    
        return outputs, these_vars

    def build_loss(self, pred, labels_one_hot, weights = 1.0):
        classification_loss = tf.losses.softmax_cross_entropy(onehot_labels = labels_one_hot,
                                                              logits = pred, 
                                                              weights = weights)
        return classification_loss
        
