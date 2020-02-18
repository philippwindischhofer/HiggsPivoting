import tensorflow as tf
#import tensorflow_probability as tfp
import tensorflow.contrib.layers as layers

from models.BaseModels import ClassifierModel

class SimpleClassifier(ClassifierModel):
    def __init__(self, name, hyperpars):
        self.name = name
        self.hyperpars = hyperpars

    def build_model(self, in_tensor, is_training):

        with tf.variable_scope(self.name):
            lay = in_tensor

            for layer in range(int(float(self.hyperpars["num_hidden_layers"]))):
                lay = layers.relu(lay, int(float(self.hyperpars["num_units"])), weights_initializer = layers.xavier_initializer(seed = 12345),
                                  weights_regularizer = layers.l2_regularizer(scale = 1.0))
                lay = layers.dropout(lay, keep_prob = 1 - float(self.hyperpars["dropout_rate"]), is_training = is_training)

            pre_output = layers.linear(lay, 1)            
            normsample = tf.math.sigmoid(pre_output)
            outputs = tf.concat([normsample, 1 - normsample], axis = 1)

        these_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self.name)
    
        return outputs, these_vars, pre_output

    def build_loss(self, pred, labels_one_hot, weights = 1.0, batchnum = 0):
        with tf.variable_scope(self.name):
            classification_loss = tf.losses.softmax_cross_entropy(onehot_labels = labels_one_hot,
                                                                  logits = pred,
                                                                  weights = weights)
        return classification_loss
