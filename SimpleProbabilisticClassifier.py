from BaseModels import ClassifierModel

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.contrib.layers as layers

class SimpleProbabilisticModel(ClassifierModel):
    def __init__(self, name, hyperpars):
        self.name = name
        self.hyperpars = hyperpars

    def build_model(self, in_tensor):
        with tf.variable_scope(self.name):
            lay = in_tensor

            for layer in range(int(float(self.hyperpars["num_hidden_layers"]))):
                lay = layers.relu(lay, int(float(self.hyperpars["num_units"])))

            nc = int(float(self.hyperpars["num_components"]))
            pre_output = layers.linear(lay, 3 * nc)
            
            mu_val = pre_output[:,:nc] # no sign restriction on the mean values
            sigma_val = tf.exp(pre_output[:,nc:2*nc]) # standard deviations need to be positive
            frac_val = tf.nn.softmax(pre_output[:,2*nc:3*nc]) # the mixture fractions need to add up to unity

            # build the GMM that forms the output of the classifier
            dists = [tfp.distributions.Normal(loc = mu_val[:,i], scale = sigma_val[:,i]) for i in range(nc)]
            mixing = tfp.distributions.Categorical(probs = frac_val)
            mixed = tfp.distributions.Mixture(cat = mixing, components = dists)

            sample = mixed.sample(1)
            sample = tf.transpose(sample, [1, 0])

            # produce again a softmax-compatible output (works only for binary classification so far)
            normsample = tf.math.sigmoid(sample)
            outputs = tf.concat([normsample, 1 - normsample], axis = 1)

        these_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self.name)
    
        return outputs, these_vars

    def build_loss(self, pred, labels_one_hot):
        classification_loss = tf.losses.softmax_cross_entropy(onehot_labels = labels_one_hot,
                                                              logits = pred)
        return classification_loss
