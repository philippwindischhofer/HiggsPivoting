import numpy as np
import tensorflow as tf

from TFEnvironment import TFEnvironment
from PCAWhiteningPreprocessor import PCAWhiteningPreprocessor
from MINEModel import MINEModel

class MINEClassifierEnvironment(TFEnvironment):

    def __init__(self, classifier_model):
        super(MINEClassifierEnvironment, self).__init__()
        self.classifier_model = classifier_model
        self.pre = None

    def build(self, num_inputs, num_nuisances):
        self.pre = PCAWhiteningPreprocessor(num_inputs)
        
        # build the remaining graph
        self.labels_in = tf.placeholder(tf.int32, [None, ], name = 'labels_in')
        self.data_in = tf.placeholder(tf.float32, [None, num_inputs], name = 'data_in')
        self.nuisances_in = tf.placeholder(tf.float32, [None, num_nuisances], name = 'nuisances_in')

        # build a single MINE network for testing purposes
        MI_nuisances = MINEModel(name = "MINE_nuisances", hyperpars = {"num_hidden_layers": 3, "num_units": 30})

        self.MINE_loss, self.MINE_vars = MI_nuisances.MINE_loss(self.data_in, self.nuisances_in)

        self.train_MINE = tf.train.AdamOptimizer(learning_rate = 0.01, beta1 = 0.3, beta2 = 0.5).minimize(self.MINE_loss, var_list = self.MINE_vars)
        
        self.saver = tf.train.Saver()

    def init(self, data_train):
        self.sess.run(tf.global_variables_initializer())

    def test_MINE(self):
        self.build(num_inputs = 1, num_nuisances = 1)
        self.init(data_train = None)

        # prepare some test data
        num_samples = 50000
        x = np.random.normal(loc = 0, scale = 1, size = [num_samples])
        y = np.random.normal(loc = 0, scale = 0.9, size = [num_samples]) 

        x = np.expand_dims(x, axis = 1)
        y = np.expand_dims(y, axis = 1)

        for batch in range(100):
            self.sess.run(self.train_MINE, feed_dict = {self.data_in: x, self.nuisances_in: y})
            lossval = self.sess.run(self.MINE_loss, feed_dict = {self.data_in: x, self.nuisances_in: y})
            print("MI = {}".format(-lossval))

    def predict(self):
        pass

    def load(self):
        pass

    def save(self):
        pass
