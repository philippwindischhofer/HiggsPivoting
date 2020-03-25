import tensorflow as tf
from models.MINEAdversary import MINEAdversary

class NeuralMIEstimator:

    def __init__(self, name):
        self.name = name

    def add_to_graph(self, graph, width_X, width_Y):        
        hyperpars = {"num_hidden_layers": 6, "num_units": 30, "dropout_rate": 0.0}
        self.MINE = MINEAdversary(name = self.name + "_MINE", hyperpars = hyperpars)

        with graph.as_default():
            self.X_in = tf.placeholder(tf.float32, [None, width_X], name = 'X_in')
            self.Y_in = tf.placeholder(tf.float32, [None, width_Y], name = 'Y_in')
            self.is_training_in = tf.placeholder(tf.bool, name = 'is_training')
            self.weights_in = tf.placeholder(tf.float32, [None, ], name = 'weights_in')

            self.MINE_loss, self.MINE_vars = self.MINE.build_loss(self.X_in, self.Y_in, self.is_training_in, self.weights_in)
            self.update_estimator = tf.train.AdamOptimizer(learning_rate = 2e-4,
                                                           beta1 = 0.9,
                                                           beta2 = 0.999,
                                                           epsilon = 1e-8).minimize(self.MINE_loss, var_list = self.MINE_vars)

    def estimate(self, sess, X_in, Y_in, weights):
        # first, update the MINE estimator
        print("/ / / / / / / / / / / / / /")

        for cur_step in range(100):
            sess.run(self.update_estimator, feed_dict = {self.X_in: X_in, self.Y_in: Y_in, self.weights_in: weights, self.is_training_in: True})
            cur_loss = sess.run(self.MINE_loss, feed_dict = {self.X_in: X_in, self.Y_in: Y_in, self.weights_in: weights, self.is_training_in: True})
            print(" MINE loss = {}".format(cur_loss))

        print("/ / / / / / / / / / / / / /")

        # return the best estimate for MI at the end of the optimisation
        return -cur_loss
