import os
import tensorflow as tf
import numpy as np

from TFEnvironment import TFEnvironment
from PCAWhiteningPreprocessor import PCAWhiteningPreprocessor

class SimpleClassifierEnvironment(TFEnvironment):

    def __init__(self, classifier_model):
        super(SimpleClassifierEnvironment, self).__init__()
        self.classifier_model = classifier_model
        self.pre = None
        
    # builds the graph
    def build(self, num_inputs):
        # build the preprocessor
        self.pre = PCAWhiteningPreprocessor(num_inputs)

        # build the remaining graph
        self.labels_in = tf.placeholder(tf.int32, [None, ], name = 'labels_in')
        self.data_in = tf.placeholder(tf.float32, [None, num_inputs], name = 'data_in')

        self.classifier_out, self.classifier_vars = self.classifier_model.classifier(self.data_in)

        self.labels_one_hot = tf.one_hot(self.labels_in, depth = 2)
        self.classification_loss = tf.losses.softmax_cross_entropy(onehot_labels = self.labels_one_hot, 
                                                                   logits = self.classifier_out)

        self.train_op = tf.train.AdamOptimizer(learning_rate = 0.01, 
                                               beta1 = 0.9, 
                                               beta2 = 0.999).minimize(self.classification_loss, var_list = self.classifier_vars)        

        self.saver = tf.train.Saver()

    def init(self, data_train):
        self.pre.setup(data_train)
        self.sess.run(tf.global_variables_initializer())

    def train_step(self, data_step, labels_step):
        data_pre = self.pre.process(data_step)
        self.sess.run(self.train_op, feed_dict = {self.data_in: data_pre, self.labels_in: labels_step})

    def loss(self, data, labels):
        data_pre = self.pre.process(data)
        loss = self.sess.run(self.classification_loss, feed_dict = {self.data_in: data_pre, self.labels_in: labels})
        return loss

    def dump_loss_information(self, data, labels):
        loss = self.loss(data = data, labels = labels)
        print("loss = {}".format(loss))

    def predict(self, data):
        data_pre = self.pre.process(data)
        retval = self.sess.run(self.classifier_out, feed_dict = {self.data_in: data_pre})
        return retval

    def save(self, file_path):
        self.saver.save(self.sess, file_path)
        self.pre.save(os.path.join(os.path.dirname(file_path), 'pre.pkl'))

    def load(self, file_path):
        self.saver.restore(self.sess, file_path)
        self.pre = PCAWhiteningPreprocessor.from_file(os.path.join(os.path.dirname(file_path), 'pre.pkl'))

