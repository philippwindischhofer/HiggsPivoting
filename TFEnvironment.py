import tensorflow as tf
import numpy as np

class TFEnvironment:
    
    def __init__(self, classifier_model, training_pars, num_inputs):
        self.classifier_model = classifier_model
        self.training_pars = training_pars
        self.num_inputs = num_inputs

        # start the tensorflow session
        self.config = tf.ConfigProto(intra_op_parallelism_threads = 8, 
                                     inter_op_parallelism_threads = 8,
                                     allow_soft_placement = True, 
                                     device_count = {'CPU': 8})
        self.sess = tf.InteractiveSession(config = self.config)
        
    # builds the training graph
    def build(self):
        self.labels_in = tf.placeholder(tf.int32, [None, ], name = 'labels_in')
        self.data_in = tf.placeholder(tf.float32, [None, self.num_inputs], name = 'data_in')

        self.classifier_out, self.classifier_vars = self.classifier_model.classifier(self.data_in)

        self.labels_one_hot = tf.one_hot(self.labels_in, depth = 2)
        self.classification_loss = tf.losses.softmax_cross_entropy(onehot_labels = self.labels_one_hot, 
                                                                   logits = self.classifier_out)

        self.train_op = tf.train.AdamOptimizer(learning_rate = 0.01, 
                                               beta1 = 0.9, 
                                               beta2 = 0.999).minimize(self.classification_loss, var_list = self.classifier_vars)        

        self.saver = tf.train.Saver()

    def train(self, number_epochs, data_sig, data_bkg):
        # prepare the training dataset
        self.training_data = np.concatenate([data_sig, data_bkg], axis = 0)
        self.training_labels = np.concatenate([np.ones(len(data_sig)), np.zeros(len(data_bkg))], axis = 0)

        # initialize the graph
        self.sess.run(tf.global_variables_initializer())

        for epoch in range(number_epochs):
            # sample from signal and background
            inds = np.random.choice(len(self.training_data), self.training_pars["batch_size"])
            data_batch = self.training_data[inds]
            labels_batch = self.training_labels[inds]

            self.sess.run(self.train_op, feed_dict = {self.data_in: data_batch, self.labels_in: labels_batch})
            loss = self.sess.run(self.classification_loss, feed_dict = {self.data_in: data_batch, self.labels_in: labels_batch})

            print("loss = {}".format(loss))

    def save(self, file_path):
        self.saver.save(self.sess, file_path)

    def load(self, file_path):
        self.saver.restore(self.sess, file_path)

    def predict(self, data_test):
        retval = self.sess.run(self.classifier_out, feed_dict = {self.data_in: data_test})
        return retval
