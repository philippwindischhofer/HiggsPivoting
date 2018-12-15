import tensorflow as tf
import numpy as np

class TrainingEnvironment:
    
    def __init__(self, classifier_model, training_pars):
        self.classifier_model = classifier_model
        self.training_pars = training_pars
        
    # builds the training graph
    def build(self):
        self.labels_in = tf.placeholder(tf.int32, [None, ], name = 'labels_in')
        self.data_in = tf.placeholder(tf.float32, [None, ], name = 'data_in')

        self.classifier_out, self.classifier_vars = self.classifier_model.classifier(self.data_in)

        self.labels_one_hot = tf.one_hot(self.labels_in, depth = 2)
        self.classification_loss = tf.losses.softmax_cross_entropy(onehot_labels = self.labels_one_hot, 
                                                                   logits = self.classifier_output)

        self.train_op = tf.train.AdamOptimizer(learning_rate = 0.01, 
                                               beta1 = 0.3, 
                                               beta2 = 0.5).minimize(self.classification_loss, var_list = self.classifier_vars)
        

    def train(self, number_epochs, data_sig, data_bkg):
        # prepare the training dataset
        self.training_data = np.concatenate([data_sig, data_bkg], axis = 0)
        self.training_labels = np.concatenate([np.ones(len(data_sig)), np.zeros(len(data_bkg))], axis = 0)

        # start the tensorflow session
        self.config = tf.ConfigProto(intra_op_parallelism_threads = 8, 
                                     inter_op_parallelism_threads = 8,
                                     allow_soft_placement = True, 
                                     device_count = {'CPU': 8})
        self.sess = tf.InteractiveSession(config = self.config)

        # initialize the graph
        self.sess.run(tf.global_variables_initializer())

        for epoch in range(number_epochs):
            # sample from signal and background
            inds = np.random.choice(len(self.training_data), self.training_pars["batch_size"])
            data_batch = self.training_data[inds]
            labels_batch = self.training_labels[inds]

            self.sess.run(self.train_op, feed_dict = {data_in: data_batch, labels_in: labels_batch})
            loss = self.sess.run(self.classification_loss, feed_dict = {data_in: data_batch, labels_in: labels_batch})

            print("loss = {}".format(loss))
