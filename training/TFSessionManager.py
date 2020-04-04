import tensorflow as tf

class TFSessionManager:

    def __init__(self, config = tf.ConfigProto(intra_op_parallelism_threads = 8, 
                                               inter_op_parallelism_threads = 8,
                                               allow_soft_placement = True, 
                                               device_count = {'CPU': 1})):
        print("starting TensorFlow session")
        self.sess = tf.Session(config = config)

        print("done!")

    def getSession(self):
        return self.sess

