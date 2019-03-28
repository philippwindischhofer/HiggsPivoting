from abc import ABC, abstractmethod
import tensorflow as tf

class TFEnvironment(ABC):
    
    def __init__(self, config = tf.ConfigProto(intra_op_parallelism_threads = 1, 
                                               inter_op_parallelism_threads = 1,
                                               allow_soft_placement = True, 
                                               device_count = {'CPU': 1})):
        print("starting TensorFlow session ...")
        # start the tensorflow session
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session(config = config)
        print("done!")

    # builds the computational graph required by this environment
    @abstractmethod
    def build(self):
        pass

    # uses the (possibly trained?) model to make a forward pass
    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass
