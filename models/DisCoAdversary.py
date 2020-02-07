import tensorflow as tf

from models.BaseModels import AdversaryModel

class DisCoAdversary(AdversaryModel):

    def __init__(self, name, hyperpars):
        # don't need to do anything for this simple adversary
        self.name = name
        self.hyperpars = hyperpars # are ignored for now anyways

    def build_loss(self, pred, nuisance, is_training, weights = 1.0, eps = 1e-6, batchnum = 0):
        self.weights_scaled = weights / tf.reduce_sum(weights, axis = 0) * tf.cast(tf.shape(nuisance)[0], tf.float32)
        self.disco = self._distance_corr(pred, nuisance, self.weights_scaled)
        self.dummy = tf.Variable(1.0)
        
        self.disco_loss = self.disco + 0.0 * self.dummy
        self.these_vars = [self.dummy]

        return -self.disco_loss, self.these_vars

    def _distance_corr(self, var_1, var_2, normedweight, power=1):
        """var_1: First variable to decorrelate (eg mass)
        var_2: Second variable to decorrelate (eg classifier output)
        normedweight: Per-example weight. Sum of weights should add up to N (where N is the number of examples)
        power: Exponent used in calculating the distance correlation
        
        va1_1, var_2 and normedweight should all be 1D tf tensors with the same number of entries
        
        Usage: Add to your loss function. total_loss = BCE_loss + lambda * distance_corr
        """

        xx = tf.reshape(var_1, [-1, 1])
        xx = tf.tile(xx, [1, tf.size(var_1)])
        xx = tf.reshape(xx, [tf.size(var_1), tf.size(var_1)])
        
        yy = tf.transpose(xx)
        amat = tf.math.abs(xx-yy)
        
        xx = tf.reshape(var_2, [-1, 1])
        xx = tf.tile(xx, [1, tf.size(var_2)])
        xx = tf.reshape(xx, [tf.size(var_2), tf.size(var_2)])
        
        yy = tf.transpose(xx)
        bmat = tf.math.abs(xx-yy)
        
        amatavg = tf.reduce_mean(amat*normedweight, axis=1)
        bmatavg = tf.reduce_mean(bmat*normedweight, axis=1)
        
        minuend_1 = tf.tile(amatavg, [tf.size(var_1)])
        minuend_1 = tf.reshape(minuend_1, [tf.size(var_1), tf.size(var_1)])
        minuend_2 = tf.transpose(minuend_1)
        Amat = amat-minuend_1-minuend_2+tf.reduce_mean(amatavg*normedweight)

        minuend_1 = tf.tile(bmatavg, [tf.size(var_2)])
        minuend_1 = tf.reshape(minuend_1, [tf.size(var_2), tf.size(var_2)])
        minuend_2 = tf.transpose(minuend_1)
        Bmat = bmat-minuend_1-minuend_2+tf.reduce_mean(bmatavg*normedweight)
        
        ABavg = tf.reduce_mean(Amat*Bmat*normedweight,axis=1)
        AAavg = tf.reduce_mean(Amat*Amat*normedweight,axis=1)
        BBavg = tf.reduce_mean(Bmat*Bmat*normedweight,axis=1)
   
        if power==1:
            dCorr = tf.reduce_mean(ABavg*normedweight)/tf.math.sqrt(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight))
        elif power==2:
            dCorr = (tf.reduce_mean(ABavg*normedweight))**2/(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight))
        else:
            dCorr = (tf.reduce_mean(ABavg*normedweight)/tf.math.sqrt(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight)))**power
  
        return dCorr
