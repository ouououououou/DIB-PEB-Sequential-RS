import tensorflow as tf

class MLP:

    def __init__(self, dims, dropout_keep, lam=0.001):

        self.numLayer = len(dims)
        self.dropout_keep = dropout_keep

        bias_initializer = tf.truncated_normal_initializer(
            mean=0.0, stddev=0.01)
        weights_regularizer = tf.contrib.layers.l2_regularizer(lam)

        with tf.variable_scope("MLP", reuse=None):
            for layerNum in range(self.numLayer-1):
                tf.get_variable(name="weight-%s" % layerNum, shape=[dims[layerNum], dims[layerNum+1]], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer=weights_regularizer
                                )
                tf.get_variable(name="bias-%s" % layerNum,
                                shape=[dims[layerNum + 1]],
                                dtype=tf.float32,
                                initializer=bias_initializer,
                                )

    def get_output(self, feature_input):

        with tf.variable_scope("MLP", reuse=True):
            output = feature_input
            for layerNum in range(self.numLayer-1):
                weight = tf.get_variable(name="weight-%s" % layerNum)
                bias = tf.get_variable(name="bias-%s" % layerNum)
                output = tf.nn.dropout(
                                tf.nn.relu(
                                    tf.nn.xw_plus_b(output, weight, bias),
                                name="score-%s" % (layerNum + 1)),
                         self.dropout_keep)

        return output