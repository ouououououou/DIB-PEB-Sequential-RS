import tensorflow as tf

class FM:

    def __init__(self, feature_size, fm_k):

        self.WFM1 = tf.Variable(
            tf.random_uniform([feature_size, 1], -0.1, 0.1), name='fm1')
        self.WFM2 = tf.Variable(
            tf.random_uniform([feature_size, fm_k], -0.1, 0.1), name='fm2')
        self.b = tf.Variable(tf.constant(0.1), name='bias')

    def get_output(self, feature_input):
        one = tf.matmul(feature_input, self.WFM1)

        inte1 = tf.matmul(feature_input, self.WFM2)
        inte2 = tf.matmul(tf.square(feature_input), tf.square(self.WFM2))

        inter = (tf.square(inte1) - inte2) * 0.5
        inter = tf.reduce_sum(inter, 1, keep_dims=True)

        self.predictions = one + inter + self.b

        return self.predictions
