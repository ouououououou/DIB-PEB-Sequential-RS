
import numpy as np
import tensorflow as tf


epsilon = 1e-9


class Capsule_Component(object):
    ''' Capsule layer.
    Args:
        input: A 4-D tensor.
        vec_len: integer, the length of the output vector of a capsule.
        layer_type: string, one of 'FC' or "CONV", the type of this layer,
            fully connected or convolution, for the future expansion capability
        with_routing: boolean, this capsule is routing with the
                      lower-level layer capsule.

    Returns:
        A 4-D tensor.
    '''
    def __init__(self, bs, num_caps_i, num_caps_j, in_vec_len, out_vec_len, user_vec_len, user_bias, T, name, lam=0.001):
        self.bs = bs
        self.T = T
        self.num_caps_i = num_caps_i
        self.num_caps_j = num_caps_j
        self.in_vec_len = in_vec_len
        self.out_vec_len = out_vec_len
        self.user_vec_len = user_vec_len
        self.user_bias = user_bias
        self.name = name

        bias_initializer = tf.truncated_normal_initializer(
            mean=0.0, stddev=0.01)
        weights_regularizer = tf.contrib.layers.l2_regularizer(lam)

        with tf.variable_scope(self.name + "-capsule-%s" % self.in_vec_len, reuse=None):

            W = tf.get_variable('Weight', shape=(1, self.num_caps_i, self.num_caps_j*self.out_vec_len, self.in_vec_len, 1), dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.1),
                                regularizer=weights_regularizer)
            user_convert_matrix = tf.get_variable('user_convert_matrix', shape=(self.user_vec_len, self.out_vec_len), dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.1),
                                                  regularizer=weights_regularizer)
            item_convert_matrix = tf.get_variable('item_convert_matrix', shape=(self.user_vec_len, self.out_vec_len), dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.1),
                                                  regularizer=weights_regularizer)
            biases = tf.get_variable('bias', shape=(1, 1, self.num_caps_j, self.out_vec_len, 1), initializer=bias_initializer)

    def get_output(self, input, userEmbedding):

        self.input = tf.reshape(input, shape=(self.bs, -1, 1, input.shape[-2].value, 1))

        # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
        # about the reason of using 'batch_size', see issue #21

        capsules = self.routing(self.input, userEmbedding)
        capsules = tf.squeeze(capsules, axis=1)

        return(capsules)

    def routing(self, input, userEmbedding):
        ''' The routing algorithm.

        Args:
            input: A Tensor with [batch_size, num_caps_l=1152, 1, length(u_i)=8, 1]
                   shape, num_caps_l meaning the number of capsule in the layer l.
        Returns:
            A Tensor of shape [batch_size, num_caps_l_plus_1, length(v_j)=16, 1]
            representing the vector output `v_j` in the layer l+1
        Notes:
            u_i represents the vector output of capsule i in the layer l, and
            v_j the vector output of capsule j in the layer l+1.
         '''

        b_IJ = tf.constant(np.zeros([self.bs, input.shape[1].value, self.num_caps_j, 1, 1], dtype=np.float32))

        # W: [1, num_caps_i, num_caps_j * len_v_j, len_u_j, 1]
        with tf.variable_scope(self.name + "-capsule-%s" % self.in_vec_len, reuse=True):
            W = tf.get_variable('Weight')
            biases = tf.get_variable('bias')
            user_convert_matrix = tf.get_variable('user_convert_matrix')

        if userEmbedding is not None:
            user_info = tf.matmul(userEmbedding, user_convert_matrix)
            user_info = tf.reshape(user_info, [self.bs, 1, 1, self.out_vec_len, 1])
            user_info_tailed = tf.tile(user_info, [1, self.num_caps_i, 1, 1, 1])
        #user_info_tailed = self.squash(user_info_tailed)
        # [batch_size, 1, num_caps, vec_len, 1]

        # item_info = tf.matmul(itemEmbedding, item_convert_matrix)
        # item_info = tf.reshape(item_info, [self.bs, 1, 1, self.out_vec_len, 1])
        # item_info_tailed = tf.tile(item_info, [1, self.num_caps_i, 1, 1, 1])
        # item_info_tailed = self.squash(item_info_tailed)

        # Eq.2, calc u_hat
        # Since tf.matmul is a time-consuming op,
        # A better solution is using element-wise multiply, reduce_sum and reshape
        # ops instead. Matmul [a, b] x [b, c] is equal to a series ops as
        # element-wise multiply [a*c, b] * [a*c, b], reduce_sum at axis=1 and
        # reshape to [a, c]
        input = tf.tile(input, [1, 1, self.num_caps_j*self.out_vec_len, 1, 1])
        # assert input.get_shape() == [-1, self.num_caps_i, self.num_caps_j*self.out_vec_len, self.in_vec_len, 1]

        u_hat = tf.reduce_sum(W * input, axis=3, keep_dims=True)
        u_hat = tf.reshape(u_hat, shape=[-1, self.num_caps_i, self.num_caps_j, self.out_vec_len, 1])
        # assert u_hat.get_shape() == [-1, num_caps_i, num_caps_j, self.out_vec_len, 1]

        # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
        u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

        # line 3,for r iterations do
        for r_iter in range(self.T):
            with tf.variable_scope('iter_' + str(r_iter)):
                # line 4:
                # => [batch_size, 1152, 10, 1, 1]
                c_IJ = tf.nn.softmax(b_IJ, dim=2)
                # [self.bs, num_caps_i, num_caps_j, 1, 1]

                # At last iteration, use `u_hat` in order to receive gradients from the following graph
                if r_iter == self.T - 1:
                    # line 5:
                    # weighting u_hat with c_IJ, element-wise in the last two dims
                    # => [batch_size, 1152, 10, 16, 1]
                    s_J = tf.multiply(c_IJ, u_hat)
                    # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                    s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True) + biases
                    # assert s_J.get_shape() == [-1, 1, self.num_caps_j, self.out_vec_len, 1]

                    # line 6:
                    # squash using Eq.1,
                    v_J = self.squash(s_J)
                    # assert v_J.get_shape() == [-1, 1, self.num_caps_j, self.out_vec_len, 1]
                elif r_iter < self.T - 1:  # Inner iterations, do not apply backpropagation
                    s_J = tf.multiply(c_IJ, u_hat_stopped)
                    # [-1, num_caps_i, num_caps_j, self.out_vec_len, 1]
                    s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True) + biases
                    # [-1, 1, num_caps_j, self.out_vec_len, 1] + [1, 1, self.num_caps_j, self.out_vec_len, 1]
                    v_J = self.squash(s_J)

                    # line 7:
                    # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 1152, 10, 16, 1]
                    # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                    # batch_size dim, resulting in [1, 1152, 10, 1, 1]
                    v_J_tiled = tf.tile(v_J, [1, self.num_caps_i, 1, 1, 1])

                    if userEmbedding is not None:
                        u_produce_v = tf.reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keep_dims=True) \
                                      + tf.reduce_sum(u_hat_stopped * user_info_tailed, axis=3, keep_dims=True)
                                      #+ tf.reduce_sum(u_hat_stopped * item_info_tailed, axis=3, keep_dims=True)
                        # + tf.reduce_sum(u_hat_stopped * user_info_tailed, axis=3, keep_dims=True) \
                    else:
                        u_produce_v = tf.reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keep_dims=True)

                    # assert u_produce_v.get_shape() == [-1, self.num_caps_i, self.num_caps_j, 1, 1]

                    b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                    b_IJ += u_produce_v

        return (v_J)

    def squash(self, vector):
        '''Squashing function corresponding to Eq. 1
        Args:
            vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
        Returns:
            A tensor with the same shape as vector but squashed in 'vec_len' dimension.
        '''
        vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
        scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
        vec_squashed = tf.multiply(vector, scalar_factor)  # element-wise
        return (vec_squashed)



