
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class EM_Capsule_Component(object):

    def __init__(self, bs, wordVec_len, num_caps_b, num_caps_c, num_caps_d, pose_len, user_vec_len, user_bias, T, name, filter_height_1, filter_height_2, filter_height_3, epsilon=1e-9, lam=5e-04, layer_num=1, routing_type='em'):
        self.bs = bs
        self.T = T
        self.wordVec_len = wordVec_len
        self.num_caps_b = num_caps_b
        self.num_caps_c = num_caps_c
        self.num_caps_d = num_caps_d
        self.filter_height_1 = filter_height_1
        self.filter_height_2 = filter_height_2
        self.filter_height_3 = filter_height_3
        self.pose_len = pose_len
        self.user_vec_len = user_vec_len
        self.user_bias = user_bias
        self.name = name
        self.lam = lam
        self.epsilon = epsilon
        self.layer_num = layer_num
        self.routing_type = routing_type

    def get_output(self, input, is_train: bool, userEmbedding, at_b=False, at_c=False, share_user_para_b=False, share_user_para_c=False, user_as_miu_bias=False, user_reg_cost=False):

        return self.get_output_em(input, bool, userEmbedding, at_b, at_c, share_user_para_b, share_user_para_c, user_as_miu_bias, user_reg_cost)
        # if self.routing_type == 'em':
        #
        # else:
        #     return self.get_output_dynamic(input, bool, userEmbedding, at_b, at_c, share_user_para)

    def get_output_em(self, input, is_train: bool, userEmbedding, at_b, at_c, share_user_para_b, share_user_para_c, user_as_miu_bias, user_reg_cost):
        # input shape: [bs, height, width, 1]

        # xavier initialization is necessary here to provide higher stability
        # initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        # instead of initializing bias with constant 0, a truncated normal initializer is exploited here for higher stability
        bias_initializer = tf.truncated_normal_initializer(
            mean=0.0, stddev=0.01)  # tf.constant_initializer(0.0)
        # The paper didnot mention any regularization, a common l2 regularizer to weights is added here
        weights_regularizer = tf.contrib.layers.l2_regularizer(self.lam)

        # weights_initializer=initializer,
        with slim.arg_scope([slim.conv2d], trainable=True, biases_initializer=bias_initializer,
                            weights_regularizer=weights_regularizer):
            with tf.variable_scope(self.name) as outScope:
                with tf.variable_scope('primary_caps') as scope:
                    if at_b:
                        real_num_caps_b = self.num_caps_b + 1
                    else:
                        real_num_caps_b = self.num_caps_b
                    pose = slim.conv2d(input, num_outputs=self.num_caps_b * self.pose_len,
                                       kernel_size=[1, self.wordVec_len], stride=1, padding='VALID', scope=scope,
                                       activation_fn=None)
                    activation = slim.conv2d(input, num_outputs=self.num_caps_b, kernel_size=[
                        1, self.wordVec_len], stride=1, padding='VALID', scope='primary_caps/activation',
                                             activation_fn=tf.nn.sigmoid)

                    data_height = int(input.get_shape()[1]) - self.filter_height_1 + 1

                    if at_b:
                        if share_user_para_b:
                            user_trans_matrix = slim.variable(name='user_trans_matrix',
                                                                shape=(self.user_vec_len, self.pose_len))
                            user_trans_bias = slim.variable(name='user_trans_bias',
                                                              shape=(self.user_vec_len, 1))
                            user_pose = tf.matmul(userEmbedding, user_trans_matrix)  # [bs, data_height * pose_len]
                            user_pose = tf.reshape(user_pose, shape=[self.bs, 1, 1, self.pose_len])
                            user_pose = tf.tile(user_pose, [1, data_height, 1, 1])
                            # user_pose = tf.tile(user_pose, [1, data_height, 1, 1])
                            user_activ = tf.nn.sigmoid(tf.matmul(userEmbedding, user_trans_bias))  # [bs, data_height]
                            user_activ = tf.reshape(user_activ, shape=[self.bs, 1, 1, 1])
                            user_activ = tf.tile(user_activ, [1, data_height, 1, 1])

                        else:
                            user_trans_matrix = slim.variable(name='user_trans_matrix', shape=(self.user_vec_len, data_height * self.pose_len))
                            user_trans_bias = slim.variable(name='user_trans_bias', shape=(self.user_vec_len, data_height))
                            user_pose = tf.matmul(userEmbedding, user_trans_matrix)  # [bs, data_height * pose_len]
                            user_pose = tf.reshape(user_pose, shape=[self.bs, data_height, 1, self.pose_len])
                            # user_pose = tf.tile(user_pose, [1, data_height, 1, 1])
                            user_activ = tf.nn.sigmoid(tf.matmul(userEmbedding, user_trans_bias))  # [bs, data_height]
                            user_activ = tf.reshape(user_activ, shape=[self.bs, data_height, 1, 1])

                        pose = tf.reshape(pose, shape=[self.bs, data_height, self.num_caps_b, self.pose_len])
                        pose = tf.concat([pose, user_pose], axis=2)
                        activation = tf.reshape(activation, shape=[self.bs, data_height, self.num_caps_b, 1])
                        activation = tf.concat([activation, user_activ], axis=2)
                    else:
                        pose = tf.reshape(pose, shape=[self.bs, data_height, real_num_caps_b, self.pose_len])
                        activation = tf.reshape(activation, shape=[self.bs, data_height, real_num_caps_b, 1])

                    output = tf.concat([pose, activation], axis=3)
                    output = tf.reshape(output, shape=[self.bs, data_height, real_num_caps_b, -1])
                    assert output.get_shape() == [self.bs, data_height, real_num_caps_b, self.pose_len + 1]

                with tf.variable_scope('conv_caps1') as scope:
                    if at_c:
                        real_num_caps_c = self.num_caps_c + 1
                    else:
                        real_num_caps_c = self.num_caps_c
                    data_height = data_height - self.filter_height_2 + 1
                    output = self.kernel_tile(input=output, filter_height=self.filter_height_2,
                                              filter_width=real_num_caps_b, stride=1)
                    # [batch_size, data_height, 1,  filter_height_2 * num_caps_c, pose_len + 1]

                    output = tf.reshape(output, shape=[self.bs * data_height, self.filter_height_2 * real_num_caps_b, self.pose_len + 1])

                    activation = tf.reshape(output[:, :, self.pose_len], shape=[self.bs * data_height, self.filter_height_2 * real_num_caps_b, 1])

                    with tf.variable_scope('v') as scope:
                        votes = self.mat_transform(output[:, :, :self.pose_len], self.num_caps_c, weights_regularizer, tag=True)
                        # [batch_size, caps_num_c, caps_num_d, pose_len]

                    with tf.variable_scope('routing') as scope:
                        miu, activation = self.em_routing(votes, activation, self.num_caps_c, weights_regularizer, userEmbedding, user_as_miu_bias, user_reg_cost)
                        # miu = [batch_size, 1, caps_num_d, pose_len]
                        # activation_out = [batch_size, caps_num_d]

                        if at_c:
                            if share_user_para_c:
                                user_trans_matrix = slim.variable(name='user_trans_matrix',
                                                                    shape=(self.user_vec_len, self.pose_len))
                                user_trans_bias = slim.variable(name='user_trans_bias',
                                                                  shape=(self.user_vec_len, 1))
                                user_pose = tf.matmul(userEmbedding, user_trans_matrix)  # [bs, data_height * pose_len]
                                user_pose = tf.reshape(user_pose, shape=[self.bs, 1, 1, self.pose_len])
                                user_pose = tf.tile(user_pose, [1, data_height, 1, 1])
                                # user_pose = tf.tile(user_pose, [1, data_height, 1, 1])
                                user_activ = tf.nn.sigmoid(
                                    tf.matmul(userEmbedding, user_trans_bias))  # [bs, data_height]
                                user_activ = tf.reshape(user_activ, shape=[self.bs, 1, 1, 1])
                                user_activ = tf.tile(user_activ, [1, data_height, 1, 1])

                            else:
                                user_trans_matrix = slim.variable(name='user_trans_matrix',
                                                                    shape=(
                                                                    self.user_vec_len, data_height * self.pose_len))
                                user_trans_bias = slim.variable(name='user_trans_bias',
                                                                  shape=(self.user_vec_len, data_height))

                                user_pose = tf.matmul(userEmbedding, user_trans_matrix)  # [bs, data_height * pose_len]
                                user_pose = tf.reshape(user_pose, shape=[self.bs, data_height, 1, self.pose_len])
                                # user_pose = tf.tile(user_pose, [1, data_height, 1, 1])
                                user_activ = tf.nn.sigmoid(
                                    tf.matmul(userEmbedding, user_trans_bias))  # [bs, data_height]
                                user_activ = tf.reshape(user_activ, shape=[self.bs, data_height, 1, 1])

                            pose = tf.reshape(miu, shape=[self.bs, data_height, self.num_caps_c, self.pose_len])
                            pose = tf.concat([pose, user_pose], axis=2)
                            activation = tf.reshape(activation, shape=[self.bs, data_height, self.num_caps_c, 1])
                            activation = tf.concat([activation, user_activ], axis=2)

                        else:
                            pose = tf.reshape(miu, shape=[self.bs, data_height, real_num_caps_c, self.pose_len])
                            activation = tf.reshape(activation, shape=[self.bs, data_height, real_num_caps_c, 1])

                if self.layer_num > 1:
                    output = tf.reshape(tf.concat([pose, activation], axis=3),
                                        [self.bs, data_height, real_num_caps_c, -1])

                    with tf.variable_scope('conv_caps2') as scope:

                        output = self.kernel_tile(input=output, filter_height=self.filter_height_3,
                                                  filter_width=real_num_caps_c, stride=1)
                        data_height = data_height - self.filter_height_3 + 1

                        output = tf.reshape(output,
                                            shape=[self.bs * data_height, self.filter_height_3 * real_num_caps_c,
                                                   self.pose_len + 1])
                        activation = tf.reshape(output[:, :, self.pose_len],
                                                shape=[self.bs * data_height, self.filter_height_3 * real_num_caps_c,
                                                       1])

                        with tf.variable_scope('v') as scope:
                            votes = self.mat_transform(output[:, :, :self.pose_len], self.num_caps_d,
                                                       weights_regularizer, tag=True)
                            # [batch_size, caps_num_c, caps_num_d, pose_len]

                        with tf.variable_scope('routing') as scope:
                            miu, activation = self.em_routing(votes, activation, self.num_caps_d, weights_regularizer, userEmbedding, user_as_miu_bias, user_reg_cost)
                            # miu = [batch_size, 1, caps_num_e, pose_len]
                            # activation_out = [batch_size, caps_num_d]

                            pose = tf.reshape(miu, shape=[self.bs, data_height, self.num_caps_d, self.pose_len])
                            activation = tf.reshape(activation, shape=[self.bs, data_height, self.num_caps_d, 1])

        return pose, activation

    def get_output_dynamic(self, input, is_train: bool, userEmbedding, at_b, at_c, share_user_para):
        # input shape: [bs, height, width, 1]

        # xavier initialization is necessary here to provide higher stability
        # initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        # instead of initializing bias with constant 0, a truncated normal initializer is exploited here for higher stability
        bias_initializer = tf.truncated_normal_initializer(
            mean=0.0, stddev=0.01)  # tf.constant_initializer(0.0)
        # The paper didnot mention any regularization, a common l2 regularizer to weights is added here
        weights_regularizer = tf.contrib.layers.l2_regularizer(self.lam)

        # weights_initializer=initializer,
        with slim.arg_scope([slim.conv2d], trainable=True, biases_initializer=bias_initializer,
                            weights_regularizer=weights_regularizer):
            with tf.variable_scope(self.name) as outScope:
                with tf.variable_scope('primary_caps') as scope:
                    pose = slim.conv2d(input, num_outputs=self.num_caps_b * self.pose_len,
                                       kernel_size=[1, self.wordVec_len], stride=1, padding='VALID', scope=scope,
                                       activation_fn=None)

                    data_height = int(input.get_shape()[1]) - self.filter_height_1 + 1

                    pose = tf.reshape(pose, shape=[self.bs, data_height, self.num_caps_b, self.pose_len])

                    output = pose
                    assert output.get_shape() == [self.bs, data_height, self.num_caps_b, self.pose_len]

                with tf.variable_scope('conv_caps1') as scope:
                    data_height = data_height - self.filter_height_2 + 1
                    output = self.kernel_tile(input=output, filter_height=self.filter_height_2,
                                              filter_width=self.num_caps_b, stride=1)
                    # [batch_size, data_height, 1,  filter_height_2 * num_caps_d, pose_len + 1]

                    output = tf.reshape(output, shape=[self.bs * data_height, self.filter_height_2 * self.num_caps_b,
                                                       self.pose_len])

                    with tf.variable_scope('v') as scope:
                        votes = self.mat_transform(output[:, :, :self.pose_len], self.num_caps_c, weights_regularizer,
                                                   tag=True)
                        # [batch_size, caps_num_c, caps_num_d, pose_len]

                    with tf.variable_scope('routing') as scope:
                        pose = self.dynamic_routing(votes, weights_regularizer, bias_initializer)
                        # miu = [batch_size, 1, caps_num_d, pose_len]
                        # activation_out = [batch_size, caps_num_d]

                        pose = tf.reshape(pose, shape=[self.bs, data_height, self.num_caps_c, self.pose_len])

                if self.layer_num > 1:
                    output = pose

                    with tf.variable_scope('conv_caps2') as scope:
                        output = self.kernel_tile(input=output, filter_height=self.filter_height_3,
                                                  filter_width=self.num_caps_c, stride=1)
                        data_height = data_height - self.filter_height_3 + 1

                        output = tf.reshape(output, shape=[self.bs * data_height, self.filter_height_3 * self.num_caps_c, self.pose_len])

                        with tf.variable_scope('v') as scope:
                            votes = self.mat_transform(output[:, :, :self.pose_len], self.num_caps_d, weights_regularizer, tag=True)
                            # [batch_size, caps_num_i, caps_num_e, pose_len]

                        with tf.variable_scope('routing') as scope:
                            pose = self.dynamic_routing(votes, weights_regularizer, bias_initializer)
                            # miu = [batch_size, 1, caps_num_e, pose_len]
                            # activation_out = [batch_size, caps_num_d]

                            pose = tf.reshape(pose, shape=[self.bs, data_height, self.num_caps_d, self.pose_len])

        return pose, None

    def get_output_shape(self, input_height, filter_height):

        conv1_height = input_height - filter_height + 1
        primCap1_height = conv1_height - self.filter_height_1 + 1
        convCap1_height = primCap1_height - self.filter_height_2 + 1
        if self.layer_num == 1:
            output_size = convCap1_height * self.num_caps_c * self.pose_len
        else:
            convCap2_height = convCap1_height - self.filter_height_3 + 1
            output_size = convCap2_height * self.num_caps_d * self.pose_len

        return output_size

    # input should be a tensor with size as [batch_size, caps_num_i, pose_len]
    def mat_transform(self, input, caps_num_j, regularizer, tag=False):
        batch_size = int(input.get_shape()[0])
        caps_num_i = int(input.get_shape()[1])
        sqrt_pose_len = int(np.sqrt(self.pose_len))
        output = tf.reshape(input, shape=[batch_size, caps_num_i, 1, sqrt_pose_len, sqrt_pose_len])
        # the output of capsule is miu, the mean of a Gaussian, and activation, the sum of probabilities
        # it has no relationship with the absolute values of w and votes
        # using weights with bigger stddev helps numerical stability
        w = slim.variable('w', shape=[1, caps_num_i, caps_num_j, sqrt_pose_len, sqrt_pose_len], dtype=tf.float32,
                          initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0),
                          regularizer=regularizer)

        w = tf.tile(w, [batch_size, 1, 1, 1, 1])
        output = tf.tile(output, [1, 1, caps_num_j, 1, 1])
        votes = tf.reshape(tf.matmul(output, w), [batch_size, caps_num_i, caps_num_j, self.pose_len])

        return votes

    def dynamic_routing(self, votes, regularizer, bias_initializer, tag=False, userEmbedding=None):

        # votes = [batch_size, caps_num_i, caps_num_c, pose_len]
        # caps_num_i = self.filter_height_2 * self.filter_width_2
        batch_size = int(votes.get_shape()[0])
        num_caps_i = int(votes.get_shape()[1])
        caps_num_c = int(votes.get_shape()[2])
        biases = tf.get_variable('bias', shape=(1, 1, caps_num_c, self.pose_len, 1), initializer=bias_initializer)
        user_convert_matrix = tf.get_variable('user_convert_matrix', shape=(self.user_vec_len, self.pose_len),
                                              dtype=tf.float32,
                                              initializer=tf.random_normal_initializer(stddev=0.1),
                                              regularizer=regularizer)


        b_IJ = tf.constant(np.zeros([batch_size, num_caps_i, caps_num_c, 1, 1], dtype=np.float32))

        if userEmbedding is not None:
            user_info = tf.matmul(userEmbedding, user_convert_matrix)
            user_info = tf.reshape(user_info, [self.bs, 1, 1, self.pose_len, 1])
            user_info_tailed = tf.tile(user_info, [int(batch_size / self.bs), num_caps_i, 1, 1, 1])

        u_hat = votes

        u_hat = tf.reshape(u_hat, shape=[-1, num_caps_i, caps_num_c, self.pose_len, 1])
        # assert u_hat.get_shape() == [-1, self.num_caps_i, self.num_caps_j, self.out_vec_len, 1]

        # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
        u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

        # line 3,for r iterations do
        for r_iter in range(self.T):
            with tf.variable_scope('iter_' + str(r_iter)):
                # line 4:
                # => [batch_size, 1152, 10, 1, 1]
                c_IJ = tf.nn.softmax(b_IJ, dim=2)

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
                    s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True) + biases
                    v_J = self.squash(s_J)

                    # line 7:
                    # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 1152, 10, 16, 1]
                    # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                    # batch_size dim, resulting in [1, 1152, 10, 1, 1]
                    v_J_tiled = tf.tile(v_J, [1, num_caps_i, 1, 1, 1])

                    if userEmbedding is not None:
                        u_produce_v = tf.reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keep_dims=True) \
                                      + tf.reduce_sum(u_hat_stopped * user_info_tailed, axis=3, keep_dims=True)
                        # + tf.reduce_sum(u_hat_stopped * item_info_tailed, axis=3, keep_dims=True)
                        # + tf.reduce_sum(u_hat_stopped * user_info_tailed, axis=3, keep_dims=True) \
                    else:
                        u_produce_v = tf.reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keep_dims=True)

                    # assert u_produce_v.get_shape() == [-1, self.num_caps_i, self.num_caps_j, 1, 1]

                    b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                    b_IJ += u_produce_v

        return (v_J)  # [batch_size, num_caps_c, pose_len, 1]


    def em_routing(self, votes, activation, caps_num_c, regularizer, userEmbedding, user_as_miu_bias, user_reg_cost):
        # batch_size = self.bs * data_height,
        # caps_num_i = self.filter_height_2 * self.filter_width_2
        # vote = [batch_size, caps_num_i, caps_num_c, pose_len]
        # activ = [batch_size, caps_num_i, 1]

        # return: miu = [batch_size, 1, caps_num_c, pose_len]
        #         activation_out = [batch_size, caps_num_c]

        batch_size = int(votes.get_shape()[0])
        caps_num_i = int(activation.get_shape()[1])
        pose_len = int(votes.get_shape()[-1])

        data_height = batch_size // self.bs

        sigma_square = []
        miu = []
        activation_out = []
        beta_v = slim.variable('beta_v', shape=[caps_num_c, pose_len], dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0),
                               # tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                               regularizer=regularizer)
        beta_a = slim.variable('beta_a', shape=[caps_num_c], dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0),
                               # tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                               regularizer=regularizer)
        if user_as_miu_bias or user_reg_cost:
            user_trans_matrices = []
            for i in range(data_height):
                user_trans_matrix = slim.variable('user_trans_matrix-' + str(i),
                                              shape=[self.user_vec_len, caps_num_c * self.pose_len], dtype=tf.float32,
                                              # tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                              regularizer=regularizer)
                user_trans_matrices.append(user_trans_matrix)

        votes_in = votes
        activation_in = activation

        for iters in range(self.T):
            # if iters == self.T-1:

            # e-step
            if iters == 0:
                r = tf.constant(np.ones([batch_size, caps_num_i, caps_num_c], dtype=np.float32) / caps_num_c)
            else:
                # log and exp here provide higher numerical stability especially for bigger number of iterations
                log_p_c_h = -tf.log(tf.sqrt(sigma_square)) - \
                            (tf.square(votes_in - miu) / (2 * sigma_square))
                log_p_c_h = log_p_c_h - \
                            (tf.reduce_max(log_p_c_h, axis=[2, 3], keep_dims=True) - tf.log(10.0))
                p_c = tf.exp(tf.reduce_sum(log_p_c_h, axis=3))

                ap = p_c * tf.reshape(activation_out, shape=[batch_size, 1, caps_num_c])

                r = ap / (tf.reduce_sum(ap, axis=2, keep_dims=True) + self.epsilon)

            # m-step
            r = r * activation_in
            # activ = [batch_size, caps_num_i, 1]
            # r = [batch_size, caps_num_i, caps_num_c]

            r = r / (tf.reduce_sum(r, axis=2, keep_dims=True) + self.epsilon)
            # r = [batch_size, caps_num_i, caps_num_c]

            r_sum = tf.reduce_sum(r, axis=1, keep_dims=True)
            # r_sum = [batch_size, 1, caps_num_c]

            r1 = tf.reshape(r / (r_sum + self.epsilon), shape=[batch_size, caps_num_i, caps_num_c, 1])
            # r_1 = [batch_size, caps_num_i, caps_num_c, 1]

            miu = tf.reduce_sum(votes_in * r1, axis=1, keep_dims=True)
            # miu = [batch_size, 1, caps_num_c, pose_len]
            if user_as_miu_bias:
                user_biases = []
                for i in range(data_height):
                    miu_u_bias = tf.reshape(tf.matmul(userEmbedding, user_trans_matrices[i]),
                                        [self.bs, 1, caps_num_c, self.pose_len])
                    user_biases.append(miu_u_bias)
                    # [bs, caps_num_c * self.pose_len] -> [bs, 1, caps_num_c, pose_len]
                user_biases_concat = tf.reshape(tf.concat([user_biases], axis=0), shape=[batch_size, 1, caps_num_c, self.pose_len])  # [bs * data_height, 1, caps_num_c, pose_len]
                miu = miu + user_biases_concat

            sigma_square = tf.reduce_sum(tf.square(votes_in - miu) * r1, axis=1, keep_dims=True) + self.epsilon
            # sigma_square = [batch_size, 1, caps_num_c, pose_len]

            if user_reg_cost:
                transed_users = []
                for i in range(data_height):
                    transed_user = tf.reshape(tf.matmul(userEmbedding, user_trans_matrices[i]),
                                              [self.bs, 1, caps_num_c, self.pose_len])
                    # [bs, caps_num_c * self.pose_len] -> [bs, 1, caps_num_c, pose_len]
                    transed_users.append(transed_user)
                transed_users_concat = tf.reshape(tf.concat([transed_users], axis=0), shape=[batch_size, 1, caps_num_c, self.pose_len])
                miu_minus_user_square_sum = tf.reduce_sum(tf.square(miu - transed_users_concat), axis=3,
                                                              keep_dims=False)  # [bs * data_height, 1, caps_num_c]
                miu_minus_user_square_sum = tf.reduce_sum(miu_minus_user_square_sum, axis=1,
                                                              keep_dims=False)  # [bs * data_height, caps_num_c]

            if iters == self.T - 1:
                r_sum = tf.reshape(r_sum, [batch_size, caps_num_c, 1])
                # r_sum = [batch_size, caps_num_c, 1]

                cost_h = (beta_v + tf.log(tf.sqrt(tf.reshape(sigma_square, shape=[batch_size, caps_num_c, pose_len])))) * r_sum
                # beta_v = [caps_num_c, pose_len]
                # cost_h = [batch_size, caps_num_c, pose_len]
                if user_reg_cost:
                    activation_out = tf.nn.softmax(0.01 * (beta_a - tf.reduce_sum(cost_h, axis=2) - miu_minus_user_square_sum))
                else:
                    activation_out = tf.nn.softmax(
                        0.01 * (beta_a - tf.reduce_sum(cost_h, axis=2)))
                # beta_a = [caps_num_c]
                # activation_out = [batch_size, caps_num_c]

            else:
                activation_out = tf.nn.softmax(r_sum)
                # activation_out = [batch_size, 1, caps_num_c]

        # miu = [batch_size, 1, caps_num_c, pose_len]
        # activation_out = [batch_size, caps_num_c]
        return miu, activation_out


    def kernel_tile(self, input, filter_height, filter_width, stride=1):
        # input: [batch_size, height, width, pose_len + 1]

        input_shape = input.get_shape()
        tile_filter = np.zeros(shape=[filter_height, filter_width, input_shape[3], filter_height * filter_width],
                               dtype=np.float32)

        for i in range(filter_height):
            for j in range(filter_width):
                tile_filter[i, j, :, i * filter_width + j] = 1.0

        tile_filter_op = tf.constant(tile_filter, dtype=tf.float32)
        output = tf.nn.depthwise_conv2d(input, tile_filter_op, strides=[1, stride, stride, 1], padding='VALID')
        output_shape = output.get_shape()
        output = tf.reshape(output, shape=[int(output_shape[0]), int(
            output_shape[1]), int(output_shape[2]), int(input_shape[3]), filter_height * filter_width])
        output = tf.transpose(output, perm=[0, 1, 2, 4, 3])  # [batch_size, L-K+1, 1, width * height, pose_len + 1]

        return output


    def squash(self, vector, epsilon=1e-9):
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










