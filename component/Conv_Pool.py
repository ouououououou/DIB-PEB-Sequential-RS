import tensorflow as tf

class CNN_Pool_Compoment:

    def __init__(self, filter_num,
                 filter_sizes,
                 output_size,
                 wordvec_size,
                 max_review_length,
                 word_matrix,
                 review_wordId_print,
                 review_input_print,
                 cnn_lambda,
                 dropout_keep_prob,
                 component_raw_output,
                 item_pad_num,
                 if_fc=True,
                 name='plain'
                 ):
        # dimension settings
        self.num_filters = filter_num
        self.filter_sizes = filter_sizes
        self.wordVec_size = wordvec_size
        self.output_size = output_size
        self.maxReviewLength = max_review_length
        self.item_pad_num = item_pad_num
        # word dict
        self.word_embedding_matrix = word_matrix
        # Print objects
        self.review_wordId_print = review_wordId_print
        self.review_input_print = review_input_print
        # regularization parameters
        self.cnn_lambda = cnn_lambda
        self.dropout_keep_prob = dropout_keep_prob
        self.component_raw_output = component_raw_output
        self.if_fc = if_fc
        self.name=name

        if self.name == 'hor':
            filter_width = self.wordVec_size
        else:
            filter_width = 1

        # cnn weights
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope(self.name + "-conv-maxpool-%s" % filter_size, reuse=None):
                # Convolution Layer
                filter_shape = [filter_size, filter_width, 1, self.num_filters]
                W = tf.get_variable(name="W", initializer=tf.truncated_normal(filter_shape, stddev=0.1),
                                    dtype=tf.float32)
                b = tf.get_variable(name="b", initializer=tf.constant(0.1, shape=[self.num_filters]), dtype=tf.float32)


        with tf.variable_scope(self.name + "-cnn_final_output", reuse=None):
            output_W = tf.get_variable(
                "W",
                shape=[self.num_filters_total, self.output_size],
                initializer=tf.contrib.layers.xavier_initializer(),
                dtype=tf.float32
            )
            b = tf.get_variable(
                name="b",
                initializer=tf.constant(0.1, shape=[self.output_size]),
                dtype=tf.float32
            )

    def get_l2_loss(self):
        with tf.variable_scope("cnn_final_output", reuse=True):
            output_W = tf.get_variable(
                "W",
                shape=[self.num_filters_total, self.output_size],
                initializer=tf.contrib.layers.xavier_initializer(),
                dtype=tf.float32
            )
        return self.cnn_lambda * tf.nn.l2_loss(output_W)

    def get_outputs(self, input_reviews):
        """
        get the output of all reviews

        :param      input_reviews: all reviews
        :return:    the output of all reviews
        """
        split_list = [1] * self.item_pad_num
        splitted_review_wordId_intputs = tf.split(input_reviews, split_list, 1)
        cnn_outputs = []
        for i in range(self.item_pad_num):
            input_review = tf.squeeze(splitted_review_wordId_intputs[i], [1])
            cnn_output = self.get_single_output(input_review=input_review, index=i)
            cnn_outputs.append(cnn_output)

        return cnn_outputs

    def get_horizontal_output(self, input_indices, local_position_embed=None, globla_position_embed = None):

        reshaped_embedding_input = tf.nn.embedding_lookup(self.word_embedding_matrix, input_indices)
        review_embedding_input = tf.reshape(reshaped_embedding_input, [-1, self.maxReviewLength, self.wordVec_size])
        if local_position_embed is not None:
            review_embedding_input += local_position_embed
        if globla_position_embed is not None:
            review_embedding_input += globla_position_embed
        review_input_expanded = tf.expand_dims(review_embedding_input, -1)

        pooled_outputs = []

        # Create a convolution + maxpool layer for each filter size
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("hor-conv-maxpool-%s" % filter_size, reuse=True):
                # Convolution Layer
                # filter_shape = [filter_height, filter_width, in_channels, out_channels]
                filter_shape = [filter_size, self.wordVec_size, 1, self.num_filters]
                W = tf.get_variable(name="W", initializer=tf.truncated_normal(filter_shape, stddev=0.1),
                                    dtype=tf.float32)
                b = tf.get_variable(name="b", initializer=tf.constant(0.1, shape=[self.num_filters]),
                                    dtype=tf.float32)
                # conv = [batchsize, self.maxReviewLength - filter_size + 1, in_channels, out_channels]
                conv = tf.nn.conv2d(
                    input=review_input_expanded,
                    filter=W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv_layer",
                    use_cudnn_on_gpu=True)
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                # pooled = [batchsize, 1, in_channels, out_channels]
                pooled = tf.nn.max_pool(
                    value=h,
                    ksize=[1, self.maxReviewLength - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool_layer")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters_total])

        return h_pool_flat

    def get_vertical_output(self, input_indices, local_position_embed=None, globla_position_embed = None):
        reshaped_embedding_input = tf.nn.embedding_lookup(self.word_embedding_matrix, input_indices)
        review_embedding_input = tf.reshape(reshaped_embedding_input, [-1, self.maxReviewLength, self.wordVec_size])
        if local_position_embed is not None:
            review_embedding_input += local_position_embed
        if globla_position_embed is not None:
            review_embedding_input += globla_position_embed
        review_input_expanded = tf.expand_dims(review_embedding_input, -1)

        outputs = []

        # Create a convolution + maxpool layer for each filter size
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("ver-conv-maxpool-%s" % filter_size, reuse=True):
                # Convolution Layer
                # filter_shape = [filter_height, filter_width, in_channels, out_channels]
                filter_shape = [self.maxReviewLength, 1, 1, self.num_filters]
                W = tf.get_variable(name="W", initializer=tf.truncated_normal(filter_shape, stddev=0.1),
                                    dtype=tf.float32)
                b = tf.get_variable(name="b", initializer=tf.constant(0.1, shape=[self.num_filters]),
                                    dtype=tf.float32)
                # conv = [batchsize, self.maxReviewLength - filter_size + 1, in_channels, out_channels]
                conv = tf.nn.conv2d(
                    input=review_input_expanded,
                    filter=W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv_layer",
                    use_cudnn_on_gpu=True)
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                outputs.append(h)

        h_pool = tf.concat(outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters * self.wordVec_size])

        return h_pool_flat

    def get_single_output(self, input_review, index=-1):
        """
        get the output of a single piece of review

        :param      input_review:   shape: [batchsize, self.maxReviewLength]
        :param      index:          distinguish the input index of reviews
        :return:    the output of a single piece of review
        """
        if index >= 0:
            self.review_wordId_print[index] = tf.Print(
                input_review,
                [input_review],
                message="review_wordId_input%d" % (index),
                summarize=10
            )

        reshaped_embedding_input = tf.nn.embedding_lookup(self.word_embedding_matrix, input_review)
        review_embedding_input = tf.reshape(reshaped_embedding_input, [-1, self.maxReviewLength, self.wordVec_size])

        if index >= 0:
            self.review_input_print[index] = tf.Print(
                review_embedding_input,
                [review_embedding_input],
                message="review_input_%d" % (index),
                summarize=600
            )

        review_input_expanded = tf.expand_dims(review_embedding_input, -1)

        pooled_outputs = []
        # Create a convolution + maxpool layer for each filter size
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size, reuse=True):
                # Convolution Layer
                # filter_shape = [filter_height, filter_width, in_channels, out_channels]
                filter_shape = [filter_size, self.wordVec_size, 1, self.num_filters]
                W = tf.get_variable(name="W", initializer=tf.truncated_normal(filter_shape, stddev=0.1),
                                    dtype=tf.float32)
                b = tf.get_variable(name="b", initializer=tf.constant(0.1, shape=[self.num_filters]),
                                    dtype=tf.float32)
                # conv = [batchsize, self.maxReviewLength - filter_size + 1, in_channels, out_channels]
                conv = tf.nn.conv2d(
                    input=review_input_expanded,
                    filter=W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv_layer",
                    use_cudnn_on_gpu=True)
                # Apply nonlinearity
                h = tf.nn.sigmoid(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                # pooled = [batchsize, 1, in_channels, out_channels]
                pooled = tf.nn.max_pool(
                    value=h,
                    ksize=[1, self.maxReviewLength - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool_layer")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

        with tf.variable_scope("cnn_final_output", reuse=True):
            output_W = tf.get_variable(
                "W",
                shape=[self.num_filters_total, self.output_size],
                initializer=tf.contrib.layers.xavier_initializer(),
                dtype=tf.float32
            )
            b = tf.get_variable(
                name="b",
                initializer=tf.constant(0.1, shape=[self.output_size]),
                dtype=tf.float32
            )
            scores = tf.nn.xw_plus_b(h_drop, output_W, b, name="scores")

        return scores