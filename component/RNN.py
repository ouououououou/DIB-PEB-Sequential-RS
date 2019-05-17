import tensorflow as tf

class RNN_Compoment:

    def __init__(self,
                 rnn_unit_num,
                 rnn_layer_num,
                 rnn_cell,
                 output_size,
                 wordvec_size,
                 input_placeholder,
                 max_review_length,
                 word_matrix,
                 review_wordId_print,
                 review_input_print,
                 rnn_lambda,
                 dropout_keep_prob,
                 component_raw_output,
                 item_pad_num
                 ):
        # dimension settings
        self.rnn_unit_num = rnn_unit_num
        self.rnn_layer_num = rnn_layer_num
        self.rnn_cell = rnn_cell
        self.wordVec_size = wordvec_size
        self.output_size = output_size
        self.maxReviewLength = max_review_length
        self.item_pad_num = item_pad_num
        # input placeholder
        self.rnn_review_input = input_placeholder
        # word dict
        self.word_embedding_matrix = word_matrix
        # Print objects
        self.review_wordId_print = review_wordId_print
        self.review_input_print = review_input_print
        # regularization parameters
        self.rnn_lambda = rnn_lambda
        self.dropout_keep_prob = dropout_keep_prob
        self.component_raw_output = component_raw_output

        # build rnn network
        self.rnn_network = self.RNN()
        self.rnn_outputWeights = None
        self.rnn_outputBias = None

    def return_network(self):
        return self.rnn_network

    def RNN(self):
        num_units = self.rnn_unit_num
        num_layers = self.rnn_layer_num

        cells = []

        for _ in range(num_layers):
            if self.rnn_cell == 'GRU':
                cell = tf.contrib.rnn.GRUCell(num_units)  # Or LSTMCell(num_units)
            else:
                cell = tf.contrib.rnn.LSTMCell(num_units)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
            cells.append(cell)
        return tf.contrib.rnn.MultiRNNCell(cells)

    def get_single_output(self, input_review, index=-1):
        # output weight/bias of rnn
        self.rnn_outputWeights = tf.get_variable(
            name="rnn_outputWeights",
            initializer=tf.contrib.layers.xavier_initializer(),
            shape=[self.rnn_unit_num, self.output_size],
            dtype=tf.float32
        )

        self.rnn_outputBias = tf.get_variable(
                name="rnn_outputBias",
                initializer=tf.constant(0.1, shape=[self.output_size]),
                dtype=tf.float32
        )

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
        # Batch size x time steps x features.
        review_embedding_input = tf.reshape(reshaped_embedding_input, [-1, self.maxReviewLength, self.wordVec_size])

        if index >= 0:
            self.review_input_print[index] = tf.Print(
                review_embedding_input,
                [review_embedding_input],
                message="review_input_%d" % (index),
                summarize=600)

        length = self.length(review_embedding_input)
        output, _ = tf.nn.dynamic_rnn(
            cell=self.rnn_network,
            inputs=review_embedding_input,
            dtype=tf.float32,
            sequence_length=length,
        )
        last = self.last_relevant(output, length)
        rele_output = tf.nn.relu(tf.matmul(last, self.rnn_outputWeights) + self.rnn_outputBias)

        return rele_output

    def get_seq_outputs(self, inputs, init_state):
        outputs, new_state = tf.nn.dynamic_rnn(
            cell=self.rnn_network,
            inputs=inputs,
            initial_state=init_state
        )
        return outputs, new_state



    def get_l2_loss(self):

        return self.rnn_lambda * tf.nn.l2_loss(self.rnn_outputWeights)

    def get_outputs(self, input_reviews):
        """
        get the output of all reviews

        :param      input_reviews: all reviews
        :return:    the output of all reviews
        """
        split_list = [1] * self.item_pad_num
        splitted_review_wordId_intputs = tf.split(self.rnn_review_input, split_list, 1)
        rnn_outputs = []
        for i in range(self.item_pad_num):
            input_review = tf.squeeze(splitted_review_wordId_intputs[i], [1])
            rnn_output = self.get_single_output(input_review=input_review, index=i)
            rnn_outputs.append(rnn_output)

        return rnn_outputs

    def last_relevant(self, output, length):
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant

    def length(self, review):
        used = tf.sign(tf.reduce_max(tf.abs(review), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length