import numpy as np
import tensorflow as tf
import random
from recommender.BasicRcommender_soft import BasicRecommender_soft
import time
from component.Conv import CNN_Compoment
from component.Capsule import Capsule_Component
from component.MLP import MLP

class RUMIRecommender_Ksoft(BasicRecommender_soft):

    def __init__(self, dataModel, config):

        super(RUMIRecommender_Ksoft, self).__init__(dataModel, config)

        self.train_users = dataModel.train_users
        self.train_sequences_input = dataModel.train_sequences_input
        self.train_sequences_user_input = dataModel.train_sequences_user_input
        self.train_sequences_target = dataModel.train_sequences_target
        self.user_pred_sequences = dataModel.user_pred_sequences
        self.user_pred_user_sequences = dataModel.user_pred_user_sequences
        "testSize 和 numUser 数量上比较一致"
        self.trainSize = len(self.train_users)
        "这个是一个epoch中有几个batch，而不是一个batch中有多少个number，每个epoch中同时对多个user进行训练"
        self.trainBatchNum = int(self.trainSize // self.trainBatchSize) + 1

        self.name = 'RUM-ASS'

        self.familiar_user_num = dataModel.familiar_user_num

        self.numFactor = config['numFactor']
        self.factor_lambda = config['factor_lambda']
        self.input_length = config['input_length']
        self.target_length = config['target_length']
        self.dropout_keep = config['dropout_keep']
        self.loss = config['loss']
        self.khsoft = config['khsoft']
        self.layers = config['layers']
        self.drop_user = config['dropout_user']
        self.drop_userL = config['dropout_userL']
        self.drop_memory = config['dropout_memory']
        self.valid_mask = None
        self.num_units = config['cell_numbers']
        self.gru_model = config['gru_model']
        self.neg_num = config['negative_numbers']
        self.decrease_soft = config['decrease soft']
        self.loss_type = config['loss_type']
        self.old_loss = config['old_loss']
        self.merge_type = config['merge_type']
        self.dynamic_item_type = config['dynamic_item_type']

        # placeholders
        self.u_id = tf.placeholder(tf.int32, [self.trainBatchSize, 1])
        self.input_seq = tf.placeholder(tf.int32, [self.trainBatchSize, self.input_length])
        self.input_user_seq = tf.placeholder(tf.int32, [self.trainBatchSize, self.input_length * self.familiar_user_num])
        self.target_seq_pos = tf.placeholder(tf.int32, [self.trainBatchSize, self.target_length])
        self.target_seq_neg = tf.placeholder(tf.int32, [self.trainBatchSize, self.neg_num])
        self.u_id_test = tf.placeholder(tf.int32, [self.testBatchSize, 1])
        self.input_seq_test = tf.placeholder(tf.int32, [self.testBatchSize, self.input_length])
        self.input_user_seq_test = tf.placeholder(tf.int32, [self.testBatchSize, self.input_length * self.familiar_user_num])
        self.pred_seq = tf.placeholder(tf.int32, [self.testBatchSize, self.eval_item_num])

        if self.khsoft:
            self.node_Idx = tf.placeholder(tf.int32, [self.trainBatchSize, self.max_codelen])
            self.node_Code = tf.placeholder(tf.float32, [self.trainBatchSize, self.max_codelen])
            self.test_node_Idx = tf.placeholder(tf.int32, [self.testBatchSize, self.eval_item_num, self.max_codelen])
            self.test_node_Code = tf.placeholder(tf.float32, [self.testBatchSize, self.eval_item_num, self.max_codelen])

        "dropout层的应用"
        self.dropout_keep_placeholder = tf.placeholder_with_default(1.0, shape=())
        self.dropout_gru = 0.5
        self.cell = tf.contrib.rnn.GRUCell(self.num_units)
        self.target_weight = config['target_weight']

        self.numK = config['numK']

        if self.khsoft:
            self.hidden_units = 2 * self.numFactor
        elif self.gru_model:
            self.hidden_units = self.num_units + self.numFactor
        else:
            self.hidden_units = 2 * self.numFactor

        if not self.decrease_soft:
            self.numK = config['numK']
        elif self.loss_type == 'soft':
            self.numK = config['numK']
        else:
            self.numK = 1

        # user/item embedding
        "numFactor为embedding的size"
        with tf.variable_scope("Embedding"):
            self.denselayer = tf.get_variable("denselayer", shape=[self.numK * self.numFactor, self.hidden_units],
                                              initializer=tf.random_uniform_initializer(
                                                  minval=-1/tf.sqrt(float(self.numFactor)),
                                                  maxval=1/tf.sqrt(float(self.numFactor))),
                                              dtype=tf.float32, trainable=True)
            self.AItemLayer = tf.get_variable("A_denselayer", shape=[self.numFactor, 2 * self.numFactor],
                                              initializer=tf.random_uniform_initializer(
                                                  minval=-1/tf.sqrt(float(self.numFactor)),
                                                  maxval=1/tf.sqrt(float(self.numFactor))),
                                              dtype=tf.float32, trainable=True)

            self.CItemLayer = tf.get_variable("C_denselayer", shape=[self.numFactor, 2 * self.numFactor],
                                              initializer=tf.random_uniform_initializer(
                                                  minval=-1 / tf.sqrt(float(self.numFactor)),
                                                  maxval=1 / tf.sqrt(float(self.numFactor))),
                                              dtype=tf.float32, trainable=True)

            self.prior_weight = tf.get_variable("priorweight", shape=[self.numK, self.hidden_units],
                                                initializer=tf.random_uniform_initializer(
                                                    minval=-1 / tf.sqrt(float(self.numFactor)),
                                                    maxval=1 / tf.sqrt(float(self.numFactor))),
                                                dtype=tf.float32, trainable=True)
            if self.khsoft:
                self.nodeEmbedding = tf.get_variable("nodeEmbedding", shape=[self.numNode, self.numFactor],
                                                     initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                                                     dtype=tf.float32, trainable=True)
                self.mask_padding_lookup_table = tf.get_variable("mask_padding_lookup_table", initializer=self.NodeMask,
                                                                 dtype=tf.float32, trainable=False)

            self.priorBias = tf.Variable(tf.random_normal([self.numK], 0, 0.1))
            self.userBias = tf.Variable(tf.random_normal([2 * self.numFactor], 0, 0.1))
            self.userBias_gru = tf.Variable(tf.random_normal([self.numFactor], 0, 0.1))
            self.denseBias = tf.Variable(tf.random_normal([self.numK * self.numFactor], 0, 0.1))
            self.A_Layer_Bias = tf.Variable(tf.random_normal([self.numFactor], 0, 0.1))
            self.C_Layer_Bias = tf.Variable(tf.random_normal([self.numFactor], 0, 0.1))
            self.A_Item_Bias = tf.Variable(tf.random_normal([2 * self.numFactor], 0, 0.1))
            self.C_Item_Bias = tf.Variable(tf.random_normal([2 * self.numFactor], 0, 0.1))

            labels_vector = self.create_labels()
            self.labels = tf.constant(labels_vector)

            labels_vector1 = tf.constant(1.0, shape=[self.trainBatchSize, self.numK, 1])
            labels_vector2 = tf.constant(0.0, shape=[self.trainBatchSize, self.numK, self.neg_num])
            self.labels2 = tf.concat([labels_vector1, labels_vector2], axis=2)

            self.rnn_outputWeights = tf.get_variable(
                name="rnn_outputWeights",
                initializer=tf.contrib.layers.xavier_initializer(),
                shape=[self.num_units, self.numFactor],
                dtype=tf.float32
            )

            self.rnn_outputBias = tf.get_variable(
                name="rnn_outputBias",
                initializer=tf.constant(0.1, shape=[self.numFactor]),
                dtype=tf.float32
            )

        self.A_userEmbedding = tf.Variable(tf.random_normal([self.numUser, self.numFactor], 0, 1/tf.sqrt(float(self.numFactor))))
        self.C_userEmbedding = tf.Variable(tf.random_normal([self.numUser, self.numFactor], 0, 1/tf.sqrt(float(self.numFactor))))
        self.A_itemEmbedding = tf.Variable(tf.random_normal([self.numItem, self.numFactor], 0, 1/tf.sqrt(float(self.numFactor))))
        self.C_itemEmbedding = tf.Variable(tf.random_normal([self.numItem, self.numFactor], 0, 1/tf.sqrt(float(self.numFactor))))
        self.itemBias = tf.Variable(tf.random_normal([self.numItem], 0, 0.1))
        self.pri_weight = tf.Variable(tf.constant(0.1, shape=[1, self.numFactor]))
        self.userA_weight = tf.Variable(tf.constant(0.5, shape=[1, self.numFactor]))
        self.userC_weight = tf.Variable(tf.constant(0.5, shape=[1, self.numFactor]))
        self.a = tf.constant(1, shape=[self.trainBatchSize, self.numK, self.neg_num], dtype=tf.float32)
        self.b = tf.constant(0, shape=[self.trainBatchSize, self.numK, self.neg_num], dtype=tf.float32)
        self.c = tf.constant(0, shape=[self.trainBatchSize, self.numK, 1], dtype=tf.float32)

    def buildModel(self):
        with tf.variable_scope(tf.get_variable_scope()) as scope:

            userEmbedding = tf.reshape(tf.nn.embedding_lookup(self.C_userEmbedding, self.u_id),
                                       [-1, self.numFactor])
            user_embedding_drop = tf.nn.dropout(userEmbedding, self.drop_user)

            userEmbedding_test = tf.reshape(tf.nn.embedding_lookup(self.C_userEmbedding, self.u_id_test),
                                            [-1, self.numFactor])

            A_familiar_userEmbedding = tf.reshape(tf.nn.embedding_lookup(self.C_userEmbedding, self.input_user_seq),
                                                  [-1, self.numFactor])
            A_familiar_user_embedding_drop = tf.nn.dropout(A_familiar_userEmbedding, self.drop_user)

            A_familiar_user_embedding_test = tf.reshape(tf.nn.embedding_lookup(self.C_userEmbedding, self.input_user_seq_test),
                                                        [-1, self.numFactor])

            C_familiar_userEmbedding = tf.reshape(tf.nn.embedding_lookup(self.C_userEmbedding, self.input_user_seq),
                                                  [-1, self.numFactor])
            C_familiar_user_embedding_drop = tf.nn.dropout(C_familiar_userEmbedding, self.drop_user)

            C_familiar_user_embedding_test = tf.reshape(tf.nn.embedding_lookup(self.C_userEmbedding, self.input_user_seq_test),
                                                        [-1, self.numFactor])

            "memory: trainSize * input_seq * numfactor"
            A_memory = tf.nn.embedding_lookup(self.C_itemEmbedding, self.input_seq)
            C_memory = tf.nn.embedding_lookup(self.C_itemEmbedding, self.input_seq)

            A_memory_test = tf.nn.embedding_lookup(self.C_itemEmbedding, self.input_seq_test)
            C_memory_test = tf.nn.embedding_lookup(self.C_itemEmbedding, self.input_seq_test)

            target_item_test = tf.nn.embedding_lookup(self.C_itemEmbedding, self.pred_seq)

            if self.khsoft:
                pathEmbedding = tf.nn.embedding_lookup(self.nodeEmbedding, self.node_Idx)
                pathEmbedding_drop = tf.nn.dropout(pathEmbedding, 0.8)
                nodemask = tf.nn.embedding_lookup(self.mask_padding_lookup_table, self.node_Idx)

                test_pathEmbedding = tf.nn.embedding_lookup(self.nodeEmbedding, self.test_node_Idx)
                test_nodemask = tf.nn.embedding_lookup(self.mask_padding_lookup_table, self.test_node_Idx)

                pos_preds = self.get_h_pred(user_embedding_drop, A_familiar_user_embedding_drop, C_familiar_user_embedding_drop,
                                            A_memory, C_memory, pathEmbedding_drop, self.node_Code, nodemask)
                rating_loss = tf.reduce_mean(pos_preds)
                self.cost = rating_loss
                "r_pred中存放的是batch * 100 个概率"
                self.r_pred = self.get_test_h_pred(userEmbedding_test, A_familiar_user_embedding_test,
                                                   C_familiar_user_embedding_test,
                                                   A_memory_test, C_memory_test, test_pathEmbedding,
                                                   self.test_node_Code, test_nodemask)
            elif self.gru_model:
                rating_loss = self.get_gru_pred(user_embedding_drop, C_familiar_user_embedding_drop, C_memory,
                                                self.target_seq_pos, self.target_seq_neg, self.target_length)
                rating_loss = tf.reduce_mean(rating_loss)
                self.cost = rating_loss
                self.r_pred = self.get_test_gru_pred(userEmbedding_test, C_familiar_user_embedding_test,
                                                     C_memory_test, target_item_test)
            else:
                rating_loss = self.get_pred(user_embedding_drop, A_familiar_user_embedding_drop, C_familiar_user_embedding_drop,
                                            A_memory, C_memory, self.target_seq_pos, self.target_seq_neg,
                                            self.target_length)
                rating_loss = tf.reduce_mean(rating_loss)
                self.cost = rating_loss
                self.r_pred = self.get_test_pred(userEmbedding_test, A_familiar_user_embedding_test, C_familiar_user_embedding_test,
                                                 A_memory_test, C_memory_test, target_item_test)

    def get_pred(self, user_embedding_drop, A_familiar_user_embedding_drop, C_familiar_user_embedding_drop,
                 A_memory, C_memory, target_item_ids, negative_item_ids, target_length):
        "memory_out: train_size * numfactor" \
        "user_embedding_new : train_size * numfactor"
        if self.dynamic_item_type == 'user':
            A_user_memory_out = self.dynamic_item_block_user(user_embedding_drop, A_familiar_user_embedding_drop)
            C_user_memory_out = self.dynamic_item_block_user(user_embedding_drop, C_familiar_user_embedding_drop)
        else:
            A_user_memory_out = self.dynamic_item_block_item(A_memory, A_familiar_user_embedding_drop)
            C_user_memory_out = self.dynamic_item_block_item(C_memory, C_familiar_user_embedding_drop)

        if self.merge_type == 'add':
            memory_out = self.read_memory_add(user_embedding_drop, A_memory, C_memory, A_user_memory_out,
                                              C_user_memory_out, 0.5)
        else:
            memory_out = self.read_memory_concat(user_embedding_drop, A_memory, C_memory, A_user_memory_out,
                                              C_user_memory_out, 0.5)

        memory_drop = tf.nn.dropout(memory_out, self.drop_memory)
        k_user_embedding, dot_weight = self.project_merge(user_embedding_drop, memory_drop, self.drop_userL)
        soft_weight = tf.nn.softmax(dot_weight, axis=1)
        sig_weight = tf.nn.sigmoid(dot_weight)

        user_embedding_new = tf.tanh(k_user_embedding + self.denseBias)
        user_embedding_drop = tf.nn.dropout(user_embedding_new, self.drop_userL)

        if self.decrease_soft:
            user_embedding_drop = tf.reshape(user_embedding_drop, [-1, self.numK, self.numFactor])

            pos_embedding = tf.nn.embedding_lookup(self.C_itemEmbedding, target_item_ids)
            neg_embedding = tf.nn.embedding_lookup(self.C_itemEmbedding, negative_item_ids)
    
            element_pos = tf.matmul(user_embedding_drop, pos_embedding, transpose_b=True)
            element_neg = tf.matmul(user_embedding_drop, neg_embedding, transpose_b=True)

            if self.loss_type == 'bpr':
                "BPR loss"
                item_wise = - tf.subtract(element_neg, element_pos)
                log_loss = - tf.reduce_mean(tf.reduce_sum(tf.log(tf.nn.sigmoid(item_wise) + 1e-7), axis=2), axis=1)
                log_intention = tf.reshape(log_loss, [-1, 1])
            elif self.loss_type == 'neg':
                "Negative Sampling"
                log_loss = tf.reduce_mean(- tf.log(tf.nn.sigmoid(element_pos) + 1e-7) - tf.reduce_mean(
                    tf.log(1 - tf.nn.sigmoid(element_neg) + 1e-7), axis=2), axis=1)
                log_intention = tf.reshape(log_loss, [-1, 1])
            elif self.loss_type == 'top1':
                "TOP1 loss"
                item_wise = tf.nn.sigmoid(tf.subtract(element_neg, element_pos))
                reg = tf.nn.sigmoid(tf.square(element_neg))
                log_loss = tf.reduce_mean(tf.reduce_mean(item_wise + reg, axis=2), axis=1)
                log_intention = tf.reshape(log_loss, [-1, 1])
            else:
                "PEB"
                if self.old_loss:
                    element_wise_mul = tf.nn.softmax(tf.concat([element_pos, element_neg], axis=2), axis=2)
                    mse_log = tf.abs(element_wise_mul - sig_weight)
                    mse_t = (self.target_weight + element_wise_mul) - self.target_weight * (mse_log + element_wise_mul)
                    mse_p = tf.log(mse_t + 1e-7)
                    mse_n = tf.log((1 - mse_t) + 1e-7)
                    mse_loss = tf.reshape(tf.reduce_mean(tf.reduce_sum(self.labels2 * (mse_n - mse_p) - mse_n,
                                                                       axis=2), axis=1), [-1, 1])
                else:
                    element_wise_mul = tf.nn.softmax(tf.concat([element_pos, element_neg], axis=2), axis=2)
                    element_log = tf.expand_dims(tf.reduce_sum(tf.multiply(element_wise_mul, soft_weight), axis=1),
                                                 axis=1)
                    mse_log = element_wise_mul - element_log
                    log_list = tf.where(tf.greater_equal(mse_log, 0), tf.multiply(sig_weight, element_wise_mul),
                                        tf.multiply((1 - sig_weight), element_wise_mul))
                    mse_p = - tf.log(log_list + 1e-7)
                    mse_n = - tf.log(1 - element_wise_mul + 1e-7)
                    mse_p_loss = tf.reshape(tf.reduce_mean(tf.reduce_sum(self.labels2 * mse_p, axis=2), axis=1),
                                            [-1, 1])
                    mse_n_loss = tf.reshape(tf.reduce_mean(tf.reduce_sum((1 - self.labels2) * mse_n, axis=2), axis=1),
                                            [-1, 1])
                    mse_loss = mse_p_loss + mse_n_loss
                log_intention = mse_loss
        else:
            "Softmax"
            user_embedding_drop = tf.reshape(user_embedding_drop, [-1, self.numFactor])

            element_wise_mul = tf.reshape(tf.matmul(user_embedding_drop, self.C_itemEmbedding, transpose_b=True),
                                          [self.trainBatchSize, self.numK, -1])
            element_wise_soft = tf.reshape(tf.reduce_sum(tf.multiply(element_wise_mul, soft_weight), axis=1),
                                           [-1, 1, self.numItem])
            log_loss = tf.reshape(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_item_ids,
                                                                                 logits=element_wise_soft), [-1, 1])
            log_intention = tf.reshape(log_loss, [-1, 1])

        "preds: list [(train_size * 1),(train_size * 1)]"
        "return: (trainBatchSize * target_size(1))"
        return log_intention

    def get_test_pred(self, user_embedding_test, A_familiar_user_embedding_test, C_familiar_user_embedding_test,
                      A_memory_test, C_memory_test, target_item_test):

        if self.dynamic_item_type == 'user':
            A_user_memory_out = self.dynamic_item_block_user(user_embedding_test, A_familiar_user_embedding_test)
            C_user_memory_out = self.dynamic_item_block_user(user_embedding_test, C_familiar_user_embedding_test)
        else:
            A_user_memory_out = self.dynamic_item_block_item(A_memory_test, A_familiar_user_embedding_test)
            C_user_memory_out = self.dynamic_item_block_item(C_memory_test, C_familiar_user_embedding_test)

        if self.merge_type == 'add':
            memory_out = self.read_memory_add(user_embedding_test, A_memory_test, C_memory_test, A_user_memory_out,
                                              C_user_memory_out, 1.0)
        else:
            memory_out = self.read_memory_concat(user_embedding_test, A_memory_test, C_memory_test, A_user_memory_out,
                                                 C_user_memory_out, 1.0)

        k_user_embedding, dot_weight = self.project_merge(user_embedding_test, memory_out, 1.0)
        soft_weight = tf.nn.softmax(dot_weight, axis=1)
        sig_weight = tf.nn.sigmoid(dot_weight)
        user_embedding_test = tf.tanh(k_user_embedding + self.denseBias)

        "element_wise_mul: train_size * numItem"
        "gru_test = self.GRU(user_embedding_test, 1.0)"
        user_embedding_test = tf.reshape(user_embedding_test, [-1, self.numK, self.numFactor])
        target_item_test = tf.reshape(target_item_test, [-1, self.eval_item_num, self.numFactor])
        element_wise_mul = tf.nn.softmax(
            tf.reshape(tf.matmul(user_embedding_test, target_item_test, transpose_b=True),
                       [-1, self.numK, self.eval_item_num]), axis=2)

        if self.numK > 1:
            element_wise_soft = tf.reduce_sum(tf.multiply(element_wise_mul, soft_weight), axis=1)
        else:
            element_wise_soft = tf.reshape(element_wise_mul, [-1, self.eval_item_num])
        "return shape: (testBatchSize, self.eval_item_num)"
        return element_wise_soft

    def get_h_pred(self, user_embedding_drop, A_familiar_user_embedding_drop, C_familiar_user_embedding_drop,
                   A_memory, C_memory, pathEmbedding, nodeCode, nodeMask):
        "memory_out shape: (train_size, numfactor)  target_item_ids shape：(train_batch，1), nparray"
        "user_embedding_new shape : (train_size, numfactor)"
        "pathEmbedding: shape: [trainBatch, max_codelen, numfactor]"
        A_user_memory_out = self.read_user_memory(user_embedding_drop, A_familiar_user_embedding_drop)
        C_user_memory_out = self.read_user_memory(user_embedding_drop, C_familiar_user_embedding_drop)
        memory_out = self.read_memory(user_embedding_drop, A_memory, C_memory, A_user_memory_out,
                                      C_user_memory_out)
        memory_drop = tf.nn.dropout(memory_out, self.drop_memory)
        k_user_embedding, dot_weight = self.project_merge(user_embedding_drop, memory_drop, self.drop_userL)
        soft_weight = tf.nn.softmax(dot_weight, axis=1)
        soft_weight = tf.reshape(soft_weight, [-1, self.numK, 1])
        user_embedding_new = tf.tanh(k_user_embedding + self.denseBias)
        user_embedding_drop = tf.reshape(tf.nn.dropout(user_embedding_new, self.drop_userL), [-1, self.numK, self.numFactor])

        nodeMask = tf.reshape(nodeMask, [self.trainBatchSize, -1])
        pre_logit = tf.reshape(tf.matmul(user_embedding_drop, pathEmbedding, transpose_b=True), [self.trainBatchSize, self.numK, -1])
        pre_soft = tf.reduce_sum(tf.multiply(pre_logit, soft_weight), axis=1)
        pre_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=nodeCode, logits=pre_soft)
        pre_loss2 = tf.multiply(pre_loss, nodeMask)
        re_loss = tf.reduce_sum(pre_loss2, axis=1)
        return re_loss

    def get_test_h_pred(self, user_embedding_test, A_familiar_user_embedding_test, C_familiar_user_embedding_test,
                        A_memory, C_memory, pathEmbedding, nodeCode, nodeMask):

        A_user_memory_out = self.read_user_memory(user_embedding_test, A_familiar_user_embedding_test)
        C_user_memory_out = self.read_user_memory(user_embedding_test, C_familiar_user_embedding_test)

        memory_out = self.read_memory(user_embedding_test, A_memory, C_memory, A_user_memory_out,
                                      C_user_memory_out)
        k_user_embedding, dot_weight = self.project_merge(user_embedding_test, memory_out, 1.0)
        soft_weight = tf.nn.softmax(dot_weight, axis=1)
        user_embedding_new = tf.tanh(k_user_embedding + self.denseBias)
        user_embedding_test = tf.reshape(user_embedding_new, [-1, self.numK, self.numFactor])

        nodeMask = tf.reshape(nodeMask, [self.testBatchSize, -1, self.max_codelen])
        "pathEmbedding: shape: [testBatch, eval_item, max_codelen, numFactor]"
        pathEmbedding = tf.reshape(pathEmbedding, [self.testBatchSize, -1, self.numFactor])
        pre_logit = tf.reshape(tf.matmul(user_embedding_test, pathEmbedding, transpose_b=True),
                               [-1, self.eval_item_num, self.numK, self.max_codelen])
        soft_weight = tf.reshape(soft_weight, [-1, 1, self.numK, 1])
        pre_soft = tf.reduce_sum(tf.multiply(pre_logit, soft_weight), axis=2)
        pre_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=nodeCode, logits=pre_soft)
        re_logit = tf.negative(tf.reduce_sum(tf.multiply(pre_loss, nodeMask), axis=2))
        "return shape: [testBatch, eval_item]"
        return re_logit

    def get_gru_pred(self, user_embedding_drop, C_familiar_user_embedding_drop, C_memory, target_item_ids,
                     negative_item_ids, target_length):
        "memory_out: train_size * numfactor" \
        "user_embedding_new : train_size * numfactor"

        if self.dynamic_item_type == 'user':
            C_user_memory_out = tf.reshape(self.dynamic_item_block_user(user_embedding_drop, C_familiar_user_embedding_drop),
                                           [-1, self.input_length, self.numFactor])
        else:
            C_user_memory_out = tf.reshape(self.dynamic_item_block_item(C_memory, C_familiar_user_embedding_drop),
                                           [-1, self.input_length, self.numFactor])

        if self.merge_type == 'add':
            C_item_pre_embedding = tf.add(C_memory, tf.multiply(tf.clip_by_value(self.userC_weight, 0.1, 1.0),
                                                                C_user_memory_out))
            item_embed_input = tf.nn.dropout(tf.reshape(C_item_pre_embedding, [-1, self.input_length, self.numFactor]),
                                             self.drop_memory)
        else:
            C_item_pre_embedding = tf.reshape(tf.nn.dropout(tf.tanh(tf.concat([C_memory, C_user_memory_out], axis=2)
                                              + self.C_Item_Bias), self.drop_memory), [-1, 2 * self.numFactor])

            C_project = tf.tanh(tf.matmul(C_item_pre_embedding, self.CItemLayer, transpose_b=True) + self.C_Layer_Bias)

            item_embed_input = tf.nn.dropout(tf.reshape(C_project, [-1, self.input_length, self.numFactor]),
                                             self.drop_memory)

        rnn_outputs, curr_state = tf.nn.dynamic_rnn(
            cell=self.cell,
            inputs=item_embed_input,
            dtype=tf.float32,
        )
        split_outputs = tf.reshape(rnn_outputs, [-1, self.input_length, self.num_units])
        user_embedding_drop = tf.tanh(user_embedding_drop + self.userBias_gru)
        gru_vector = tf.nn.dropout(tf.concat([tf.reshape(split_outputs[:, -1:, :], [-1, self.num_units]),
                                              tf.reshape(user_embedding_drop, [-1, self.numFactor])], axis=1),
                                   self.drop_user)
        gru_vector = tf.reshape(gru_vector, [-1, self.num_units + self.numFactor])
        dot_weight = tf.reshape(tf.matmul(gru_vector, self.prior_weight, transpose_b=True), [-1, self.numK, 1])

        user_embedding_new = tf.reshape(tf.tanh(tf.matmul(gru_vector, self.denselayer, transpose_b=True)
                                                + self.denseBias), [-1, self.numK, self.numFactor])

        soft_weight = tf.nn.softmax(dot_weight, axis=1)
        sig_weight = tf.nn.sigmoid(dot_weight)
        user_embedding_drop = tf.nn.dropout(user_embedding_new, self.drop_userL)

        if self.decrease_soft:
            user_embedding_drop = tf.reshape(user_embedding_drop, [-1, self.numK, self.numFactor])

            pos_embedding = tf.nn.embedding_lookup(self.C_itemEmbedding, target_item_ids)
            neg_embedding = tf.nn.embedding_lookup(self.C_itemEmbedding, negative_item_ids)

            element_pos = tf.matmul(user_embedding_drop, pos_embedding, transpose_b=True)
            element_neg = tf.matmul(user_embedding_drop, neg_embedding, transpose_b=True)

            if self.loss_type == 'bpr':
                "BPR loss"
                item_wise = - tf.subtract(element_neg, element_pos)
                log_loss = - tf.reduce_mean(tf.reduce_sum(tf.log(tf.nn.sigmoid(item_wise) + 1e-7), axis=2), axis=1)
                log_intention = tf.reshape(log_loss, [-1, 1])
            elif self.loss_type == 'neg':
                "Negative Sampling"
                log_loss = tf.reduce_mean(- tf.log(tf.nn.sigmoid(element_pos) + 1e-7) - tf.reduce_mean(
                    tf.log(1 - tf.nn.sigmoid(element_neg) + 1e-7), axis=2), axis=1)
                log_intention = tf.reshape(log_loss, [-1, 1])
            elif self.loss_type == 'top1':
                item_wise = tf.nn.sigmoid(tf.subtract(element_neg, element_pos))
                reg = tf.nn.sigmoid(tf.square(element_neg))
                log_loss = tf.reduce_mean(tf.reduce_mean(item_wise + reg, axis=2), axis=1)
                log_intention = tf.reshape(log_loss, [-1, 1])
            else:
                "PEB"
                if self.old_loss:
                    element_wise_mul = tf.nn.softmax(tf.concat([element_pos, element_neg], axis=2), axis=2)
                    mse_log = tf.abs(element_wise_mul - sig_weight)
                    mse_t = (self.target_weight + element_wise_mul) - self.target_weight * (mse_log + element_wise_mul)
                    mse_p = tf.log(mse_t + 1e-7)
                    mse_n = tf.log((1 - mse_t) + 1e-7)
                    mse_loss = tf.reshape(tf.reduce_mean(tf.reduce_sum(self.labels2 * (mse_n - mse_p) - mse_n,
                                                                       axis=2), axis=1), [-1, 1])
                else:
                    element_wise_mul = tf.nn.softmax(tf.concat([element_pos, element_neg], axis=2), axis=2)
                    """
                    element_wise_soft = tf.reduce_sum(tf.multiply(element_wise_mul, sig_weight), axis=1)
                    element_log = tf.expand_dims(tf.multiply(1 / tf.reduce_sum(sig_weight, axis=1), element_wise_soft),
                                                             axis=1)
                    """
                    element_log = tf.expand_dims(tf.reduce_sum(tf.multiply(element_wise_mul, soft_weight), axis=1),
                                                 axis=1)
                    mse_log = element_wise_mul - element_log
                    log_list = tf.where(tf.greater_equal(mse_log, 0), tf.multiply(sig_weight, element_wise_mul),
                                        tf.multiply((1 - sig_weight), element_wise_mul))
                    mse_p = - tf.log(log_list + 1e-7)
                    mse_n = - tf.log(1 - element_wise_mul + 1e-7)
                    mse_p_loss = tf.reshape(tf.reduce_mean(tf.reduce_sum(self.labels2 * mse_p, axis=2), axis=1),
                                            [-1, 1])
                    mse_n_loss = tf.reshape(tf.reduce_mean(tf.reduce_sum((1 - self.labels2) * mse_n, axis=2), axis=1),
                                            [-1, 1])
                    mse_loss = mse_p_loss + mse_n_loss

                log_intention = mse_loss
        else:
            "All Softmax"
            user_embedding_drop = tf.reshape(user_embedding_drop, [-1, self.numFactor])

            element_wise_mul = tf.reshape(tf.matmul(user_embedding_drop, self.C_itemEmbedding, transpose_b=True),
                                          [self.trainBatchSize, self.numK, -1])
            element_wise_soft = tf.reshape(tf.reduce_sum(tf.multiply(element_wise_mul, soft_weight), axis=1),
                                           [-1, 1, self.numItem])
            log_loss = tf.reshape(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_item_ids,
                                                                                 logits=element_wise_soft), [-1, 1])
            log_intention = tf.reshape(log_loss, [-1, 1])

        return log_intention

    def get_test_gru_pred(self, user_embedding_test, C_familiar_user_embedding_test, C_memory_test, target_item_test):

        if self.dynamic_item_type == 'user':
            C_user_memory_out = self.dynamic_item_block_user(user_embedding_test, C_familiar_user_embedding_test)
        else:
            C_user_memory_out = tf.reshape(self.dynamic_item_block_item(C_memory_test, C_familiar_user_embedding_test),
                                           [-1, self.input_length, self.numFactor])

        if self.merge_type == 'add':
            C_item_pre_embedding = tf.add(C_memory_test, tf.multiply(tf.clip_by_value(self.userC_weight, 0.1, 1.0),
                                                                     C_user_memory_out))
            item_embed_input = tf.reshape(C_item_pre_embedding, [-1, self.input_length, self.numFactor])
        else:
            C_item_pre_embedding = tf.reshape(tf.tanh(tf.concat([C_memory_test, C_user_memory_out], axis=2)
                                                      + self.C_Item_Bias), [-1, 2 * self.numFactor])

            C_project = tf.tanh(tf.matmul(C_item_pre_embedding, self.CItemLayer, transpose_b=True) + self.C_Layer_Bias)

            item_embed_input = tf.reshape(C_project, [-1, self.input_length, self.numFactor])

        rnn_outputs, curr_state = tf.nn.dynamic_rnn(
            cell=self.cell,
            inputs=item_embed_input,
            dtype=tf.float32,
        )
        split_outputs = tf.reshape(rnn_outputs, [-1, self.input_length, self.num_units])

        user_embedding_test = tf.tanh(user_embedding_test + self.userBias_gru)

        gru_vector = tf.concat([tf.reshape(split_outputs[:, -1:, :], [-1, self.num_units]),
                                tf.reshape(user_embedding_test, [-1, self.numFactor])], axis=1),
        gru_vector = tf.reshape(gru_vector, [-1, self.num_units + self.numFactor])

        dot_weight = tf.reshape(tf.matmul(gru_vector, self.prior_weight, transpose_b=True), [-1, self.numK, 1])

        user_embedding_test = tf.reshape(tf.tanh(tf.matmul(gru_vector, self.denselayer, transpose_b=True)
                                                 + self.denseBias), [-1, self.numK, self.numFactor])

        soft_weight = tf.nn.softmax(dot_weight, axis=1)
        "element_wise_mul: train_size * numItem"
        "gru_test = self.GRU(user_embedding_test, 1.0)"
        user_embedding_test = tf.reshape(user_embedding_test, [-1, self.numK, self.numFactor])
        target_item_test = tf.reshape(target_item_test, [-1, self.eval_item_num, self.numFactor])
        element_wise_mul = tf.nn.softmax(
            tf.reshape(tf.matmul(user_embedding_test, target_item_test, transpose_b=True),
                       [-1, self.numK, self.eval_item_num]), axis=2)

        if self.numK > 1:
            element_wise_soft = tf.reduce_sum(tf.multiply(element_wise_mul, soft_weight), axis=1)
        else:
            element_wise_soft = tf.reshape(element_wise_mul, [-1, self.eval_item_num])
        "return shape: (testBatchSize, self.eval_item_num)"
        return element_wise_soft

    def merge(self, u, m):
        merged = tf.add(m, tf.multiply(tf.clip_by_value(self.pri_weight, 0.1, 1.0), u))
        return merged

    def project_merge(self, u, m, drop_out):
        merged = tf.nn.dropout(tf.tanh(tf.concat([u, m], axis=1) + self.userBias), drop_out)
        project_out = tf.matmul(merged, self.denselayer, transpose_b=True)
        prior_weight = tf.reshape(tf.matmul(merged, self.prior_weight, transpose_b=True),
                                  [-1, self.numK, 1])
        return project_out, prior_weight

    def read_memory_concat(self, user_embedding, A_memory, C_memory, A_familiar_user_embedding, C_familiar_user_embedding,
                    drop_out):
        "user_embedding shape: (train_batch, numFactor) item_pre_embedding shape: (train_batch, input_size, numFactor)"

        A_item_pre_embedding = tf.reshape(tf.nn.dropout(tf.tanh(tf.concat([A_memory, A_familiar_user_embedding], axis=2)
                                          + self.A_Item_Bias), drop_out), [-1, 2 * self.numFactor])
        "shape: [-1, input_length, numFactor * 2]"
        C_item_pre_embedding = tf.reshape(tf.nn.dropout(tf.tanh(tf.concat([C_memory, C_familiar_user_embedding], axis=2)
                                          + self.C_Item_Bias), drop_out), [-1, 2 * self.numFactor])

        A_project = tf.tanh(tf.matmul(A_item_pre_embedding, self.AItemLayer, transpose_b=True) + self.A_Layer_Bias)

        C_project = tf.tanh(tf.matmul(C_item_pre_embedding, self.CItemLayer, transpose_b=True) + self.C_Layer_Bias)

        A_new_memory = tf.nn.dropout(tf.reshape(A_project, [-1, self.input_length, self.numFactor]), drop_out)
        C_new_memory = tf.nn.dropout(tf.reshape(C_project, [-1, self.input_length, self.numFactor]), drop_out)

        weight = tf.div(tf.matmul(A_new_memory, tf.expand_dims(user_embedding, axis=2)),
                        tf.sqrt(tf.to_float(self.numFactor)))
        attention = tf.nn.softmax(weight, axis=1)
        out = tf.reduce_mean(tf.multiply(C_new_memory, attention), axis=1)
        "return shape: (train_batch, numFactor)"
        return out

    def read_memory_add(self, user_embedding, A_memory, C_memory, A_familiar_user_embedding,
                    C_familiar_user_embedding, drop_out):
        "user_embedding shape: (train_batch, numFactor) item_pre_embedding shape: (train_batch, input_size, numFactor)"
        A_item_pre_embedding = tf.add(A_memory, tf.multiply(tf.clip_by_value(self.userA_weight, 0.1, 1.0),
                                                            A_familiar_user_embedding))
        C_item_pre_embedding = tf.add(C_memory, tf.multiply(tf.clip_by_value(self.userC_weight, 0.1, 1.0),
                                                            C_familiar_user_embedding))

        weight = tf.div(tf.matmul(A_item_pre_embedding, tf.expand_dims(user_embedding, axis=2)),
                        tf.sqrt(tf.to_float(self.numFactor)))
        attention = tf.nn.softmax(weight, axis=1)
        out = tf.reduce_mean(tf.multiply(C_item_pre_embedding, attention), axis=1)
        "return shape: (train_batch, numFactor)"
        return out

    def dynamic_item_block_user(self, user_embedding, user_memory_embedding):
        """user_embedding shape: (train_batch, numFactor)
           user_memory_embedding shape: (train_batch * input_size * familiar_user, numFactor)"""
        user_embedding = tf.reshape(user_embedding, [-1, 1, self.numFactor])
        user_memory_embedding = tf.reshape(user_memory_embedding,
                                           [-1, self.input_length * self.familiar_user_num, self.numFactor])

        weight = tf.reshape(tf.div(tf.matmul(user_memory_embedding, user_embedding, transpose_b=True),
                                   tf.sqrt(tf.to_float(self.numFactor))),
                            [-1, self.input_length, self.familiar_user_num])

        attention = tf.expand_dims(tf.nn.softmax(weight, axis=2), axis=3)
        out = tf.reduce_mean(tf.multiply(
            tf.reshape(user_memory_embedding, [-1, self.input_length, self.familiar_user_num, self.numFactor]),
            attention), axis=2)
        "return shape: (train_batch, input_size, numFactor)"
        return out

    def dynamic_item_block_item(self, item_memory, user_memory_embedding):
        """item_memory shape: (train_batch, input_size, numFactor)
           user_memory_embedding shape: (train_batch * input_size * familiar_user, numFactor)"""
        item_memory_embedding = tf.reshape(item_memory, [-1, 1, self.numFactor])
        user_memory_embedding = tf.reshape(user_memory_embedding,
                                           [-1, self.familiar_user_num, self.numFactor])

        weight = tf.reshape(tf.div(tf.matmul(item_memory_embedding, user_memory_embedding, transpose_b=True),
                                   tf.sqrt(tf.to_float(self.numFactor))), [-1, self.input_length, self.familiar_user_num])

        attention = tf.expand_dims(tf.nn.softmax(weight, axis=2), axis=3)
        out = tf.reduce_mean(tf.multiply(tf.reshape(user_memory_embedding, [-1, self.input_length, self.familiar_user_num, self.numFactor]),
                                         attention), axis=2)
        "return shape: (train_batch, input_size, numFactor)"
        return out

    def create_labels(self):
        label_lists = [[i, 0] for i in range(self.trainBatchSize)]
        return label_lists

    def trainEachBatch(self, epochId, batchId):
        totalLoss = 0
        start = time.time()
        feed_dict = self.getTrainData(batchId)

        self.optimizer.run(feed_dict=feed_dict)
        loss = self.sess.run(self.cost, feed_dict=feed_dict)

        totalLoss += loss
        end = time.time()
        if epochId % 5 == 0 and batchId == 0:
            self.logger.info("----------------------------------------------------------------------")
            self.logger.info(
                "batchId: %d epoch %d/%d   batch_loss: %.4f   time of a batch: %.4f" % (
                    batchId, epochId, self.maxIter, totalLoss, (end - start)))

            self.evaluateRanking(epochId, batchId)
        return totalLoss

    def getTrainData(self, batchId):
        # compute start and end
        start = time.time()
        neg_seq_batch = []

        start_idx = batchId * self.trainBatchSize
        end_idx = start_idx + self.trainBatchSize

        if end_idx > self.trainSize:
            end_idx = self.trainSize
            start_idx = end_idx - self.trainBatchSize

        if end_idx == start_idx:
            start_idx = 0
            end_idx = start_idx + self.trainBatchSize
        "从一个连续的用户列表如[1,1,1,1,1,1,2,2,3....]中选取，存在很多重复的userIdx"
        user_batch = self.train_users[start_idx:end_idx]
        input_seq_batch = self.train_sequences_input[start_idx:end_idx]
        input_user_seq_batch = self.train_sequences_user_input[start_idx:end_idx]

        pos_seq_batch = self.train_sequences_target[start_idx:end_idx]
        path_seq_batch = []
        code_seq_batch = []

        if self.khsoft:
            for itemIdx in pos_seq_batch:
                path = self.itemIdxToPath[itemIdx[0]]
                code = self.itemIdxToCode[itemIdx[0]]
                path_seq_batch.append(path)
                code_seq_batch.append(code)

        for Idx in range(len(user_batch)):
            neg_items = []
            positiveItems = pos_seq_batch[Idx]
            for i in range(self.neg_num):
                negativeItemIdx = random.randint(0, self.numItem - 1)
                while negativeItemIdx in positiveItems:
                    negativeItemIdx = random.randint(0, self.numItem - 1)
                neg_items.append(negativeItemIdx)
            neg_seq_batch.append(neg_items)

        user_batch = np.array(user_batch).reshape((end_idx - start_idx, 1))
        input_seq_batch = np.array(input_seq_batch)
        input_user_seq_batch = np.array(input_user_seq_batch).reshape((-1, self.input_length * self.familiar_user_num))
        pos_seq_batch = np.array(pos_seq_batch)
        neg_seq_batch = np.array(neg_seq_batch)

        path_seq_batch = np.array(path_seq_batch)
        code_seq_batch = np.array(code_seq_batch)

        end = time.time()
        # self.logger.info("time of collect a batch of data: " + str((end - start)) + " seconds")
        # self.logger.info("batch Id: " + str(batchId))
        if self.khsoft:
            feed_dict = {
                self.u_id: user_batch,
                self.input_seq: input_seq_batch,
                self.input_user_seq: input_user_seq_batch,
                self.target_seq_pos: pos_seq_batch,
                self.target_seq_neg: neg_seq_batch,
                self.dropout_keep_placeholder: self.dropout_keep,
                self.node_Idx: path_seq_batch,
                self.node_Code: code_seq_batch
            }
        else:
            feed_dict = {
                self.u_id: user_batch,
                self.input_seq: input_seq_batch,
                self.input_user_seq: input_user_seq_batch,
                self.target_seq_pos: pos_seq_batch,
                self.target_seq_neg: neg_seq_batch,
                self.dropout_keep_placeholder: self.dropout_keep
            }

        return feed_dict

    "test时使用，为每个user创建一个包含正样本item和负样本item的list"
    def getPredList_ByUserIdxList(self, user_idices):
        end0 = time.time()
        # build test batch
        input_seq = []
        input_user_seq = []
        target_seq = []
        path_seq = []
        code_seq = []

        for userIdx in user_idices:
            "input_seq中存放的是用来预测每个user最新的item的前几个item，作为input"
            input_seq.append(self.user_pred_sequences[userIdx])
            input_user_seq.append(self.user_pred_user_sequences[userIdx])
            "target_seq中存放的是每个user的一个正样本item，和其余的负样本item"
            target_seq.append(self.evalItemsForEachUser[userIdx])
            if self.khsoft:
                path = []
                code = []
                for itemIdx in self.evalItemsForEachUser[userIdx]:
                    path.append(self.itemIdxToPath[itemIdx])
                    code.append(self.itemIdxToCode[itemIdx])
                path_seq.append(path)
                code_seq.append(code)

        batch_u = np.array(user_idices).reshape((-1, 1))
        input_seq = np.array(input_seq)
        input_user_seq = np.array(input_user_seq).reshape((-1, self.input_length * self.familiar_user_num))
        target_seq = np.array(target_seq)
        path_seq = np.array(path_seq)
        code_seq = np.array(code_seq)

        end1 = time.time()

        if self.khsoft:
            predList = self.sess.run(self.r_pred, feed_dict={
                self.u_id_test: batch_u,
                self.input_seq_test: input_seq,
                self.input_user_seq_test: input_user_seq,
                self.pred_seq: target_seq,
                self.test_node_Idx: path_seq,
                self.test_node_Code: code_seq
            })
        else:
            predList = self.sess.run(self.r_pred, feed_dict={
                self.u_id_test: batch_u,
                self.input_seq_test: input_seq,
                self.input_user_seq_test: input_user_seq,
                self.pred_seq: target_seq,
            })
        end2 = time.time()

        output_lists = []
        for i in range(len(user_idices)):
            recommendList = {}
            start = i * self.eval_item_num
            end = start + self.eval_item_num
            for j in range(end-start):
                "将要预测的target label设置为key， 预测出的概率作为其value，然后根据value排序出最高的十个label输出"
                recommendList[target_seq[i][j]] = predList[i][j]
            sorted_RecItemList = sorted(recommendList, key=recommendList.__getitem__, reverse=True)[0:self.topN]
            "output_list [ testbatchsize * 10 ]"
            output_lists.append(sorted_RecItemList)
        end3 = time.time()

        return output_lists, end1 - end0, end2 - end1, end3 - end2

    def GRU(self, user_inputs, drop_out_ratio):
        user_inputs = tf.reshape(user_inputs, [-1, 1, self.numFactor])
        output, state = tf.nn.dynamic_rnn(
            cell=self.cell,
            inputs=user_inputs,
            dtype=tf.float32
        )
        last = tf.nn.dropout(tf.reshape(output, [-1, self.num_units]), drop_out_ratio)
        gru_output = tf.nn.tanh(tf.matmul(last, self.rnn_outputWeights) + self.rnn_outputBias)
        return gru_output