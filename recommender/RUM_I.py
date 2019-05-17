import numpy as np
import tensorflow as tf
import random
from recommender.BasicRcommender_soft import BasicRecommender_soft
import time
from component.Conv import CNN_Compoment
from component.Capsule import Capsule_Component
from component.MLP import MLP

class RUMIRecommender(BasicRecommender_soft):

    def __init__(self, dataModel, config):

        super(RUMIRecommender, self).__init__(dataModel, config)

        self.train_users = dataModel.train_users
        self.train_sequences_input = dataModel.train_sequences_input
        self.train_sequences_target = dataModel.train_sequences_target
        self.user_pred_sequences = dataModel.user_pred_sequences

        self.trainSize = len(self.train_users)
        self.trainBatchNum = int(self.trainSize // self.trainBatchSize) + 1

        self.name = 'RUM(I)'

        self.numFactor = config['numFactor']
        self.factor_lambda = config['factor_lambda']
        self.input_length = config['input_length']
        self.target_length = config['target_length']
        self.dropout_keep = config['dropout_keep']
        self.loss = config['loss']

        # placeholders
        self.u_id = tf.placeholder(tf.int32, [self.trainBatchSize, 1])
        self.input_seq = tf.placeholder(tf.int32, [self.trainBatchSize, self.input_length])
        self.target_seq_pos = tf.placeholder(tf.int32, [self.trainBatchSize, self.target_length])
        self.target_seq_neg = tf.placeholder(tf.int32, [self.trainBatchSize, self.neg_num])
        self.pred_seq = tf.placeholder(tf.int32, [self.trainBatchSize, self.eval_item_num])
        self.dropout_keep_placeholder = tf.placeholder_with_default(1.0, shape=())

        # user/item embedding
        self.userEmbedding = tf.Variable(tf.random_normal([self.numUser, self.numFactor], 0, 0.1))
        self.itemEmbedding = tf.Variable(tf.random_normal([self.numItem, self.numFactor], 0, 0.1))
        self.itemBias = tf.Variable(tf.random_normal([self.numItem], 0, 0.1))

    def buildModel(self):
        with tf.variable_scope(tf.get_variable_scope()) as scope:

            userEmbedding = tf.reshape(tf.nn.embedding_lookup(self.userEmbedding, self.u_id), [-1, self.numFactor])


            memory = tf.nn.embedding_lookup(self.itemEmbedding, self.input_seq)

            pos_preds = self.get_pred(userEmbedding, memory, self.target_seq_pos, self.target_length)
            neg_preds = self.get_pred(userEmbedding, memory, self.target_seq_neg, self.neg_num)

            if self.loss == 'bpr':
                rating_loss = - tf.reduce_sum(tf.log(tf.sigmoid(-(neg_preds - pos_preds))))
            else:
                rating_loss = - tf.reduce_mean(tf.log(tf.nn.sigmoid(pos_preds))) - tf.reduce_mean(tf.log(1 - tf.nn.sigmoid(neg_preds)))

            self.cost = rating_loss
            self.r_pred = self.get_pred(userEmbedding, memory, self.pred_seq, self.eval_item_num)

    def get_pred(self, userEmbedding, memory, target_item_ids, target_length):

        split_list = [1] * target_length
        target_item_id_list = tf.split(target_item_ids, split_list, 1)
        preds = []

        for target_item_id in target_item_id_list:
            "target_itemEmbedding : shape [trainBatchSize, self.numFactor]"
            target_itemEmbedding = tf.reshape(tf.nn.embedding_lookup(self.itemEmbedding, target_item_id),
                                              [-1, self.numFactor])

            memory_out = self.read_memory(target_itemEmbedding, memory)

            user_embedding_new = self.merge(userEmbedding, memory_out)

            element_wise_mul = tf.multiply(user_embedding_new, target_itemEmbedding)
            element_wise_mul_drop = tf.nn.dropout(element_wise_mul, self.dropout_keep_placeholder)

            log_intention = tf.reshape(tf.reduce_sum(element_wise_mul_drop, axis=1), [-1, 1])

            preds.append(log_intention)
        "return shape: [trainBatchSize, target_length]"
        return tf.concat(preds, axis=1)

    def merge(self, u, m):
        merged = tf.add(u, tf.multiply(tf.constant(0.2), m))
        return merged

    def read_memory(self, item_embedding, item_pre_embedding):
        self.weight = tf.nn.softmax(tf.matmul(item_pre_embedding, tf.expand_dims(item_embedding, axis=2)))
        out = tf.reduce_mean(tf.multiply(item_pre_embedding, self.weight), axis=1)
        return out

    def trainEachBatch(self, epochId, batchId):
        totalLoss = 0
        start = time.time()
        feed_dict = self.getTrainData(batchId)

        self.optimizer.run(feed_dict=feed_dict)
        loss = self.cost.eval(feed_dict=feed_dict)

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

        user_batch = self.train_users[start_idx:end_idx]
        input_seq_batch = self.train_sequences_input[start_idx:end_idx]
        pos_seq_batch = self.train_sequences_target[start_idx:end_idx]

        for userIdx in user_batch:
            neg_items = []
            for i in range(self.neg_num):
                positiveItems = self.user_items_train[userIdx]
                negativeItemIdx = random.randint(0, self.numItem - 1)
                while negativeItemIdx in positiveItems:
                    negativeItemIdx = random.randint(0, self.numItem - 1)
                neg_items.append(negativeItemIdx)
            neg_seq_batch.append(neg_items)

        user_batch = np.array(user_batch).reshape((end_idx - start_idx, 1))
        input_seq_batch = np.array(input_seq_batch)
        pos_seq_batch = np.array(pos_seq_batch)
        neg_seq_batch = np.array(neg_seq_batch)

        end = time.time()
        # self.logger.info("time of collect a batch of data: " + str((end - start)) + " seconds")
        # self.logger.info("batch Id: " + str(batchId))
        feed_dict = {
            self.u_id: user_batch,
            self.input_seq: input_seq_batch,
            self.target_seq_pos: pos_seq_batch,
            self.target_seq_neg: neg_seq_batch,
            self.dropout_keep_placeholder: self.dropout_keep
        }

        return feed_dict

    def getPredList_ByUserIdxList(self, user_idices):
        end0 = time.time()
        # build test batch
        input_seq = []
        target_seq = []

        for userIdx in user_idices:
            input_seq.append(self.user_pred_sequences[userIdx])
            target_seq.append(self.evalItemsForEachUser[userIdx])

        batch_u = np.array(user_idices).reshape((-1, 1))
        input_seq = np.array(input_seq)
        target_seq = np.array(target_seq)

        end1 = time.time()

        predList = self.sess.run(self.r_pred, feed_dict={
            self.u_id: batch_u,
            self.input_seq: input_seq,
            self.pred_seq: target_seq,
        })
        end2 = time.time()

        output_lists = []
        for i in range(len(user_idices)):
            recommendList = {}
            start = i * self.eval_item_num
            end = start + self.eval_item_num
            for j in range(end-start):
                recommendList[target_seq[i][j]] = predList[i][j]
            sorted_RecItemList = sorted(recommendList, key=recommendList.__getitem__, reverse=True)[0:self.topN]
            output_lists.append(sorted_RecItemList)
        end3 = time.time()

        return output_lists, end1 - end0, end2 - end1, end3 - end2
