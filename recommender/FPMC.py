import numpy as np
import tensorflow as tf
import random
from recommender.BasicRcommender_soft import BasicRecommender_soft
import time

class FPMCRecommender(BasicRecommender_soft):

    def __init__(self, dataModel, config):

        super(FPMCRecommender, self).__init__(dataModel, config)

        self.name = 'FPMCRecommender'

        self.numFactor = config['numFactor']
        self.target_length = config['target_length']
        self.factor_lambda = config['factor_lambda']
        self.input_length = config['input_length']
        self.train_users = dataModel.train_users
        self.trainSize = len(self.train_users)

        self.train_users = dataModel.train_users
        self.train_sequences_input = dataModel.train_sequences_input
        self.train_sequences_target = dataModel.train_sequences_target
        self.user_pred_sequences = dataModel.user_pred_sequences

        # placeholders
        self.u_id = tf.placeholder(tf.int32, [self.trainBatchSize, 1])
        self.input_seq = tf.placeholder(tf.int32, [self.trainBatchSize, self.input_length])
        self.target_seq_pos = tf.placeholder(tf.int32, [self.trainBatchSize, self.target_length])
        self.target_seq_neg = tf.placeholder(tf.int32, [self.trainBatchSize, self.target_length])
        self.pred_seq = tf.placeholder(tf.int32, [self.trainBatchSize, self.eval_item_num])

        # user/item embedding
        self.userEmbedding = tf.Variable(tf.random_normal([self.numUser, self.numFactor], 0, 0.1))
        self.itemEmbedding = tf.Variable(tf.random_normal([self.numItem, self.numFactor], 0, 0.1))
        # self.itemBias = tf.Variable(tf.random_normal([self.numItem, 1], 0, 0.1))

        self.sampleSize = self.trainSize

        self.input_seq = tf.placeholder(tf.int32, [self.trainBatchSize, self.input_length])


    def buildModel(self):
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            userEmedding = tf.reshape(tf.nn.embedding_lookup(self.userEmbedding, self.u_id), [-1, self.numFactor])

            input_image = tf.nn.embedding_lookup(self.itemEmbedding, self.input_seq)
            input_image = tf.reshape(input_image, [-1, self.input_length, self.numFactor])
            input_image = tf.reduce_sum(input_image, 1, keep_dims=False) / self.input_length

            itemEmedding_i = tf.reshape(tf.nn.embedding_lookup(self.itemEmbedding, self.target_seq_pos), [-1, self.numFactor])
            itemEmedding_j = tf.reshape(tf.nn.embedding_lookup(self.itemEmbedding, self.target_seq_neg), [-1, self.numFactor])
            # bias_i = tf.reshape(tf.nn.embedding_lookup(self.itemBias, self.i_id), [-1, 1])
            # bias_j = tf.reshape(tf.nn.embedding_lookup(self.itemBias, self.j_id), [-1, 1])

            user_pred_i = tf.reduce_sum(tf.multiply(userEmedding, itemEmedding_i), 1, keep_dims=True)
            user_pred_j = tf.reduce_sum(tf.multiply(userEmedding, itemEmedding_j), 1, keep_dims=True)

            item_pred_i = tf.reduce_sum(tf.multiply(input_image, itemEmedding_i), 1, keep_dims=True)
            item_pred_j = tf.reduce_sum(tf.multiply(input_image, itemEmedding_j), 1, keep_dims=True)

            pred_i = user_pred_i + item_pred_i
            pred_j = user_pred_j + item_pred_j

            predDiff = pred_i - pred_j

            l2_norm = self.factor_lambda * tf.nn.l2_loss(userEmedding) \
                     + self.factor_lambda * tf.nn.l2_loss(itemEmedding_i) \
                     + self.factor_lambda * tf.nn.l2_loss(itemEmedding_j)


            bprLoss = - tf.reduce_sum(tf.log(tf.sigmoid(predDiff)))
            self.cost = bprLoss + l2_norm

            self.r_pred = self.pred_for_a_user_no_bias(
                W=self.itemEmbedding,
                numFactor=self.numFactor,
                input_feature=userEmedding + input_image,
                ids=self.pred_seq,
                tar_length=self.eval_item_num
            )

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

        # lazy operate
        input_seq_batch = self.train_sequences_input[start_idx:end_idx]
        pos_seq_batch = self.train_sequences_target[start_idx:end_idx]

        for userIdx in user_batch:
            neg_items = []
            for i in range(self.target_length):
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
        }

        return feed_dict

    def getPredList_ByUserIdxList(self, user_idices):
        end0 = time.time()
        # build test batch
        input_seq = []
        cluster_seq = []
        target_seq = []

        for userIdx in user_idices:
            cluster_seq.append(1)
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
            for j in range(end - start):
                recommendList[target_seq[i][j]] = predList[i][j]
            sorted_RecItemList = sorted(recommendList, key=recommendList.__getitem__, reverse=True)[0:self.topN]
            output_lists.append(sorted_RecItemList)
        end3 = time.time()

        return output_lists, end1 - end0, end2 - end1, end3 - end2