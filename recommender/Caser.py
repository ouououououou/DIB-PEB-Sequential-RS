import numpy as np
import tensorflow as tf
import random
from recommender.BasicRcommender_soft import BasicRecommender_soft
import time
from component.Conv_Pool import CNN_Pool_Compoment
from component.MLP import MLP

class CaserRecommender(BasicRecommender_soft):

    def __init__(self, dataModel, config):

        super(CaserRecommender, self).__init__(dataModel, config)

        self.train_users = dataModel.train_users
        self.train_sequences_input = dataModel.train_sequences_input
        self.train_sequences_target = dataModel.train_sequences_target
        self.user_pred_sequences = dataModel.user_pred_sequences

        self.trainSize = len(self.train_users)
        self.trainBatchNum = int(self.trainSize // self.trainBatchSize) + 1

        self.name = 'Caser'

        self.numFactor = config['numFactor']
        self.factor_lambda = config['factor_lambda']
        self.input_length = config['input_length']
        self.target_length = config['target_length']
        self.hor_filter_num = config['hor_filter_num']
        self.ver_filter_num = config['ver_filter_num']
        self.dropout_keep = config['dropout_keep']
        self.dropout_user = config['dropout_user']
        self.dropout_item = config['dropout_item']
        self.item_fc_dim = [self.hor_filter_num * self.input_length + self.ver_filter_num * self.numFactor] + config['item_fc_dim']

        # placeholders
        self.u_id = tf.placeholder(tf.int32, [None, 1])
        self.test_u_id = tf.placeholder(tf.int32, [None, 1])
        self.input_seq = tf.placeholder(tf.int32, [None, self.input_length])
        self.target_seq_pos = tf.placeholder(tf.int32, [None, self.target_length])
        self.target_seq_neg = tf.placeholder(tf.int32, [None, self.neg_num])
        self.pred_seq = tf.placeholder(tf.int32, [None, self.eval_item_num])
        self.dropout_keep_placeholder = tf.placeholder_with_default(1.0, shape=())


        # user/item embedding
        self.userEmbedding = tf.Variable(tf.random_normal([self.numUser, self.numFactor], 0, 0.1))
        self.itemEmbedding = tf.Variable(tf.random_normal([self.numItem, self.numFactor], 0, 0.1))

        self.vertical_CNN = CNN_Pool_Compoment(
            filter_num=self.ver_filter_num,
            filter_sizes=[self.input_length],
            wordvec_size=self.numFactor,
            max_review_length=self.input_length,
            word_matrix=self.itemEmbedding,
            output_size=1,
            review_wordId_print=None,
            review_input_print=None,
            cnn_lambda=None,
            dropout_keep_prob=None,
            component_raw_output=None,
            item_pad_num=None,
            name='ver'
        )

        self.horizontal_CNN = CNN_Pool_Compoment(
            filter_num=self.hor_filter_num,
            filter_sizes=[i+1 for i in range(self.input_length)],
            wordvec_size=self.numFactor,
            max_review_length=self.input_length,
            word_matrix=self.itemEmbedding,
            output_size=1,
            review_wordId_print=None,
            review_input_print=None,
            cnn_lambda=None,
            dropout_keep_prob=None,
            component_raw_output=None,
            item_pad_num=None,
            name='hor'
        )

        self.item_MLP = MLP(self.item_fc_dim, dropout_keep=self.dropout_keep_placeholder)

        self.output_fc_W = tf.get_variable(
            name="output_fc_W",
            dtype=tf.float32,
            shape=[self.numItem, self.item_fc_dim[-1] + self.numFactor],
            initializer=tf.contrib.layers.xavier_initializer()
        )

        self.output_fc_b = tf.get_variable(
            name="output_fc_b",
            dtype=tf.float32,
            initializer=tf.constant(0.1, shape=[self.numItem, 1])
        )


    def buildModel(self):
        with tf.variable_scope(tf.get_variable_scope()) as scope:

            userEmedding = tf.reshape(tf.nn.embedding_lookup(self.userEmbedding, self.u_id), [-1, self.numFactor])
            userEmedding = tf.nn.dropout(userEmedding, self.dropout_user)

            test_userEmedding = tf.reshape(tf.nn.embedding_lookup(self.userEmbedding, self.test_u_id),
                                           [-1, self.numFactor])

            ver_embed = self.vertical_CNN.get_vertical_output(self.input_seq)
            hor_embed = self.horizontal_CNN.get_horizontal_output(self.input_seq)

            item_fc_feature = tf.concat([hor_embed, ver_embed], axis=1)

            item_fc_output = self.item_MLP.get_output(feature_input=item_fc_feature)

            user_fc_feature = tf.concat([item_fc_output, userEmedding], axis=1)
            test_user_fc_feature = tf.concat([item_fc_output, test_userEmedding], axis=1)

            self.r_pred = self.pred_for_a_user(
                W=self.output_fc_W,
                b=self.output_fc_b,
                numFactor=self.item_fc_dim[-1] + self.numFactor,
                input_feature=test_user_fc_feature,
                ids=self.pred_seq,
                tar_length=self.eval_item_num
            )

            pos_pred = self.pred_for_a_user(
                W=self.output_fc_W,
                b=self.output_fc_b,
                numFactor=self.item_fc_dim[-1] + self.numFactor,
                input_feature=user_fc_feature,
                ids=self.target_seq_pos,
                tar_length=self.target_length
            )

            neg_pred = self.pred_for_a_user(
                W=self.output_fc_W,
                b=self.output_fc_b,
                numFactor=self.item_fc_dim[-1] + self.numFactor,
                input_feature=user_fc_feature,
                ids=self.target_seq_neg,
                tar_length=self.neg_num
            )

            pos_loss = - tf.reduce_mean(tf.log(tf.nn.sigmoid(pos_pred) + 1e-7))
            neg_loss = - tf.reduce_mean(tf.log(1 - tf.nn.sigmoid(neg_pred) + 1e-7))

            self.cost = pos_loss + neg_loss

    def trainEachBatch(self, epochId, batchId):
        totalLoss = 0
        start = time.time()
        user_batch, input_seq_batch, pos_seq_batch, neg_seq_batch = self.getTrainData(batchId)

        # print(batchId)

        self.optimizer.run(feed_dict={
            self.u_id: user_batch,
            self.input_seq: input_seq_batch,
            self.target_seq_pos: pos_seq_batch,
            self.target_seq_neg: neg_seq_batch,
            self.dropout_keep_placeholder: self.dropout_keep
        })

        loss = self.cost.eval(feed_dict={
            self.u_id: user_batch,
            self.input_seq: input_seq_batch,
            self.target_seq_pos: pos_seq_batch,
            self.target_seq_neg: neg_seq_batch,
            self.dropout_keep_placeholder: self.dropout_keep
        })
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

        user_batch = []
        input_seq_batch = []
        pos_seq_batch = []
        neg_seq_batch = []

        start_idx = batchId * self.trainBatchSize
        end_idx = start_idx + self.trainBatchSize

        if end_idx > self.trainSize:
            end_idx = self.trainSize
            start_idx = end_idx - self.trainBatchSize

        if end_idx == start_idx:
            start_idx = 0
            end_idx = start_idx + self.trainBatchSize

        if end_idx > self.trainSize:
            end_idx = self.trainSize

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

        return user_batch, input_seq_batch, pos_seq_batch, neg_seq_batch

    # def getPredList_ByUserIdxList(self, user_idices):
    #     end0 = time.time()
    #     # build test batch
    #     testBatch_user = []
    #
    #     for i in range(len(user_idices)):
    #         userIdx = user_idices[i]
    #         pred_seq = self.user_pred_sequences[userIdx]
    #         for itemIdx in self.evalItemsForEachUser[userIdx]:
    #             testBatch_user.append([userIdx] + pred_seq + [itemIdx])
    #
    #     testBatch_user = np.array(testBatch_user)
    #
    #     batch_u = testBatch_user[:, 0:1].astype(np.int32)
    #     input_seq = testBatch_user[:, 1:6].astype(np.int32)
    #     target_seq = testBatch_user[:, 6:].astype(np.int32)
    #     end1 = time.time()
    #
    #     predList = self.sess.run(self.r_pred, feed_dict={
    #             self.u_id: batch_u,
    #             self.input_seq: input_seq,
    #             self.pred_seq: target_seq,
    #     })
    #     end2 = time.time()
    #
    #     output_lists = []
    #     for i in range(len(user_idices)):
    #         recommendList = {}
    #         start = i * self.eval_item_num
    #         end = start + self.eval_item_num
    #         for j in range(start, end):
    #             recommendList[target_seq[j][0]] = predList[j][0]
    #         sorted_RecItemList = sorted(recommendList, key=recommendList.__getitem__, reverse=True)[0:self.topN]
    #         output_lists.append(sorted_RecItemList)
    #     end3 = time.time()
    #
    #     return output_lists, end1 - end0, end2 - end1, end3 - end2

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
            self.test_u_id: batch_u,
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

        # def getPredList_ByUserIdx(self, userIdx):
        #     # build test batch
        #     testBatch_user = []
        #
        #     for itemIdx in self.evalItemsForEachUser[userIdx]:
        #         testBatch_user.append([userIdx, itemIdx, 1.0])
        #
        #     batch_u = np.array(testBatch_user)[:, 0:1].astype(np.int32)
        #     batch_v = np.array(testBatch_user)[:, 1:2].astype(np.int32)
        #
        #     predList = self.sess.run(self.r_pred, feed_dict={
        #         self.u_id: batch_u,
        #         self.i_id: batch_v,
        #     })
        #
        #     recommendList = {}
        #     for i in range(len(testBatch_user)):
        #         recommendList[batch_v[i][0]] = predList[i][0]
        #
        #     sorted_RecItemList = sorted(recommendList, key=recommendList.__getitem__, reverse=True)[0:self.topN]
        #
        #     return sorted_RecItemList