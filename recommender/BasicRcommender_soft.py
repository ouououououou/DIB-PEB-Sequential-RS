import numpy as np
import tensorflow as tf
import time
import eval.RankingEvaluator
import eval.RatingEvaluator
import os
import random
import math

epsilon = 1e-9

class BasicRecommender_soft:

    def __init__(self, dataModel, config):

        self.config = config
        self.name = 'BasicRecommender'
        tf.set_random_seed(config['random_seed'])
        random.seed(config['random_seed'])

        self.trainSet = np.array(dataModel.trainSet)
        self.testSet = np.array(dataModel.testSet)
        self.testMatrix = dataModel.buildTestMatrix()
        self.trainSize = len(dataModel.trainSet)
        self.testSize = len(dataModel.testSet)
        self.numUser = dataModel.numUser
        self.numItem = dataModel.numItem
        self.numWord = dataModel.numWord
        self.numNode = dataModel.numNode
        self.evalItemsForEachUser = dataModel.evalItemsForEachUser
        self.userIdxToUserId = dataModel.userIdxToUserId
        self.itemIdxToItemId = dataModel.itemIdxToItemId
        self.userIdToUserIdx = dataModel.userIdToUserIdx
        self.itemIdxToItemInfor = dataModel.itemIdxToItemInfor
        self.itemIdxToPath = dataModel.itemIdxToPath
        self.itemIdxToCode = dataModel.itemIdxToCode
        self.NodeMask = dataModel.NodeMask
        self.user_items_train = dataModel.user_items_train
        self.user_items_train_paded = dataModel.user_items_train_paded
        self.user_items_test = dataModel.user_items_test
        self.itemsInTestSet = dataModel.itemsInTestSet
        self.fileName = dataModel.fileName
        self.logger = dataModel.logger
        self.max_codelen = dataModel.max_codelen
        self.userIdxToPreSeq = dataModel.userIdxToPreSeq
        self.r_pred = None
        self.r_label = None
        self.cost = tf.constant(0.0)
        self.min_loss = np.PINF
        self.loss_increas_count = 0
        self.auc = None
        self.precision = None
        self.va_loss = None
        self.va_loss2 = None
        self.neg_num = config['negative_numbers']
        self.save_model = config['save_model']
        self.load_model = config['load_model']
        self.saver = None

        self.best_NDCG = 0
        self.best_NDCG_BatchId = 0
        self.best_NDCG_EpochId = 0

        self.best_AUC = 0
        self.best_AUC_BatchId = 0
        self.best_AUC_EpochId = 0

        self.best_Precision = 0
        self.best_Precision_BatchId = 0
        self.best_Precision_EpochId = 0

        self.best_Recall = 0
        self.best_t0_hit = 0
        self.best_t1_hit = 0
        self.best_t2_hit = 0
        self.best_t3_hit = 0
        self.best_total_hit = 0
        self.best_total_user = 0
        self.best_Recall_BatchId = 0
        self.best_Recall_EpochId = 0

        self.bestNDCG = 0
        self.bestNDCGBatchId = 0
        self.bestNDCGEpochId = 0

        self.bestNDCG = 0
        self.bestNDCGBatchId = 0
        self.bestNDCGEpochId = 0

        self.bestRMSE = 100
        self.bestRatingMetricBatchId = 0
        self.bestRatingMetrixEpochId = 0

        self.optimizer = None
        self.sess = None
        self.seed = 123

        self.last_loss = 0
        self.learn_stop_count = 0

        self.fileName = config['fileName']
        self.outputPath = './dataset/processed_datasets'

        self.learnRate = config['learnRate']
        self.maxIter = config['maxIter']
        self.trainBatchSize = config['trainBatchSize']
        self.testBatchSize = config['testBatchSize']
        self.topN = config['topN']
        self.goal = config['goal']
        self.eval_item_num = config['eval_item_num']
        self.early_stop = config['early_stop']

        self.increaseTestcase = dataModel.increaseTestcase
        self.userIdxToAddItemNum = dataModel.userIdxToAddItemNum
        self.eval_user_lists = self.generate_eval_user_lists()

        if self.trainSize % self.trainBatchSize == 0:
            self.trainBatchNum = int(self.trainSize // self.trainBatchSize)
        else:
            self.trainBatchNum = int(self.trainSize // self.trainBatchSize) + 1

        if self.testSize % self.testBatchSize == 0:
            self.testBatchNum = int(self.testSize // self.testBatchSize)
        else:
            self.testBatchNum = int(self.testSize // self.testBatchSize) + 1

        "初始化的时候直接将test dataset中的正确label传进了eval类"
        self.evalRanking = eval.RankingEvaluator.RankingEvaluator(groundTruthLists=self.user_items_test,
                                                                  user_items_train=self.user_items_train,
                                                                  itemInTestSet=self.itemsInTestSet,
                                                                  topK=self.topN,
                                                                  testMatrix=self.testMatrix)
        # self.evalRating = eval.RatingEvaluator.RatingEvaluator(r_label=self.testSet[:, 2:3])

        # user/item embedding
        self.userEmbedding = None
        self.itemEmbedding = None


    def pred_for_a_user(self, W, b, ids, numFactor, input_feature, tar_length):

        split_list = [1] * tar_length
        itemIds = tf.split(ids, split_list, 1)
        preds = []

        for itemId in itemIds:

            item_embeddings = tf.reshape(tf.nn.embedding_lookup(W, itemId), [-1, numFactor])
            item_bias = tf.reshape(tf.nn.embedding_lookup(b, itemId), [-1, 1])
            dotproduct = tf.multiply(item_embeddings, input_feature)
            dotproduct = tf.reshape(dotproduct, shape=[-1, numFactor])
            pred = tf.reduce_sum(dotproduct, 1, keep_dims=True) + item_bias
            preds.append(pred)

        return tf.concat(preds, axis=1)

    def pred_for_a_user_no_bias(self, W, ids, numFactor, input_feature, tar_length):

        split_list = [1] * tar_length
        itemIds = tf.split(ids, split_list, 1)
        preds = []

        for itemId in itemIds:

            item_embeddings = tf.reshape(tf.nn.embedding_lookup(W, itemId), [-1, numFactor])

            dotproduct = tf.multiply(item_embeddings, input_feature)
            dotproduct = tf.reshape(dotproduct, shape=[-1, numFactor])
            pred = tf.reduce_sum(dotproduct, 1, keep_dims=True)

            preds.append(pred)

        return tf.concat(preds, axis=1)

    def pred_for_a_user_item(self, itemEmbedding, itemBias, numFactor, user_input_feature, itemId):

        item_embeddings = tf.reshape(tf.nn.embedding_lookup(itemEmbedding, itemId), [-1, numFactor])
        item_bias = tf.reshape(tf.nn.embedding_lookup(itemBias, itemId), [-1, 1])
        dotproduct = tf.multiply(item_embeddings, user_input_feature)
        dotproduct = tf.reshape(dotproduct, shape=[-1, numFactor])
        pred = tf.reduce_sum(dotproduct, 1, keep_dims=True) + item_bias

        return pred



    def buildModel(self):
        pass

    def trainModel(self):
        self.sess = tf.InteractiveSession()

        self.optimizer = tf.train.AdamOptimizer(self.learnRate, name='Adam_optimizer').minimize(self.cost)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

        for epochId in range(self.maxIter):
            start = time.time()
            totalLoss = 0
            for batchId in range(self.trainBatchNum):
                loss = self.trainEachBatch(epochId, batchId)
                totalLoss += loss
            end = time.time()
            self.logger.info("time cost of an epoch:" + str(end - start) + ", totalLoss: " + str(totalLoss))

            if np.isnan(totalLoss):
                self.logger.info("the loss is nan, training stopped.")
                break
            if totalLoss < self.min_loss:
                self.min_loss = totalLoss
            # when the loss doesn't decrease, stop
            if self.early_stop:
                if self.loss_increas_count > 300:
                    break
                # when the performance doesn't increase, stop
                if epochId - self.best_AUC_EpochId > 200 and epochId - self.best_NDCG_EpochId > 200:
                    break

        tf.reset_default_graph()

    def loadModel(self):
        self.sess = tf.InteractiveSession()
        new_saver = tf.train.import_meta_graph("./ksoft_model-5.meta")
        new_saver.restore(self.sess, "./ksoft_model-5")

        self.evaluateRanking(1, 1)

    def trainEachBatch(self, epochId, batchId):
        pass

    def run(self):
        self.printInfo()
        self.buildModel()
        if self.load_model:
            self.loadModel()
        else:
            self.trainModel()

    def getTrainData(self, batchId):
        pass

    def generate_eval_user_lists(self):
        eval_user_lists = []
        test_user_list = list(self.user_items_test.keys())
        idx_range = len(test_user_list)

        if idx_range % self.testBatchSize == 0:
            step_num = idx_range // self.testBatchSize
        else:
            step_num = idx_range // self.testBatchSize + 1

        for i in range(step_num):
            start = self.testBatchSize * i
            end = start + self.testBatchSize
            if end > idx_range:
                end = idx_range
                start = end - self.testBatchSize
            user_idices = test_user_list[start:end]
            "eval_user_lists中分testBatchSize存放了特定长度的userIdx子列表, 如果不够会从end往前数，造成有重复的"
            eval_user_lists.append(user_idices)
        return eval_user_lists

    def generate_eval_user_lists_increase(self):
        eval_user_lists = []
        eval_itemSeq_lists = []
        test_user_list = list(self.user_items_test.keys())
        new_test_user_list = []
        new_add_preseq_list = []
        idx_range = 0
        for userIdx in test_user_list:
            addNum = len(self.userIdxToPreSeq[userIdx])
            idx_range += addNum
            new_test_user_list.extend([userIdx for i in range(addNum)])
            new_add_preseq_list.extend(self.userIdxToPreSeq[userIdx])

        if idx_range % self.testBatchSize == 0:
            step_num = idx_range // self.testBatchSize
        else:
            step_num = idx_range // self.testBatchSize + 1

        for i in range(step_num):
            start = self.testBatchSize * i
            end = start + self.testBatchSize
            if end > idx_range:
                user_idices = [0 for i in range(self.testBatchSize)]
                user_idices[:(idx_range - start)] = new_test_user_list[start:idx_range]
                pre_idices = [[0 for i in range(5)] for j in range(self.testBatchSize)]
                pre_idices[:(idx_range - start)] = new_add_preseq_list[start:idx_range]
            else:
                user_idices = new_test_user_list[start:end]
                pre_idices = new_add_preseq_list[start:end]
            "会有一部分重复的user idx放入了测试集中"
            "eval_user_lists中分testBatchSize存放了特定长度的userIdx子列表, 如果不够会从end往前数，造成有重复的"
            eval_user_lists.append(user_idices)
            eval_itemSeq_lists.append(pre_idices)
        return eval_user_lists, eval_itemSeq_lists

    def evaluateRanking(self, epochId, batchId):

        userPredLists = {}
        userIdxInTestSet = set()
        start = time.time()

        packTime_total = 0
        runTime_total = 0
        sortTime_total = 0
        seq_index = 0

        for user_list in self.eval_user_lists:
            "user_list中存放了userIdx子列表，shape 为 testBatchSize，user_pred_lists中存放了预测出的概率最高的10个itemIdx"
            if self.increaseTestcase:
                preSeq_list = self.eval_itemSeq_lists[seq_index:(seq_index + self.testBatchSize)]
                user_pred_lists, packTime, runTime, sortTime = self.getPredList_ByUserIdxList(user_list, preSeq_list)
                seq_index += self.testBatchSize
                packTime_total += packTime
                runTime_total += runTime
                sortTime_total += sortTime
                "userPredLists[userIdx] 中存放了每个用户概率最高的十个itemIdx"
                "userPredList是个字典，重复的userIdx会对应于同一个key，消除了重复情况"
                for i in range(len(user_list)):
                    userIdx = user_list[i]
                    pred_list = user_pred_lists[i]
                    if userIdx in userIdxInTestSet:
                        if len(userPredLists[userIdx]) >= (self.userIdxToAddItemNum[userIdx]+1):
                            continue
                        else:
                            userPredLists[userIdx].append(pred_list)
                    else:
                        userPredLists[userIdx] = [pred_list]
            else:
                user_pred_lists, packTime, runTime, sortTime = self.getPredList_ByUserIdxList(user_list)
                packTime_total += packTime
                runTime_total += runTime
                sortTime_total += sortTime
                "userPredLists[userIdx] 中存放了每个用户概率最高的十个itemIdx"
                "userPredList是个字典，重复的userIdx会对应于同一个key，消除了重复情况"
                for i in range(len(user_list)):
                    userIdx = user_list[i]
                    pred_list = user_pred_lists[i]
                    userPredLists[userIdx] = pred_list

        end = time.time()

        self.logger.info("generate recList time cost: %.4f" % (end - start))
        self.logger.info("packTime: %.4f, runTime: %.4f, sortTime: %.4f" %
                         (packTime_total, runTime_total, sortTime_total))

        "将预测出的TopK个item放入eval函数中，与在函数初始化时就已经存在的groundtruth比较计算指标" \
        "userPredLists是个字典"
        self.evalRanking.setPredLists(userPredLists)

        newNDCG, ndcgTime = self.evalRanking.calNDCG()
        newAUC, aucTime = self.evalRanking.calAUC()
        newPrecision, precisionTime = self.evalRanking.calPrecision()
        newRecall, recallTime, r_0, r_1, r_2, r_3, total_hit, total_user = self.evalRanking.calRecall()

        self.logger.info("Recall: " + str([newRecall, recallTime]))
        self.logger.info("Precision: " + str([newPrecision, precisionTime]))
        self.logger.info("AUC: " + str([newAUC, aucTime]))
        self.logger.info("NDCG: " + str([newNDCG, ndcgTime]))
        self.logger.info("t0_hit  " + str([r_0[0]]) + "   " + str([r_0[1]]) + " / " + str([r_0[2]]))
        self.logger.info("t1_hit  " + str([r_1[0]]) + "   " + str([r_1[1]]) + " / " + str([r_1[2]]))
        self.logger.info("t2_hit  " + str([r_2[0]]) + "   " + str([r_2[1]]) + " / " + str([r_2[2]]))
        self.logger.info("t3_hit  " + str([r_3[0]]) + "   " + str([r_3[1]]) + " / " + str([r_3[2]]))
        self.logger.info("total_hit: " + str([total_hit]))
        self.logger.info("total_user: " + str([total_user]))

        self.saveBestResult(newNDCG=newNDCG,
                            newAUC=newAUC,
                            newPrecision=newPrecision,
                            newRecall=newRecall,
                            new_t_0=r_0[0],
                            new_t_1=r_1[0],
                            new_t_2=r_2[0],
                            new_t_3=r_3[0],
                            new_total_hit=total_hit,
                            new_total_user=total_user,
                            epochId=epochId,
                            batchId=batchId)

        self.showBestRankingResult()
        # self.logger.info('saving results of epoch:' + str(epochId) + ', batchId:' + str(batchId))
        # self.printRankResult()
        # self.logger.info('saving results finished')

    def saveBestResult(self, newNDCG, newAUC, newPrecision, newRecall, new_t_0, new_t_1, new_t_2, new_t_3,
                       new_total_hit, new_total_user, epochId, batchId):

        if newNDCG > self.best_NDCG:
            self.best_NDCG = newNDCG
            self.best_NDCG_EpochId = epochId
            self.best_NDCG_BatchId = batchId
            self.saver.save(self.sess, './ksoft_model', global_step=epochId)

        if newAUC > self.best_AUC:
            self.best_AUC = newAUC
            self.best_AUC_EpochId = epochId
            self.best_AUC_BatchId = batchId
            if self.config['save_model']:
                self.saveWeight()

        if newPrecision > self.best_Precision:
            self.best_Precision = newPrecision
            self.best_Precision_EpochId = epochId
            self.best_Precision_BatchId = batchId

        if newRecall > self.best_Recall:
            self.best_Recall = newRecall
            self.best_t0_hit = new_t_0
            self.best_t1_hit = new_t_1
            self.best_t2_hit = new_t_2
            self.best_t3_hit = new_t_3
            self.best_total_hit = new_total_hit
            self.best_total_user = new_total_user
            self.best_Recall_EpochId = epochId
            self.best_Recall_BatchId = batchId


    def saveWeight(self):
        np.savetxt('./save_model/' + self.config['fileName'] + '-' + self.name + '-user_embed.txt', self.userEmbedding.eval())
        np.savetxt('./save_model/' + self.config['fileName'] + '-' + self.name + '-item_embed.txt', self.itemEmbedding.eval())

    def evaluateRating(self, epochId, batchId):
        start = time.time()
        r_pred = self.getRatingPredictions()
        end = time.time()
        self.logger.info("pred time cost: " + str(end - start))

        self.evalRating.set_r_pred(r_pred)
        rmse, mae, timeCost = self.evalRating.cal_RMSE_and_MAE()
        self.logger.info("(RMSE, MAE, EvalTimeCost)="+str([rmse, mae, timeCost]))

        if rmse < self.bestRMSE:
            self.bestRMSE = rmse
            self.bestRatingMetrixEpochId = epochId
            self.bestRatingMetricBatchId = batchId
        self.showBestRatingResult()

    def printRankResult(self):
        outputLines = []
        for userIdx in self.user_items_test:
            eachLine = ''
            userId = self.userIdxToUserId[userIdx]
            eachLine += (str(userId) + ':')
            itemidices = self.getPredList_ByUserIdx(userIdx)
            for itemIdx in itemidices:
                itemId = self.itemIdxToItemId[itemIdx]
                eachLine += (str(itemId) + ' ')
            eachLine += '\n'
            outputLines.append(eachLine)

        fullOutputPath = self.outputPath + '/' + self.fileName + '/userTimeRatio'
        with open(fullOutputPath + '/result.txt', 'w') as resultFile:
            resultFile.writelines(outputLines)

    def getRatingPredictions(self):
        pass

    def getPredList_ByUserIdx(self, userIdx):
        pass

    def getPredList_ByUserIdxList(self, userIdx):
        pass

    def getPredList_ByUserIdxList_increase(self, userIdx, preSeq_Idx):
        pass

    def getTestData(self):
        pass

    def showBestRankingResult(self):
        self.logger.info("best Precision result: %.4f, batchId: %d, epochId: %d" %
                         (self.best_Precision, self.best_Precision_BatchId, self.best_Precision_EpochId))
        self.logger.info("best Recall result: %.4f, batchId: %d, epochId: %d" %
                         (self.best_Recall, self.best_Recall_BatchId, self.best_Recall_EpochId))
        self.logger.info("best t0: %.4f, t1: %.4f, t2: %.4f, t3: %.4f, batchId: %d, epochId: %d" %
                         (self.best_t0_hit, self.best_t1_hit, self.best_t2_hit, self.best_t3_hit,
                          self.best_Recall_BatchId, self.best_Recall_EpochId))
        self.logger.info("best AUC result: %.4f, batchId: %d, epochId: %d" %
                         (self.best_AUC, self.best_AUC_BatchId, self.best_AUC_EpochId))
        self.logger.info("best NDCG result: %.4f, batchId: %d, epochId: %d" %
                         (self.best_NDCG, self.best_NDCG_BatchId, self.best_NDCG_EpochId))

    def showBestRatingResult(self):
        self.logger.info("best RMSE result: RMSE:" + str(self.bestRMSE) + ", batchId: " + str(self.bestRatingMetricBatchId) + ", epochId: " + str(self.bestRatingMetrixEpochId))

    def printInfo(self):
        self.logger.info("\n###### Recommender Info #########\n")
        self.logger.info("Name: %s" % (self.name))
        self.logger.info("num core: %d" % (os.cpu_count()))
        for key, value in self.config.items():
            self.logger.info("%s = %s" % (str(key), str(self.config[key])))

    def sigmoid(self, x):
        """
        Compute the sigmoid of x

        Arguments:
        x -- A scalar or numpy array of any size

        Return:
        s -- sigmoid(x)
        """

        ### START CODE HERE ### (≈ 1 line of code)
        s = 1.0 / (1.0 + np.exp(-x))
        ### END CODE HERE ###

        return s

    def activ(self, name, tensor):
        if name == 'sigmoid':
            return tf.nn.sigmoid(tensor)
        elif name == 'relu':
            return tf.nn.relu(tensor)
        elif name == 'tanh':
            return tf.nn.tanh(tensor)
        else:
            return tensor

    def squash(self, vector):
        '''Squashing function corresponding to Eq. 1
        Args:
            vector: A tensor with shape [batch_size, vec_len].
        Returns:
            A tensor with the same shape as vector.
        '''
        vec_squared_norm = tf.reduce_sum(tf.square(vector), -1, keep_dims=True)
        scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
        vec_squashed = tf.multiply(vector, scalar_factor)  # element-wise
        return (vec_squashed)











