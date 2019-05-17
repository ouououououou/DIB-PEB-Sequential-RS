import random
import logging
import os.path
import time
import csv
from operator import itemgetter, attrgetter, methodcaller
import tensorflow as tf
import numpy as np


class bicluster:
    def __init__(self, vec, itemType=None, itemIdx=None):
        self.vec = vec
        self.itemType = itemType
        self.itemIdx = itemIdx


class BasicDataModel:

    def __init__(self, config):
        self.maxCodeLength = 100
        self.maxPathLength = 100
        self.fileName = config['fileName']
        self.inputPath = './dataset/processed_datasets/' + self.fileName
        self.outputPath = '../dataset/processed_datasets/' + self.fileName
        self.eval_item_num = config['eval_item_num']
        self.training_sequence_length = config['input_length']
        self.test_sparse_user = config['test_sparse_user']
        self.merge_sparse_user = config['merge_sparse_user']
        self.logger = self.initialize_logger('./log')
        self.trainSet = []
        self.validSet = []
        self.testSet = []
        self.wholeDataSet = []
        self.trainSize = 0
        self.validSize = 0
        self.testSize = 0
        self.numUser = 0
        self.numItem = 0
        self.numWord = 0
        self.numNode = 0
        self.familiar_user_num = config['familiar_user_num']
        self.userIdSet = set()
        self.itemIdSet = set()
        "都是字典，user_items_train, user_items_test"
        self.user_items_train = {}
        self.user_items_train_paded = {}
        self.user_items_test = {}
        self.user_rating_scale = {}
        self.max_codelen = 0
        self.testMatrix = {}
        self.trainMatrix = []
        self.similarityMatrix = []
        self.trainType = config['trainType']
        self.splitterType = config['splitterType']
        self.need_process_data = config['need_process_data']

        self.ratingScaleSet = set()
        self.userSet = set()
        self.itemSet = set()

        self.userIdToUserIdx = {}
        self.itemIdToItemIdx = {}
        self.itemIdxToItemInfor = {}
        self.userIdxToUserId = {}
        self.itemIdxToItemId = {}
        self.itemsInTestSet = set()
        self.itemIdxToPastUserIdx = {}
        self.itemIdxToPastUserTimePosition = {}
        self.evalItemsForEachUser = {}
        self.user_items = {}
        self.sparse_user_items = {}
        self.contains_time = False
        self.increaseTestcase = False
        self.useRating = config['useRating']
        self.csv = config['csv']
        self.khsoft = config['khsoft']
        if config['goal'] == 'rating:':
            self.threshold = -1
        else:
            self.threshold = config['threshold']

        random.seed(123)

    def readData(self):
        trainPath = self.inputPath + '/' + self.splitterType + '/train.txt'
        validPath = self.inputPath + '/' + self.splitterType + '/valid.txt'
        testPath = self.inputPath + '/' + self.splitterType + '/test.txt'

        fullPath = self.inputPath + '/' + self.splitterType + '/full.txt'
        # read train data
        basicUserIdx = 1
        basicItemIdx = 0
        self.userIdToUserIdx[0] = 'pad'
        self.userIdxToUserId['pad'] = 0

        "添加pad item ， itemIdx为0"
        """
        self.itemIdxToItemId[0] = 'pad'
        self.itemIdToItemIdx['pad'] = 0
        self.itemIdxToItemInfor[0] = {'ItemFre': 0, 'path': [], 'code': [], 'len': 0}
        basicItemIdx += 1
        """
        user_in_testSet = set()
        record_list = []
        with open(fullPath) as fullFile:
            for line in fullFile:
                record = line.split(' ')
                record_list.append((record[0], record[1], float(record[2]), float(record[3])))

            record_list.sort(key=lambda x: x[3])

            for records in record_list:
                userId, itemId, rating = records[0], records[1], records[2]
                if userId not in self.userIdToUserIdx:
                    userIdx = basicUserIdx
                    self.userIdToUserIdx[userId] = userIdx
                    self.userIdxToUserId[userIdx] = userId
                    basicUserIdx += 1
                else:
                    userIdx = self.userIdToUserIdx[userId]

                if itemId not in self.itemIdToItemIdx:
                    itemIdx = basicItemIdx
                    afterPadItemIdx = itemIdx + 1
                    self.itemIdToItemIdx[itemId] = itemIdx
                    self.itemIdxToItemId[itemIdx] = itemId
                    self.itemIdxToItemInfor[itemIdx] = {'ItemFre': 1, 'path': [], 'code': [], 'len': 0}
                    "结构为dict{（key（idx）, {itemFre, path[], code[], len}）}"
                    self.itemIdxToPastUserIdx[afterPadItemIdx] = [userIdx]
                    self.itemIdxToPastUserTimePosition[afterPadItemIdx] = {userIdx: 0}
                    basicItemIdx += 1
                else:
                    itemIdx = self.itemIdToItemIdx[itemId]
                    afterPadItemIdx = itemIdx + 1
                    self.itemIdxToItemInfor[itemIdx]['ItemFre'] += 1
                    self.itemIdxToPastUserTimePosition[afterPadItemIdx][userIdx] = len(self.itemIdxToPastUserIdx[afterPadItemIdx])
                    self.itemIdxToPastUserIdx[afterPadItemIdx].append(userIdx)

        with open(trainPath) as trainFile:
            for line in trainFile:
                record = line.split(' ')
                userId, itemId, rating = record[0], record[1], float(record[2])

                if userId not in self.userIdToUserIdx:
                    userIdx = basicUserIdx
                    self.userIdToUserIdx[userId] = userIdx
                    self.userIdxToUserId[userIdx] = userId
                    basicUserIdx += 1
                else:
                    userIdx = self.userIdToUserIdx[userId]

                if itemId not in self.itemIdToItemIdx:
                    itemIdx = basicItemIdx
                    self.itemIdToItemIdx[itemId] = itemIdx
                    self.itemIdxToItemId[itemIdx] = itemId
                    self.itemIdxToItemInfor[itemIdx] = {'ItemFre': 1, 'path': [], 'code': [], 'len': 0}
                    "结构为dict{（key（idx）, {itemFre, path[], code[], len}）}"
                    basicItemIdx += 1
                else:
                    itemIdx = self.itemIdToItemIdx[itemId]
                    self.itemIdxToItemInfor[itemIdx]['ItemFre'] += 1

                if self.threshold < 0:
                    pass
                elif rating > self.threshold:
                    if self.useRating:
                        rating = rating
                    else:
                        rating = 1.0
                else:
                        rating = 0.0
                "实际计算时都是采用Idx计算的，而不是采用Id"
                self.user_items_train.setdefault(userIdx, [])
                self.user_items_train[userIdx].append(itemIdx)
                self.ratingScaleSet.add(rating)
                self.trainSet.append([userIdx, itemIdx, rating])

        # read valid data
        with open(validPath) as validFile:
            for line in validFile:
                record = line.split(' ')
                userId, itemId, rating = record[0], record[1], float(record[2])

                if userId not in self.userIdToUserIdx:
                    userIdx = basicUserIdx
                    self.userIdToUserIdx[userId] = userIdx
                    self.userIdxToUserId[userIdx] = userId
                    basicUserIdx += 1
                else:
                    userIdx = self.userIdToUserIdx[userId]

                if itemId not in self.itemIdToItemIdx:
                    itemIdx = basicItemIdx
                    self.itemIdToItemIdx[itemId] = itemIdx
                    self.itemIdxToItemId[itemIdx] = itemId
                    self.itemIdxToItemInfor[itemIdx] = {'ItemFre': 1, 'path': [], 'code': [], 'len': 0}
                    "结构为dict{（key（idx）, {itemFre, point[], code[], len}）}"
                    basicItemIdx += 1
                else:
                    itemIdx = self.itemIdToItemIdx[itemId]
                    self.itemIdxToItemInfor[itemIdx]['ItemFre'] += 1

                if self.threshold < 0:
                    pass
                elif rating > self.threshold:
                    if self.useRating:
                        rating = rating
                    else:
                        rating = 1.0
                else:
                    rating = 0.0

                self.user_items_train.setdefault(userIdx, [])
                "user_item_train 中存放的是字典，每个user对应一个 item list"
                self.user_items_train[userIdx].append(itemIdx)
                self.ratingScaleSet.add(rating)
                "validSet 中存放的是record，一条条的评分记录"
                self.validSet.append([userIdx, itemIdx, rating])

        # read test data
        with open(testPath) as testFile:
            for line in testFile:
                record = line.split(' ')
                userId, itemId, rating = record[0], record[1], float(record[2])

                if userId not in self.userIdToUserIdx:
                    userIdx = basicUserIdx
                    self.userIdToUserIdx[userId] = userIdx
                    self.userIdxToUserId[userIdx] = userId
                    basicUserIdx += 1
                else:
                    userIdx = self.userIdToUserIdx[userId]

                if itemId not in self.itemIdToItemIdx:
                    itemIdx = basicItemIdx
                    self.itemIdToItemIdx[itemId] = itemIdx
                    self.itemIdxToItemId[itemIdx] = itemId
                    self.itemIdxToItemInfor[itemIdx] = {'ItemFre': 1, 'path': [], 'code': [], 'len': 0}
                    "结构为dict{ key: idx , item: {itemFre, path[], code[], len} }"
                else:
                    itemIdx = self.itemIdToItemIdx[itemId]
                    self.itemIdxToItemInfor[itemIdx]['ItemFre'] += 1

                user_in_testSet.add(userIdx)

                # no binarize for testset
                # if self.threshold < 0:
                #     pass
                # elif rating > self.threshold:
                #     rating = 1.0
                # else:
                #     rating = 0.0
                if userIdx not in self.user_items_test.keys():
                    self.user_items_test[userIdx] = []
                self.user_items_test[userIdx].append(itemIdx)
                self.testSet.append([int(userIdx), int(itemIdx), float(rating)])
                self.itemsInTestSet.add(itemIdx)

                if userIdx not in self.evalItemsForEachUser:
                    self.evalItemsForEachUser[userIdx] = set()

        if self.trainType == 'test':
            self.trainSet = self.trainSet + self.validSet
        else:
            self.testSet = self.validSet

        self.logger.info('Num user in testSet: ' + str(len(user_in_testSet)))

        self.trainSize = len(self.trainSet)
        self.testSize = len(self.testSet)
        self.numUser = len(self.userIdToUserIdx)
        self.numItem = len(self.itemIdToItemIdx)

        self.trainMatrix = np.zeros(shape=[self.numUser, self.numItem], dtype=np.float32)

        for line in self.trainSet:
            userIdx, itemIdx, rating = line
            self.trainMatrix[userIdx, itemIdx] = rating

        "处理训练的评分矩阵"
        "self.trainMatrix = self.ratingProcessMean(self.trainMatrix)"
        "self.similarityMatrix = np.corrcoef(self.trainMatrix, rowvar=False)"

        "构建Huffman树并保存进self.itemIdxToItemInfor"
        self.createHuffmanTree(self.itemIdxToItemInfor)

        "self.createHuffmanTreeByClusting(self.itemIdxToItemInfor, self.trainMatrix.transpose())"

        # print userIdx to userId
        userIdx_id_output_path = self.inputPath + '/' + self.splitterType + '/user_idx_id.txt'
        userIdx_id_output_outputLines = []
        userIdx_id_output_outputLines.append("idx   id\n")
        for userId in self.userIdToUserIdx:
            userIdx = self.userIdToUserIdx[userId]
            userIdx_id_output_outputLines.append(str(userIdx) + "   " + str(userId) + "\n")
        with open(userIdx_id_output_path, 'w') as userIdx_id_output_file:
            userIdx_id_output_file.writelines(userIdx_id_output_outputLines)

        # print itemIdx to itemId
        itemIdx_id_output_path = self.inputPath + '/' + self.splitterType + '/item_idx_id.txt'
        itemIdx_id_outputLines = []
        itemIdx_id_outputLines.append("idx  id\n")
        for itemId in self.itemIdToItemIdx:
            itemIdx = self.itemIdToItemIdx[itemId]
            itemIdx_id_outputLines.append(str(itemIdx) + "  " + str(itemId + "\n"))
        with open(itemIdx_id_output_path, 'w') as itemIdx_id_output_file:
            itemIdx_id_output_file.writelines(itemIdx_id_outputLines)

        # print itemInfor to itemIdx
        itemInfor_idx_output_path = self.inputPath + '/' + self.splitterType + '/itemInfor_idx.txt'
        itemInfor_idx_outputLines = []
        itemInfor_idx_outputLines.append("idx:    ItemFre:   path:                  code:                  len: \n")
        "注意dict的key值有可能是整型1也有可能是字符型 '1' "
        for itemIdx in self.itemIdxToItemInfor:
            itemInfor = self.itemIdxToItemInfor[itemIdx]
            itemFre   = str(itemInfor['ItemFre'])
            itemPath  = ''.join([str(i) for i in itemInfor['path']])
            itemCode  = ''.join([str(i) for i in itemInfor['code']])
            itemCodelen  = str(itemInfor['len'])
            itemInfor_idx_outputLines.append(str(itemIdx) + "  " + itemFre + "  " + itemPath + "  " +
                                             itemCode + "  " +
                                             itemCodelen + "\n")
        with open(itemInfor_idx_output_path, 'w') as itemInfor_idx_output_file:
            itemInfor_idx_output_file.writelines(itemInfor_idx_outputLines)

    def pearson(self, v1, v2):
        sum1 = sum(v1)
        sum2 = sum(v2)

        sum1Sq = sum([pow(v, 2) for v in v1])
        sum2Sq = sum([pow(v, 2) for v in v2])

        pSum = sum([v1[i] * v2[i] for i in range(len(v1))])

        num = pSum - ((sum1 * sum2) / len(v1))
        den = np.sqrt((sum1Sq - pow(sum1, 2) / len(v1)) * (sum2Sq - pow(sum2, 2) / len(v1)))

        if den == 0:
            return 0.0
        return 1.0 - num / den

    def createHuffmanTree(self, itemIdxToItemInfor):
        sort_itemIdxToItemInfor = sorted(itemIdxToItemInfor.items(), key=lambda x: x[1]['ItemFre'], reverse=True)
        "path存储的是临时的路径，code存储的是暂时的Huffman编码"
        path = [-1 for i in range(self.maxPathLength)]
        code = [-1 for i in range(self.maxCodeLength)]
        "count数组中前vocab_size存储的是每一个词的对应的词频，后面初始化的是很大的数，用来存储生成节点的频数"
        "parent_node数组中前vocab_size存储的是每一个词的对应的父节点，后面初始化的是0，用来存储生成节点的父节点"
        "原有的word2vec中，经过排序后，数组的下标就对应与数组中元素的Idx，而在Item中则不是，我们不能根据count数组的小标"
        "而是要找到节点的Idx路径"
        item_size = len(itemIdxToItemInfor)
        self.numNode = item_size - 1
        num_len = item_size * 2 + 1
        count = [0 for i in range(num_len)]
        binary = [0.0 for i in range(num_len)]
        parent_node = [0 for i in range(num_len)]

        for a in range(item_size):
            'sort_itemIdxToItemInfor是个list， 元素为tuple：(Idx:, { ItemFre: 7, path:, code:, len:})'
            count[a] = sort_itemIdxToItemInfor[a][1]['ItemFre']
        for a in range(item_size, num_len - 1):
            count[a] = 1e7

        pos1 = item_size - 1
        pos2 = item_size

        '生成n-1个叶子节点，要通过Idx来找节点，要转换成Idx版本，与原有的C不一样'
        for a in range(item_size - 1):
            if (pos1>= 0):
                if (count[pos1] < count[pos2]):
                    min1i = pos1
                    pos1-=1
                else:
                    min1i = pos2
                    pos2+=1
            else:
                min1i = pos2
                pos2+=1

            if pos1 >= 0:
                if (count[pos1] < count[pos2]):
                    min2i = pos1
                    pos1-=1
                else:
                    min2i = pos2
                    pos2+=1
            else:
                min2i = pos2
                pos2+=1

            count[item_size + a] = count[min1i] + count[min2i]
            "存储算法生成的中间节点的词频"
            parent_node[min1i] = item_size + a
            "存储父节点的编号：为叶子节点数目+a，a表示当前生成第a个节点"
            parent_node[min2i] = item_size + a
            binary[min2i] = 1.0
            "存储两个节点中 词频大的节点定为1，代表负类"

        "进行Huffamn编码，对于n个叶子类别节点"
        for a in range(item_size):
            b = a
            i = 0
            "找到一个word的huffman编码,叶子节点也要加进编码中,而最上层的生成节点不加入编码中"
            code[i] = binary[b]
            path[i] = sort_itemIdxToItemInfor[b][0]
            i += 1
            while(1):
                b = parent_node[b]
                if b == item_size * 2 - 2:
                    break
                code[i] = binary[b]
                path[i] = b
                i += 1

                "到达根节点所在索引，结束"
            "一开始生成的编码和父节点路径都是由底向上的，后续需要将其翻转，变化成自顶向下的编码和节点路径"
            itemNewIdx = sort_itemIdxToItemInfor[a][0]
            itemIdxToItemInfor[itemNewIdx]['len'] = i
            itemIdxToItemInfor[itemNewIdx]['path'].append(item_size - 2)
            "路径第一个节点为：根节点"
            for b in range(i):
                itemIdxToItemInfor[itemNewIdx]['code'].append(code[i-b-1])
            for c in range(i-1):
                itemIdxToItemInfor[itemNewIdx]['path'].append(path[i-c-1] - item_size)
                "生成的第i个节点，记录的是从根结点到叶子节点的路径，path中将叶子节点也记录进来了，但是在变量更新时我们"
                "根据codelen来更新，这样path的最后一个节点也就是叶子节点将不被我们考虑在其中，我们只要关注非叶子节点即可，"
                "同时，到达一个叶子也就是item节点的过程中不会经过其他的item节点"

        sort_itemIdxToItemInfor = sorted(itemIdxToItemInfor.items(), key=lambda x: x[1]['len'], reverse=True)
        self.max_codelen = sort_itemIdxToItemInfor[0][1]['len']

    def createHuffmanTreeByClusting(self, itemIdxToItemInfor, itemMatrix):
        lowestpair = None

        "path存储的是临时的路径，code存储的是暂时的Huffman编码"
        path = [-1 for i in range(self.maxPathLength)]
        code = [-1 for i in range(self.maxCodeLength)]

        item_size = len(itemIdxToItemInfor)
        self.numNode = item_size - 1
        num_len = item_size * 2 + 1

        binary = [0.0 for i in range(num_len)]
        parent_node = [0 for i in range(num_len)]
        j = 0

        # 最开始的聚类就是数据集中的行
        clusts = [bicluster(vec=itemMatrix[i], itemIdx=i) for i in range(len(itemIdxToItemInfor))]

        while len(clusts) > 1:
            closest = self.pearson(clusts[0].vec, clusts[1].vec)
            # 遍历每一个配对，寻找最小距离
            for i in range(len(clusts) - 1):
                for j in range(i + 1, len(clusts)):
                    # 用distances来缓存距离的计算值
                    d = self.pearson(clusts[i].vec, clusts[j].vec)
                    # 寻找最相似的两个群
                    if d < closest:
                        closest = d
                        lowestpair = (i, j)
            bic1, bic2 = lowestpair
            # 计算两个聚类的平均值
            mergevec = [((clusts[bic1].vec[i] + clusts[bic2].vec[i]) / 2.0) for i
                        in range(len(clusts[bic1].vec))]
            # 建立新的聚类
            newcluster = bicluster(mergevec, itemIdx=item_size + j)
            parent_node[clusts[bic1].itemIdx] = item_size + j
            "存储父节点的编号：为叶子节点数目+a，a表示当前生成第a个节点"
            parent_node[clusts[bic2].itemIdx] = item_size + j
            binary[clusts[bic2].itemIdx] = 1.0
            del clusts[bic2]
            del clusts[bic1]
            clusts.append(newcluster)
            j += 1

        "进行Huffamn编码，对于n个叶子类别节点"
        for a in range(item_size):
            b = a
            i = 0
            "找到一个word的huffman编码,叶子节点也要加进编码中,而最上层的生成节点不加入编码中"
            code[i] = binary[b]
            path[i] = b
            i += 1
            while(1):
                b = parent_node[b]
                if b == item_size * 2 - 2:
                    break
                code[i] = binary[b]
                path[i] = b
                i += 1

                "到达根节点所在索引，结束"
            "一开始生成的编码和父节点路径都是由底向上的，后续需要将其翻转，变化成自顶向下的编码和节点路径"
            itemNewIdx = a
            itemIdxToItemInfor[itemNewIdx]['len'] = i
            itemIdxToItemInfor[itemNewIdx]['path'].append(item_size - 2)
            "路径第一个节点为：根节点"
            for b in range(i):
                itemIdxToItemInfor[itemNewIdx]['code'].append(code[i-b-1])
            for c in range(i-1):
                itemIdxToItemInfor[itemNewIdx]['path'].append(path[i-c-1] - item_size)
                "生成的第i个节点，记录的是从根结点到叶子节点的路径，path中将叶子节点也记录进来了，但是在变量更新时我们"
                "根据codelen来更新，这样path的最后一个节点也就是叶子节点将不被我们考虑在其中，我们只要关注非叶子节点即可，"
                "同时，到达一个叶子也就是item节点的过程中不会经过其他的item节点"

        sort_itemIdxToItemInfor = sorted(itemIdxToItemInfor.items(), key=lambda x: x[1]['len'], reverse=True)
        self.max_codelen = sort_itemIdxToItemInfor[0][1]['len']

    def generateEvalItemsForEachUser(self):
        "当测试的用户的item数量不足时，随机从不在train set中的item中选出足够的测试item"
        for userIdx in self.evalItemsForEachUser:
            "test和train中的itemIdx已经全部加1了"
            itemsToEval = set(self.user_items_test[userIdx])
            itemsInTrain = self.user_items_train[userIdx]
            "最后用于评价的 item list中一定包括了正样本item"
            while len(itemsToEval) < self.eval_item_num:
                newItemIdx = random.randint(1, self.numItem-1)
                if newItemIdx not in itemsInTrain:
                    if newItemIdx not in itemsToEval:
                        itemsToEval.add(newItemIdx)
            "测试时也是一个字典，每个user对应一个item list，list中的item都没在train datasets 中出现过"
            self.evalItemsForEachUser[userIdx] = list(itemsToEval)

    def ratingProcessMean(self, ratingMatrix, eq=1e-8):
        for i in range(self.numUser):
            ave_rating = ratingMatrix[i].sum() / (len(ratingMatrix[i].nonzero()) + eq)  # 记得只取非零的
            for j in range(self.numItem):
                if ratingMatrix[i][j] <= 0:
                    continue
                else:
                    ratingMatrix[i][j] -= ave_rating
        return ratingMatrix

    def ratingProcessVar(self, ratingMatrix, eq=1e-8):
        for i in range(self.numUser):
            ave_rating = ratingMatrix[i].sum() / (len(ratingMatrix[i].nonzero()) + eq)  # 记得只取非零的
            var_rating = ratingMatrix[i].var()
            for j in range(self.numItem):
                if ratingMatrix[i][j] <= 0:
                    continue
                else:
                    ratingMatrix[i][j] = (ratingMatrix[i][j] - ave_rating) / (var_rating + eq)
        return ratingMatrix

    def printInfo(self):
        self.logger.info('dataset: ' + str(self.fileName))
        self.logger.info('trainType: ' + str(self.trainType))
        self.logger.info('trainSize: ' + str(self.trainSize))
        self.logger.info('testSize: ' + str(self.testSize))
        self.logger.info('numUser: ' + str(self.numUser))
        self.logger.info('numItem: ' + str(self.numItem))
        self.logger.info('numRating:' + str(self.trainSize + self.testSize))
        self.logger.info('ratingScale: ' + str(self.ratingScaleSet))
        self.logger.info('density: ' + str((self.trainSize + self.testSize) / (self.numItem * self.numUser)))
        self.logger.info('Num item in testSet: ' + str(len(self.itemsInTestSet)))
        self.logger.info('max_codelen: ' + str(self.max_codelen))

    def initialize_logger(self, output_dir):
        logger = logging.getLogger(name='hhh')

        logger.setLevel(logging.DEBUG)

        # create console handler and set level to info
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # create error file handler and set level to error
        handler = logging.FileHandler(os.path.join(output_dir, "error.log"), "w", encoding=None, delay="true")
        handler.setLevel(logging.ERROR)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # create debug file handler and set level to debug
        handler = logging.FileHandler(os.path.join(output_dir, str(time.time()) + '.log'), "w")
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def buildTestMatrix(self):
        for line in self.testSet:
            userIdx, itemIdx, rating = line
            self.testMatrix[userIdx, itemIdx] = rating
        return self.testMatrix

    def listToString(self, list):
        listString = ''
        for i in range(len(list)):
            if i == len(list)-1:
                listString = listString + str(list[i])
            else:
                listString = listString + str(list[i]) + ':'

        return listString

    def pre_process(self):
        if self.csv:
            f = csv.reader(open(self.inputPath + '/' + self.splitterType + '/full.csv', 'r'))
            for line in f:
                userId, itemId, rating, r_time = line
                self.userIdSet.add(userId)
                self.itemIdSet.add(itemId)
                self.wholeDataSet.append([userId, itemId, rating, r_time])
        else:
            f = open(self.inputPath + '/' + self.splitterType + '/full.txt')
            lines = f.readlines()
            # check if there is time column
            first_line = lines[0]

            if len(first_line.strip().split(' ')) == 4:
                self.contains_time = True

            for line in lines:
                record = line.strip().split(' ')

                userId, itemId, rating = record[0], record[1], record[2]

                self.userIdSet.add(userId)
                self.itemIdSet.add(itemId)

                if self.contains_time:
                    self.wholeDataSet.append([userId, itemId, rating, record[3]])
                else:
                    self.wholeDataSet.append([userId, itemId, rating])

        self.numRatingBeforeSplit = len(self.wholeDataSet)
        self.logger.info('raw_numUser:' + str(len(self.userIdSet)))
        self.logger.info('raw_numItem:' + str(len(self.itemIdSet)))
        self.logger.info('raw_numRating:' + str(self.numRatingBeforeSplit))

        user_items = {}
        for record in self.wholeDataSet:
            userId, itemId, rating, timeStamp = record
            if userId not in user_items.keys():
                user_items[userId] = []
            user_items[userId].append([itemId, float(rating), float(timeStamp)])

        for userId, items in user_items.items():
            # get and sort items
            numItems = len(items)
            "按照一定rating 数量来划分"
            if numItems < 10:
                self.sparse_user_items[userId] = items
            else:
                self.user_items[userId] = items

    def split_user_loo(self):
        random.seed(123)
        user_items = {}
        userItemToRating = {}

        # collect all data
        for record in self.wholeDataSet:
            userId, itemId, rating = record[0], record[1], record[2]
            user_items.setdefault(userId, [])
            user_items[userId].append(itemId)
            userItemToRating[userId, itemId] = rating

        # split the data
        trainSet = []
        validSet = []
        testSet = []
        itemsInTrainSet = set()
        usersInTrainSet = set()

        for userId in user_items:
            items = user_items[userId]
            random.shuffle(items)
            numItem = len(items)

            if numItem <= 2:
                for itemId in items:
                    rating = userItemToRating[userId, itemId]
                    trainSet.append([userId, itemId, rating])
            else:
                for itemId in items[: -2]:
                    rating = userItemToRating[userId, itemId]
                    trainSet.append([userId, itemId, rating])
                    itemsInTrainSet.add(itemId)
                    usersInTrainSet.add(userId)
                for itemId in items[-2: -1]:
                    rating = userItemToRating[userId, itemId]
                    validSet.append([userId, itemId, rating])
                for itemId in items[-1:]:
                    rating = userItemToRating[userId, itemId]
                    testSet.append([userId, itemId, rating])

        self.write_split_data(trainSet, validSet, testSet, usersInTrainSet, itemsInTrainSet)

    def write_split_data(self, trainSet, validSet, testSet, usersInTrainSet, itemsInTrainSet):
        trainSetToWrite = []
        validSetToWrite = []
        testSetToWrite = []

        for record in trainSet:
            userId, itemId, rating = record[0], record[1], record[2]
            trainSetToWrite.append(
                str(userId) + " " + str(itemId) + " " + str(rating) + '\n')

        for record in validSet:
            userId, itemId, rating = record[0], record[1], record[2]
            if userId in usersInTrainSet and itemId in itemsInTrainSet:
                validSetToWrite.append(
                    str(userId) + " " + str(itemId) + " " + str(rating) + '\n')

        for record in testSet:
            userId, itemId, rating = record[0], record[1], record[2]
            if userId in usersInTrainSet and itemId in itemsInTrainSet:
                testSetToWrite.append(
                    str(userId) + " " + str(itemId) + " " + str(rating) + '\n')

        self.trainSize = len(trainSetToWrite)
        self.validSize = len(validSetToWrite)
        self.testSize = len(testSetToWrite)
        self.logger.info('trainSize: ' + str(self.trainSize))
        self.logger.info('validSize: ' + str(self.validSize))
        self.logger.info('testSize: ' + str(self.testSize))
        self.numUserAfterSplit = len(usersInTrainSet)
        self.numItemAfterSplit = len(itemsInTrainSet)
        self.logger.info('numUser after split: ' + str(self.numUserAfterSplit))
        self.logger.info('numItem after aplit: ' + str(self.numItemAfterSplit))

        fullOutputPath = self.inputPath + '/' + self.splitterType
        if not os.path.exists(fullOutputPath):
            os.makedirs(fullOutputPath)

        with open(fullOutputPath + '/train.txt', 'w') as trainFile:
            trainFile.writelines(trainSetToWrite)

        with open(fullOutputPath + '/valid.txt', 'w') as validFile:
            validFile.writelines(validSetToWrite)

        with open(fullOutputPath + '/test.txt', 'w') as testFile:
            testFile.writelines(testSetToWrite)

    def split_UserTimeRatio(self):
        random.seed(123)
        # collect all user item - rating review timeStamp mappings
        # split the data
        trainSet = []
        validSet = []
        testSet = []
        fullSet = []
        itemsInTrainSet = set()
        usersInTrainSet = set()
        for userId, user_items in self.user_items.items():
            # get and sort items
            items = sorted(user_items, key=itemgetter(2))
            for itemId, rating, timeStamp in items[: -2]:
                trainSet.append([userId, itemId, rating, timeStamp])
                itemsInTrainSet.add(itemId)
                usersInTrainSet.add(userId)

                if userId not in self.user_rating_scale:
                    self.user_rating_scale[userId] = set()
                self.user_rating_scale[userId].add(rating)
            # items for validSet
            for itemId, rating, timeStamp in items[-2:-1]:
                validSet.append([userId, itemId, rating, timeStamp])
                itemsInTrainSet.add(itemId)
                usersInTrainSet.add(userId)

                if userId not in self.user_rating_scale:
                    self.user_rating_scale[userId] = set()
                self.user_rating_scale[userId].add(rating)

            # items for testSet
            for itemId, rating, timeStamp in items[-1:]:
                testSet.append([userId, itemId, rating, timeStamp])

            for itemId, rating, timeStamp in items:
                fullSet.append([userId, itemId, rating, timeStamp])

        trainSetToWrite = []
        validSetToWrite = []
        testSetToWrite = []
        fullSetToWrite = []
        delete_count = 0

        for record in trainSet:
            userId, itemId, rating, timeStamp = record
            trainSetToWrite.append(
                str(userId) + " " + str(itemId) + " " + str(rating) + " "
                + str(timeStamp) + '\n')

        for record in validSet:
            userId, itemId, rating, timeStamp = record
            validSetToWrite.append(
                str(userId) + " " + str(itemId) + " " + str(rating) + " "
                + str(timeStamp) + '\n')

        for record in testSet:
            userId, itemId, rating, timeStamp = record
            if userId in usersInTrainSet and itemId in itemsInTrainSet:
                testSetToWrite.append(
                    str(userId) + " " + str(itemId) + " " + str(rating) + " "
                    + str(timeStamp) + '\n')
            else:
                delete_count += 1

        for record in fullSet:
            userId, itemId, rating, timeStamp = record
            fullSetToWrite.append(
                str(userId) + " " + str(itemId) + " " + str(rating) + " "
                + str(timeStamp) + '\n')

        self.trainSize = len(trainSetToWrite)
        self.validSize = len(validSetToWrite)
        self.testSize = len(testSetToWrite)
        self.logger.info('trainSize: ' + str(self.trainSize))
        self.logger.info('validSize: ' + str(self.validSize))
        self.logger.info('testSize: ' + str(self.testSize))

        self.numRatingAfterSplit = self.numRatingBeforeSplit - delete_count
        self.logger.info('numRatingAfterSplit: ' + str(self.numRatingAfterSplit))
        self.numUserAfterSplit = len(usersInTrainSet)
        self.numItemAfterSplit = len(itemsInTrainSet)
        self.logger.info('numUser after split: ' + str(self.numUserAfterSplit))
        self.logger.info('numItem after split: ' + str(self.numItemAfterSplit))

        fullOutputPath = self.inputPath + '/' + self.splitterType
        if not os.path.exists(fullOutputPath):
            os.makedirs(fullOutputPath)

        with open(fullOutputPath + '/train.txt', 'w') as trainFile:
            trainFile.writelines(trainSetToWrite)

        with open(fullOutputPath + '/valid.txt', 'w') as validFile:
            validFile.writelines(validSetToWrite)

        with open(fullOutputPath + '/test.txt', 'w') as testFile:
            testFile.writelines(testSetToWrite)

        with open(fullOutputPath + '/full.txt', 'w') as fullFile:
            fullFile.writelines(fullSetToWrite)

    def sparse_split_UserTimeRatio(self):
        random.seed(123)
        # collect all user item - rating review timeStamp mappings
        # split the data
        trainSet = []
        validSet = []
        testSet = []
        fullSet = []
        itemsInTrainSet = set()
        usersInTrainSet = set()
        for userId, user_items in self.sparse_user_items.items():
            # get and sort items
            items = sorted(user_items, key=itemgetter(2))
            for itemId, rating, timeStamp in items:
                fullSet.append([userId, itemId, rating, timeStamp])

            if self.test_sparse_user:
                "过滤了长度不符合测试的user，是长度的最低要求"
                for itemId, rating, timeStamp in items[: -2]:
                        trainSet.append([userId, itemId, rating, timeStamp])
                        itemsInTrainSet.add(itemId)
                        usersInTrainSet.add(userId)

                        if userId not in self.user_rating_scale:
                            self.user_rating_scale[userId] = set()
                        self.user_rating_scale[userId].add(rating)
                        # items for validSet
                for itemId, rating, timeStamp in items[-2:-1]:
                        validSet.append([userId, itemId, rating, timeStamp])
                        itemsInTrainSet.add(itemId)
                        usersInTrainSet.add(userId)

                        if userId not in self.user_rating_scale:
                            self.user_rating_scale[userId] = set()
                        self.user_rating_scale[userId].add(rating)

                        # items for testSet
                for itemId, rating, timeStamp in items[-1:]:
                        testSet.append([userId, itemId, rating, timeStamp])

        trainSetToWrite = []
        validSetToWrite = []
        testSetToWrite = []
        fullSetToWrite = []
        delete_count = 0

        for record in trainSet:
            userId, itemId, rating, timeStamp = record
            trainSetToWrite.append(
                str(userId) + " " + str(itemId) + " " + str(rating) + " "
                + str(timeStamp) + '\n')

        for record in validSet:
            userId, itemId, rating, timeStamp = record
            validSetToWrite.append(
                str(userId) + " " + str(itemId) + " " + str(rating) + " "
                + str(timeStamp) + '\n')

        for record in testSet:
            userId, itemId, rating, timeStamp = record
            if userId in usersInTrainSet and itemId in itemsInTrainSet:
                testSetToWrite.append(
                    str(userId) + " " + str(itemId) + " " + str(rating) + " "
                    + str(timeStamp) + '\n')
            else:
                delete_count += 1

        for record in fullSet:
            userId, itemId, rating, timeStamp = record
            fullSetToWrite.append(
                str(userId) + " " + str(itemId) + " " + str(rating) + " "
                + str(timeStamp) + '\n')

        self.trainSize = len(trainSetToWrite)
        self.validSize = len(validSetToWrite)
        self.testSize = len(testSetToWrite)
        self.logger.info('trainSize: ' + str(self.trainSize))
        self.logger.info('validSize: ' + str(self.validSize))
        self.logger.info('testSize: ' + str(self.testSize))

        self.numRatingAfterSplit = self.numRatingBeforeSplit - delete_count
        self.logger.info('numRatingAfterSplit: ' + str(self.numRatingAfterSplit))
        self.numUserAfterSplit = len(usersInTrainSet)
        self.numItemAfterSplit = len(itemsInTrainSet)
        self.logger.info('numUser after split: ' + str(self.numUserAfterSplit))
        self.logger.info('numItem after split: ' + str(self.numItemAfterSplit))

        fullOutputPath = self.inputPath + '/' + self.splitterType
        if not os.path.exists(fullOutputPath):
            os.makedirs(fullOutputPath)

        with open(fullOutputPath + '/sparse_train.txt', 'w') as trainFile:
            trainFile.writelines(trainSetToWrite)

        with open(fullOutputPath + '/sparse_valid.txt', 'w') as validFile:
            validFile.writelines(validSetToWrite)

        with open(fullOutputPath + '/sparse_test.txt', 'w') as testFile:
            testFile.writelines(testSetToWrite)

        with open(fullOutputPath + '/sparse_full.txt', 'w') as fullFile:
            fullFile.writelines(fullSetToWrite)


    def buildModel(self):
        self.logger.info("\n###### information of DataModel ######\n")
        if self.need_process_data:
            self.pre_process()
            self.split_UserTimeRatio()

        self.readData()
        self.generateEvalItemsForEachUser()
        self.printInfo()








