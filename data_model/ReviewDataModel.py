import random
import logging
import os.path
import time
import collections
import numpy as np

class ReviewDataModel:

    def __init__(self, config):
        self.fileName = config['fileName']
        self.inputPath = './dataset/processed_datasets/' + self.fileName
        self.trainSet = []
        self.validSet = []
        self.testSet = []
        self.trainSize = 0
        self.validSize = 0
        self.testSize = 0
        self.numUser = 0
        self.numItem = 0
        self.numWord = 0
        self.user_items_train = {}
        self.user_items_train_paded = {}
        self.user_items_test = {}
        self.user_items_reviewVectors = {}
        self.user_review_mask = {}
        self.item_reviewVectors = {}
        self.user_reviewVectors = {}
        self.logger = self.initialize_logger('./log')
        self.user_itemToWordTrain = {}
        self.testMatrix = {}
        self.wordIdToWordVector = {}
        self.nullWordIndex = 0
        self.trainType = config['trainType']
        self.splitterType = config['splitterType']
        self.item_pad_num = config['item_pad_num']
        self.if_full_review = config['if_full_review']
        self.ratingScaleSet = set()
        self.userSet = set()
        self.itemSet = set()
        self.wordSet = set()
        self.maxReviewLength = 0
        self.wordMatrix = []
        self.reviewLengthScale = {
            5: 0,
            10: 0,
            20: 0,
            30: 0,
            40: 0,
            50: 0,
            60: 0,
            70: 0,
            80: 0,
            90: 0,
            100: 0,
            150: 0,
            200: 0,
            250: 0,
            300: 0,
            350: 0,
            400: 0,
            401: 0
        }
        self.reviewLengthScale = collections.OrderedDict(sorted(self.reviewLengthScale.items()))
        self.userIdToUserIdx = {}
        self.itemIdToItemIdx = {}
        self.userIdxToUserId = {}
        self.itemIdxToItemId = {}
        self.itemsInTestSet = set()
        self.evalItemsForEachUser = {}
        if config['goal'] == 'rating:':
            self.threshold = -1
        else:
            self.threshold = config['threshold']
        self.wordVecSize = config['wordVecSize']
        random.seed(123)

    def readData(self):
        trainPath = self.inputPath + '/' + self.splitterType + '/train.txt'
        validPath = self.inputPath + '/' + self.splitterType + '/valid.txt'
        testPath = self.inputPath + '/' + self.splitterType + '/test.txt'
        # read train data
        basicUserIndex = 0
        basicItemIndex = 0
        with open(trainPath) as trainFile:
            for line in trainFile:
                userId, itemId, rating, reviewString, timeStamp = line.split(' ')
                if userId not in self.userIdToUserIdx:
                    self.userIdToUserIdx[userId] = basicUserIndex
                    self.userIdxToUserId[basicUserIndex] = userId
                    self.user_reviewVectors[basicUserIndex] = []
                    basicUserIndex += 1
                if itemId not in self.itemIdToItemIdx:
                    self.itemIdToItemIdx[itemId] = basicItemIndex
                    self.itemIdxToItemId[basicItemIndex] = itemId
                    self.item_reviewVectors[basicItemIndex] = []
                    basicItemIndex += 1
                userId = self.userIdToUserIdx[userId]
                itemId = self.itemIdToItemIdx[itemId]
                rating = float(rating)
                if self.threshold < 0:
                    pass
                elif rating > self.threshold:
                    rating = 1.0
                else:
                    rating = 0.0
                if userId not in self.user_items_train.keys():
                    self.user_items_train[userId] = []
                self.user_items_train[userId].append(itemId)
                self.ratingScaleSet.add(float(rating))
                self.userSet.add(userId)
                self.itemSet.add(itemId)
                self.trainSet.append([userId, itemId, rating])
                self.user_itemToWordTrain[userId, itemId] = []
                sentences = reviewString.split("%")
                for sentence in sentences:
                    words = sentence.split(":")
                    for word in words:
                        if word == '':
                            pass
                        else:
                            self.user_itemToWordTrain[userId, itemId].append(int(word))
                            self.item_reviewVectors[itemId].append(int(word))
                            self.user_reviewVectors[userId].append(int(word))

        # read valid data
        with open(validPath) as validFile:
            for line in validFile:
                userId, itemId, rating, reviewString, timeStamp = line.split(' ')
                if userId not in self.userIdToUserIdx:
                    self.userIdToUserIdx[userId] = basicUserIndex
                    self.userIdxToUserId[basicUserIndex] = userId
                    self.user_reviewVectors[basicUserIndex] = []
                    basicUserIndex += 1
                if itemId not in self.itemIdToItemIdx:
                    self.itemIdToItemIdx[itemId] = basicItemIndex
                    self.itemIdxToItemId[basicItemIndex] = itemId
                    self.item_reviewVectors[basicItemIndex] = []
                    basicItemIndex += 1
                userId = self.userIdToUserIdx[userId]
                itemId = self.itemIdToItemIdx[itemId]
                rating = float(rating)
                if self.trainType == 'test':
                    # binarize
                    if self.threshold < 0:
                        pass
                    elif rating > self.threshold:
                        rating = 1.0
                    else:
                        rating = 0.0

                self.userSet.add(userId)
                self.itemSet.add(itemId)
                if userId not in self.user_items_train.keys():
                    self.user_items_train[userId] = []
                self.user_items_train[userId].append(itemId)
                self.validSet.append([int(userId), int(itemId), rating])
                self.user_itemToWordTrain[userId, itemId] = []
                sentences = reviewString.split("%")
                for sentence in sentences:
                    words = sentence.split(":")
                    for word in words:
                        if word == '':
                            pass
                        else:
                            self.user_itemToWordTrain[userId, itemId].append(int(word))
                            self.item_reviewVectors[itemId].append(int(word))
                            self.user_reviewVectors[userId].append(int(word))
        # read test data
        with open(testPath) as testFile:
            for line in testFile:
                userId, itemId, rating, reviewString, timeStamp = line.split(' ')
                rating = float(rating)
                userId = self.userIdToUserIdx[userId]
                itemId = self.itemIdToItemIdx[itemId]
                # no binarize for testset
                # if self.threshold < 0:
                #     pass
                # elif rating > self.threshold:
                #     rating = 1.0
                # else:
                #     rating = 0.0
                if userId not in self.user_items_test.keys():
                    self.user_items_test[userId] = []
                self.user_items_test[userId].append(itemId)
                self.testSet.append([int(userId), int(itemId), float(rating)])
                self.itemsInTestSet.add(itemId)

                if userId not in self.evalItemsForEachUser:
                    self.evalItemsForEachUser[userId] = set()
                self.evalItemsForEachUser[userId].add(itemId)

        if self.trainType == 'test':
            self.trainSet = self.trainSet + self.validSet
        else:
            self.testSet = self.validSet

        # shuffle the data
        # random.shuffle(self.trainSet)
        # random.shuffle(self.testSet)

        self.trainSize = len(self.trainSet)
        self.testSize = len(self.testSet)
        self.numUser = len(self.userSet)
        self.numItem = len(self.itemSet)

        # count word number
        with open(self.inputPath + '/word2Vectors.txt') as wordVecFile:
            for line in wordVecFile:
                self.numWord += 1

        # print userIdx to userId
        userIdx_id_output_path = self.inputPath + '/' + self.splitterType + '/user_idx_id.txt'
        userIdx_id_output_outputLines = []
        userIdx_id_output_outputLines.append("idx   id\n")
        for userIdx in range(self.numUser):
            userId = self.userIdxToUserId[userIdx]
            userIdx_id_output_outputLines.append(str(userIdx) + "   " + str(userId) + "\n")
        with open(userIdx_id_output_path, 'w') as userIdx_id_output_file:
            userIdx_id_output_file.writelines(userIdx_id_output_outputLines)

        # print itemIdx to itemId
        itemIdx_id_output_path = self.inputPath + '/' + self.splitterType + '/item_idx_id.txt'
        itemIdx_id_outputLines = []
        itemIdx_id_outputLines.append("idx  id\n")
        for itemIdx in range(self.numItem):
            itemId = self.itemIdxToItemId[itemIdx]
            itemIdx_id_outputLines.append(str(itemIdx) + "  " + str(itemId + "\n"))
        with open(itemIdx_id_output_path, 'w') as itemIdx_id_output_file:
            itemIdx_id_output_file.writelines(itemIdx_id_outputLines)



    def generateEvalItemsForEachUser(self):
        for userIdx in self.evalItemsForEachUser:
            itemsToEval = self.evalItemsForEachUser[userIdx]
            itemsInTrain = self.user_items_train[userIdx]
            while len(itemsToEval) < 100:
                newItemIdx = random.randint(0, self.numItem-1)
                if newItemIdx not in itemsInTrain:
                    itemsToEval.add(newItemIdx)
            self.evalItemsForEachUser[userIdx] = itemsToEval

    def printInfo(self):
        self.logger.info('dataset: ' + str(self.fileName))
        self.logger.info('trainType: ' + str(self.trainType))
        self.logger.info('trainSize: ' + str(self.trainSize))
        self.logger.info('testSize: ' + str(self.testSize))
        self.logger.info('numUser: ' + str(self.numUser))
        self.logger.info('numItem: ' + str(self.numItem))
        self.logger.info('numWord:' + str(self.numWord))
        self.logger.info('numRating:' + str(self.trainSize + self.testSize))
        self.logger.info('numReviews per item:' + str((self.trainSize + self.testSize) / self.numItem))
        self.logger.info('ratingScale: ' + str(self.ratingScaleSet))
        self.logger.info('density: ' + str((self.trainSize + self.testSize) / (self.numItem * self.numUser)))
        self.logger.info('Num item in testSet: ' + str(len(self.itemsInTestSet)))


    def initialize_logger(self, output_dir):
        logger = logging.getLogger()
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

    def isContain(self, alist):
        '''check if self.wordIdToWordIndex contains the wordIds in list'''
        for wordId in alist:
            if wordId not in self.wordIdToWordIndex:
                return False
        return True

    def getWordIndices(self, alist):
        '''convert wordIds to wordIndices'''
        wordIndices = []
        for wordId in alist:
            wordIndices.append(self.wordIdToWordIndex[wordId])
        return wordIndices

    def listToString(self, list):
        listString = ''
        for i in range(len(list)):
            if i == len(list)-1:
                listString = listString + str(list[i])
            else:
                listString = listString + str(list[i]) + ':'

        return  listString

    def readWordVectors(self):
        self.wordMatrix = np.zeros([self.numWord, self.wordVecSize], dtype=np.float32)
        with open(self.inputPath + '/word2Vectors.txt') as wordVecFile:
            for line in wordVecFile:
                wordId, wordString, wordVec = line.split(' ')
                wordVecValues = wordVec.split(',')
                self.wordMatrix[int(wordId)] = [float(i) for i in wordVecValues]
                self.wordIdToWordVector[int(wordId)] = [float(i) for i in wordVecValues]
        self.wordMatrix = np.array(self.wordMatrix)

    def pad_user_item_list(self):
        for userId in self.userSet:
            paded = list(self.user_items_train[userId])
            while len(paded) < self.item_pad_num:
                paded.append(0)
            self.user_items_train_paded[userId] = paded

    def get_User_itemsReviewVectors(self):
        self.readWordVectors()
        for userId in self.userSet:
            userReviewVecs = []
            # append real reviews
            for itemId in self.user_items_train[userId]:

                if self.if_full_review == False:
                    reviewVec = self.user_itemToWordTrain[userId, itemId]
                else:
                    reviewVec = self.item_reviewVectors[itemId]

                if len(reviewVec) > self.maxReviewLength:
                    reviewVec = reviewVec[0:self.maxReviewLength]
                while len(reviewVec) < self.maxReviewLength:
                    reviewVec.append(0)
                userReviewVecs.append(reviewVec)

            # append null reviews
            while len(userReviewVecs) < self.item_pad_num:
                # random_idx = random.randint(0, self.numWord - 1)
                fake_review_vec = userReviewVecs[-1]
                userReviewVecs.append(fake_review_vec)

            self.user_items_reviewVectors[userId] = userReviewVecs

    def getItemReviewVectors(self):

        for itemId in self.itemSet:
            reviewVec = self.item_reviewVectors[itemId]
            if len(reviewVec) > self.maxReviewLength:
                reviewVec = reviewVec[0:self.maxReviewLength]
            while len(reviewVec) < self.maxReviewLength:
                reviewVec.append(0)
            self.item_reviewVectors[itemId] = reviewVec

    def getUserReviewVectors(self):

        for userId in self.userSet:
            reviewVec = self.user_reviewVectors[userId]
            if len(reviewVec) > self.maxReviewLength:
                reviewVec = reviewVec[0:self.maxReviewLength]
            while len(reviewVec) < self.maxReviewLength:
                reviewVec.append(0)
            self.user_reviewVectors[userId] = reviewVec



    def getUserReviewMask(self):
        for userId in self.userSet:
            mask = [True] * len(self.user_items_train[userId])
            while len(mask) < self.item_pad_num:
                mask.append(False)
            self.user_review_mask[userId] = mask

    def calReviewLengthThreshold(self, ratio_threshold=0.5, manual=-1):
        if manual == -1:
            if self.if_full_review == False:
                trainPath = self.inputPath + '/' + self.splitterType + '/train.txt'
                validPath = self.inputPath + '/' + self.splitterType + '/valid.txt'

                for filePath in [trainPath, validPath]:
                    with open(filePath) as inputFile:
                        for line in inputFile:
                            _, _, _, reviewString, _ = line.split(' ')
                            reviewLen = 0
                            sentences = reviewString.split('%')
                            for sentence in sentences:
                                words = sentence.split(':')[:-1]
                                reviewLen += len(words)
                            self.saveReviewLength(reviewLen)
            else: # not full review
                for reviewVec in self.item_reviewVectors.values():
                    self.saveReviewLength(len(reviewVec))

            # get the number of reviews
            sum = 0
            for value in self.reviewLengthScale.values():
                sum += value

                # convert number to ratio
            for key, value in self.reviewLengthScale.items():
                self.reviewLengthScale[key] = value / sum
                # print review length distribution
            self.logger.info(self.reviewLengthScale)
            # select maxReviewLength
            accumulate = 0
            for key, value in self.reviewLengthScale.items():
                accumulate += value
                if accumulate >= ratio_threshold:
                    self.maxReviewLength = key
                    break
        else:
            self.maxReviewLength = manual

        self.logger.info('maxReviewLength:' + str(self.maxReviewLength))


    def saveReviewLength(self, reviewLen):
        if reviewLen <= 5:
            self.reviewLengthScale[5] += 1
        elif reviewLen <= 10:
            self.reviewLengthScale[10] += 1
        elif reviewLen <= 20:
            self.reviewLengthScale[20] += 1
        elif reviewLen <= 30:
            self.reviewLengthScale[30] += 1
        elif reviewLen <= 40:
            self.reviewLengthScale[40] += 1
        elif reviewLen <= 50:
            self.reviewLengthScale[50] += 1
        elif reviewLen <= 60:
            self.reviewLengthScale[60] += 1
        elif reviewLen <= 70:
            self.reviewLengthScale[70] += 1
        elif reviewLen <= 80:
            self.reviewLengthScale[80] += 1
        elif reviewLen <= 90:
            self.reviewLengthScale[90] += 1
        elif reviewLen <= 100:
            self.reviewLengthScale[100] += 1
        elif reviewLen <= 150:
            self.reviewLengthScale[150] += 1
        elif reviewLen <= 200:
            self.reviewLengthScale[200] += 1
        elif reviewLen <= 250:
            self.reviewLengthScale[250] += 1
        elif reviewLen <= 300:
            self.reviewLengthScale[300] += 1
        elif reviewLen <= 350:
            self.reviewLengthScale[350] += 1
        elif reviewLen <= 400:
            self.reviewLengthScale[400] += 1
        else:
            self.reviewLengthScale[401] += 1

    def generate_data_for_HFT(self):
        trainLines = self.read_data_for_HFT('train') + self.read_data_for_HFT('valid')
        testLines = self.read_data_for_HFT('test')
        fullLines = trainLines + testLines

        with open(self.inputPath + '/' + self.splitterType + '/hft_full.txt', 'w') as outputFile:
            outputFile.writelines(fullLines)
        with open(self.inputPath + '/' + self.splitterType + '/hft_train.txt', 'w') as outputFile:
            outputFile.writelines(trainLines)
        with open(self.inputPath + '/' + self.splitterType + '/hft_test.txt', 'w') as outputFile:
            outputFile.writelines(testLines)
        self.generate_eval_for_HFT()

    def read_data_for_HFT(self, pathName):
        inputPath = self.inputPath + '/' + self.splitterType + '/' + pathName + '.txt'
        outputLines = []

        with open(inputPath) as trainFile:
            for line in trainFile:
                userId, itemId, rating, reviewString, timeStamp = line.strip().split(' ')

                rating = float(rating)
                if self.threshold < 0:
                    pass
                elif rating > self.threshold:
                    rating = '1.0'
                else:
                    rating = '0.0'

                sentences = reviewString.split("%")
                HFT_words = []
                for sentence in sentences:
                    words = sentence.split(":")
                    for word in words:
                        if word == '':
                            pass
                        else:
                            HFT_words.append(word)

                output_line = userId + ' ' + itemId + ' ' + str(rating) + ' ' + timeStamp + ' ' + str(len(HFT_words))
                for word in HFT_words:
                    output_line += ' ' + word
                output_line += '\n'
                outputLines.append(output_line)
        return outputLines

    def generate_eval_for_HFT(self):
        outputLines = []
        for userIdx in self.evalItemsForEachUser:
            userId = self.userIdxToUserId[userIdx]
            outputLine = userId
            for itemIdx in self.evalItemsForEachUser[userIdx]:
                itemId = self.itemIdxToItemId[itemIdx]
                outputLine += ' ' + itemId
            outputLine += '\n'
            outputLines.append(outputLine)
        with open(self.inputPath + '/' + self.splitterType + '/eval.txt', 'w') as outputFile:
            outputFile.writelines(outputLines)

    def buildModel(self):
        self.logger.info("\n###### information of DataModel ######\n")
        self.readData()
        self.calReviewLengthThreshold(ratio_threshold=0.7)
        self.generateEvalItemsForEachUser()
        self.getItemReviewVectors()
        self.getUserReviewVectors()
        self.get_User_itemsReviewVectors()
        self.getUserReviewMask()
        self.pad_user_item_list()
        self.printInfo()






