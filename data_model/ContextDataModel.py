import random
import logging
import os.path
import time
from data_model.BasicDataModel import BasicDataModel
import collections
import numpy as np

class ContextDataModel(BasicDataModel):

    def __init__(self, config):
        super(ContextDataModel, self).__init__(config)

        self.if_context = config['if_context']

        self.user_item_like = {}
        self.user_item_dislike = {}
        self.user_features = {}

        self.user_ages = set()
        self.user_gender = {'M': 0, 'F': 1}
        self.user_occupations = {}
        self.user_zip_codes = {}
        self.featureSize = 0
        self.maxHist = 0

        self.itemsInTestSet = set()
        self.evalItemsForEachUser = {}

        self.min_item_num_per_user = 100
        self.min_item_like = 100
        self.min_item_dislike = 100

    def readUserContext(self):
        contextPath = self.inputPath + '/' + self.splitterType + '/user-context.txt'
        basicOccupyIndex = 0
        basicZipCodeIndex = 0
        with open(contextPath) as contextFile:
            for line in contextFile:
                userId, age, gender, occupy, zipcode = line.split('|')
                genderIdx = self.user_gender[gender]
                userIdx = self.userIdToUserIdx[userId]
                self.user_ages.add(int(age))
                if occupy not in self.user_occupations:
                    self.user_occupations[occupy] = basicOccupyIndex
                    basicOccupyIndex += 1
                if zipcode not in self.user_zip_codes:
                    self.user_zip_codes[zipcode] = basicZipCodeIndex
                    basicZipCodeIndex += 1
                occupyIdx = self.user_occupations[occupy]
                zipcodeIdx = self.user_zip_codes[zipcode]

                self.user_features[userIdx] = [int(age), genderIdx, occupyIdx, zipcodeIdx]
        self.build_user_features()
        self.getMaxHist()

    def getMaxHist(self):
        for userIdx in self.user_items_train:
            if len(self.user_items_train[userIdx]) > self.maxHist:
                self.maxHist = len(self.user_items_train[userIdx])

        for userIdx in self.user_items_train:
            while len(self.user_items_train[userIdx]) < self.maxHist:
                self.user_items_train[userIdx].append(self.numItem)

    def build_user_features(self):
        minAge = sorted(self.user_ages)[0]
        maxAge = sorted(self.user_ages, reverse=True)[0]
        ageScalue = maxAge - minAge
        self.featureSize = 1 + 2 + len(self.user_occupations) + len(self.user_zip_codes)
        for userIdx in self.userSet:
            age, genderIdx, occupyIdx, zipcodeIdx = self.user_features[userIdx]
            normAge = (age - minAge) / ageScalue
            ageVec = [normAge]
            genderVec = [0] * 2
            genderVec[genderIdx] = 1
            occupyVec = [0] * len(self.user_occupations)
            occupyVec[occupyIdx] = 1
            zipVec = [0] * len(self.user_zip_codes)
            zipVec[zipcodeIdx] = 1
            self.user_features[userIdx] = ageVec + genderVec + occupyVec + zipVec

    def readData(self):
        trainPath = self.inputPath + '/' + self.splitterType + '/train.txt'
        validPath = self.inputPath + '/' + self.splitterType + '/valid.txt'
        testPath = self.inputPath + '/' + self.splitterType + '/test.txt'

        # read train data
        self.userIdToUserIdx = {}
        self.itemIdToItemIdx = {}
        basicUserIndex = 0
        basicItemIndex = 0

        with open(trainPath) as trainFile:
            for line in trainFile:
                userId, itemId, rating, timeStamp = line.split(' ')
                if userId not in self.userIdToUserIdx:
                    self.userIdToUserIdx[userId] = basicUserIndex
                    self.userIdxToUserId[basicUserIndex] = userId

                    basicUserIndex += 1
                if itemId not in self.itemIdToItemIdx:
                    self.itemIdToItemIdx[itemId] = basicItemIndex
                    self.itemIdxToItemId[basicItemIndex] = itemId

                    basicItemIndex += 1
                userId = self.userIdToUserIdx[userId]
                itemId = self.itemIdToItemIdx[itemId]
                rating = float(rating)

                if rating >= 3:
                    if userId not in self.user_item_like:
                        self.user_item_like[userId] = []
                    self.user_item_like[userId].append(itemId)
                if rating <= 2:
                    if userId not in self.user_item_dislike:
                        self.user_item_dislike[userId] = []
                    self.user_item_dislike[userId].append(itemId)

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
                self.trainSet.append([userId, itemId, rating])


        # read valid data
        with open(validPath) as validFile:
            for line in validFile:
                userId, itemId, rating, timeStamp = line.split(' ')
                if userId not in self.userIdToUserIdx:
                    self.userIdToUserIdx[userId] = basicUserIndex
                    self.userIdxToUserId[basicUserIndex] = userId

                    basicUserIndex += 1
                if itemId not in self.itemIdToItemIdx:
                    self.itemIdToItemIdx[itemId] = basicItemIndex
                    self.itemIdxToItemId[basicItemIndex] = itemId

                    basicItemIndex += 1
                userId = self.userIdToUserIdx[userId]
                itemId = self.itemIdToItemIdx[itemId]
                rating = float(rating)

                if rating >= 4:
                    if userId not in self.user_item_like:
                        self.user_item_like[userId] = []
                    self.user_item_like[userId].append(itemId)
                if rating <= 2:
                    if userId not in self.user_item_dislike:
                        self.user_item_dislike[userId] = []
                    self.user_item_dislike[userId].append(itemId)

                if self.trainType == 'test':
                    # binarize
                    if self.threshold < 0:
                        pass
                    elif rating > self.threshold:
                        rating = 1.0
                    else:
                        rating = 0.0

                if userId not in self.user_items_train.keys():
                    self.user_items_train[userId] = []
                self.user_items_train[userId].append(itemId)
                self.validSet.append([int(userId), int(itemId), rating])

        for userIdx, itemList in self.user_items_train.items():
            len_item_list = len(itemList)
            if len_item_list < self.min_item_num_per_user:
                self.min_item_num_per_user = len_item_list

        for userIdx, itemList in self.user_item_like.items():
            len_item_list = len(itemList)
            if len_item_list < self.min_item_like:
                self.min_item_like = len_item_list

        for userIdx, itemList in self.user_item_dislike.items():
            len_item_list = len(itemList)
            if len_item_list < self.min_item_dislike:
                self.min_item_dislike = len_item_list

        self.numUser = basicUserIndex
        self.numItem = basicItemIndex

        # read test data
        with open(testPath) as testFile:
            for line in testFile:
                userId, itemId, rating, timeStamp = line.split(' ')
                rating = float(rating)
                if userId not in self.userIdToUserIdx or itemId not in self.itemIdToItemIdx:
                    continue
                userId = self.userIdToUserIdx[userId]
                itemId = self.itemIdToItemIdx[itemId]

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

        if self.if_context:
            self.readUserContext()

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
        self.logger.info('numRating:' + str(self.trainSize + self.testSize))
        self.logger.info('numReviews per item:' + str((self.trainSize + self.testSize) / self.numItem))
        self.logger.info('ratingScale: ' + str(self.ratingScaleSet))
        self.logger.info('min_item_num_per_user: ' + str(self.min_item_num_per_user))
        self.logger.info('min_item_like: ' + str(self.min_item_like))
        self.logger.info('min_item_dislike: ' + str(self.min_item_dislike))
        self.logger.info('density: ' + str((self.trainSize + self.testSize) / (self.numItem * self.numUser)))
        self.logger.info('Num item in testSet: ' + str(len(self.itemsInTestSet)))
        self.logger.info('------------------------------------------------------------')



    def buildModel(self):
        self.pre_process()
        self.split_UserTimeRatio()
        self.logger.info("\n###### information of DataModel ######\n")
        self.readData()

        self.generateEvalItemsForEachUser()

        self.printInfo()






