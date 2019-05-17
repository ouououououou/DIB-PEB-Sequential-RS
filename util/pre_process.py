# data process for HFT
import json
import re
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import random
import logging
import os.path
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from operator import itemgetter, attrgetter, methodcaller
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class DataPreprocessor(object):

    def __init__(self, fileName):
        self.fileName = fileName
        self.word2VecFile = '../dataset/raw_datasets/' + self.fileName + '/word2vec/model_file/word2vec_org'
        self.userIdToUserIndex = {}
        self.itemIdToItemIndex = {}
        self.userSet = set()
        self.itemSet = set()
        self.wordToWordIndex = {}
        self.wordIndexToWord = {}
        self.wordIndexToVector = {}
        self.specialWords = {}
        self.wrongToCorrectSpelling = {}
        self.wholeDataSet = []
        self.model = None
        self.numUser = 0
        self.numItem = 0
        self.numWord = 0
        self.numRatingBeforeSplit = 0
        self.numRatingAfterSplit = 0
        self.trainSize = 0
        self.validSize = 0
        self.testSize = 0
        self.numUserAfterSplit = 0
        self.numItemAfterSplit = 0
        self.outputPath = '../dataset/processed_datasets/' + self.fileName
        self.logger = self.initialize_logger('../log')
        self.vocab_8000 = set()

        self.user_feedback_count = {'1-5': 0,
                                    '5-10': 0,
                                    '11-15': 0,
                                    '16-20': 0,
                                    '>20': 0}
        self.item_feedback_count = {'0-5': 0,
                                    '5-10': 0,
                                    '11-15': 0,
                                    '16-20': 0,
                                    '>20': 0}

    def pre_filter_important_words(self, word_num):
        '''get important words by idf'''

        if self.model == None:
            self.model = gensim.models.KeyedVectors.load_word2vec_format(self.word2VecFile, binary=True)
        # model = gensim.models.KeyedVectors.load_word2vec_format(self.word2VecFile, binary=True)
        vocab = self.model.vocab.keys()
        item_reviews = {}

        word_tokenizer = RegexpTokenizer(r'\w+')
        f = open('../dataset/raw_datasets/' + self.fileName + '/word2vec/raw_data/' + self.fileName + '.json')
        lines = f.readlines()
        for line in lines:
            json_data = json.loads(line)
            # get properties
            itemId = str(json_data['asin'])
            if itemId not in item_reviews:
                item_reviews[itemId] = ''
            reviewString = json_data['reviewText']
            reviewToSave = ''
            words = word_tokenizer.tokenize(reviewString)
            if len(words) > 2:
                for word in words:
                    word = word.lower()
                    if word in vocab:  # remove special words
                        reviewToSave = reviewToSave + str(word) + ' '
            item_reviews[itemId] += reviewToSave

        vectorizer = TfidfVectorizer(stop_words='english')
        tf_idf_matrix = vectorizer.fit_transform(item_reviews.values()).toarray()
        words = vectorizer.get_feature_names()

        row_num = len(item_reviews)
        # for row in tf_idf_matrix:
        #     non_zero_count = np.count_nonzero(row)
        #     sorted_row = row.argsort()[:int(non_zero_count // 2)]
        #
        #     for keyWordIdx in sorted_row:
        #         keyWord = words[keyWordIdx]
        #         self.vocab_8000.add(keyWord)
        # print(len(self.vocab_8000))


        # mat_array = X.toarray()
        # words = vectorizer.get_feature_names()
        # for l in mat_array:
        #     (l*-1).argsort()
        idf = vectorizer.idf_
        word_to_idf = dict(zip(words, idf))
        sorted_wordToIDF = sorted(word_to_idf.items(), key=lambda x: x[1], reverse=True)
        importantWords = []

        length = 0
        for key, value in sorted_wordToIDF:
            importantWords.append(key)
            length += 1
            if length == word_num:
                break
        self.vocab_8000 = set(importantWords)


    def pre_filter_active_users(self, wordNum, itemNum):
        '''filter active users and there feedbacks'''

        # 1. get each user's feedback count and (item, timeStamp) pairs
        word_tokenizer = RegexpTokenizer(r'\w+')
        f = open('../dataset/raw_datasets/' + self.fileName + '/word2vec/raw_data/' + self.fileName + '.json')
        lines = f.readlines()
        user_itemNum = {}
        item_userNum = {}
        outputLines = []
        user_items = {}
        item_users = {}
        efficientReview = {}
        for line in lines:
            json_data = json.loads(line)
            # get properties
            userId = str(json_data['reviewerID'])
            itemId = str(json_data['asin'])
            timeStamp = int(json_data['unixReviewTime'])
            reviewString = json_data['reviewText']
            words = word_tokenizer.tokenize(reviewString)

            efficientWordCount = 0
            for word in words:
                if word in self.vocab_8000:
                    efficientWordCount += 1

            # only consider no-empty reviews
            if efficientWordCount >= wordNum:
                if userId not in user_itemNum.keys():
                    user_items[userId] = []
                    user_itemNum[userId] = 0
                if itemId not in item_userNum.keys():
                    item_users[itemId] = []
                    item_userNum[itemId] = 0
                efficientReview[userId, itemId] = 1

                user_itemNum[userId] += 1
                item_userNum[itemId] += 1
                user_items[userId].append([itemId, timeStamp])
                item_users[itemId].append([itemId, timeStamp])

        # 2. get inactive user's 10 continuous items
        for line in lines:
            json_data = json.loads(line)
            userId = str(json_data['reviewerID'])
            itemId = str(json_data['asin'])
            # ignore users with inefficient reviews
            if userId not in user_itemNum.keys():
                # print(userId)
                continue
            if user_itemNum[userId] <= itemNum and user_itemNum[userId] >= 3 and (userId, itemId) in efficientReview:
                outputLines.append(line)
        with open('../dataset/raw_datasets/' + self.fileName + '/word2vec/raw_data/' + self.fileName + '_filtered.json', 'w') as outputFile:
            outputFile.writelines(outputLines)

    def pre_filter_active_users_for_seq(self, itemNum):
        '''filter active users and their feedbacks'''

        # 1. get each user's feedback count and (item, timeStamp) pairs
        word_tokenizer = RegexpTokenizer(r'\w+')
        f = open('../dataset/raw_datasets/' + self.fileName + '/word2vec/raw_data/' + self.fileName + '.json')
        lines = f.readlines()
        user_itemNum = {}
        outputLines = []
        for line in lines:
            json_data = json.loads(line)
            # get properties
            userId = str(json_data['reviewerID'])

            # only consider no-empty reviews
            if userId not in user_itemNum.keys():
                user_itemNum[userId] = 0
            user_itemNum[userId] += 1

        # 2. get inactive user's 10 continuous items
        for line in lines:
            json_data = json.loads(line)
            userId = str(json_data['reviewerID'])
            itemId = str(json_data['asin'])
            rating = str(json_data['overall'])
            timeStamp = str(json_data['unixReviewTime'])
            # ignore users with inefficient reviews
            if user_itemNum[userId] >= itemNum:
                outputLines.append(userId + ' ' + itemId + ' ' + rating + ' ' + timeStamp + '\n')
        with open('../dataset/raw_datasets/' + self.fileName + '/word2vec/raw_data/' + self.fileName + '_' + str(itemNum) + '_seq.txt', 'w') as outputFile:
            outputFile.writelines(outputLines)


    def pre_process(self):
        if self.model == None:
            self.model = gensim.models.KeyedVectors.load_word2vec_format(self.word2VecFile)
        # model = gensim.models.KeyedVectors.load_word2vec_format(self.word2VecFile, binary=True)
        vocab = self.model.vocab.keys()

        f = open('../dataset/raw_datasets/' + self.fileName + '/word2vec/raw_data/' + self.fileName + '_filtered.json')
        f2 = open('../dataset/raw_datasets/' + self.fileName + '/' + self.fileName + '_review.txt', 'w')
        f3 = open('../dataset/raw_datasets/' + self.fileName + '/' + self.fileName + '_review_detail.txt', 'w')

        lines = f.readlines()
        word_tokenizer = RegexpTokenizer(r'\w+')

        basicUserIndex = 0
        basicItemIndex = 0
        basicWordIndex = 1

        for line in lines:
            json_data = json.loads(line)
            # get properties
            userId = str(json_data['reviewerID'])
            itemId = str(json_data['asin'])
            rating = str(json_data['overall'])
            # raw_review = re.findall(pat, json_data['summary'])
            raw_review = str(json_data['reviewText'])
            timeStamp = int(json_data['unixReviewTime'])

            if userId not in self.userIdToUserIndex:
                self.userIdToUserIndex[userId] = basicUserIndex
                basicUserIndex += 1
            if itemId not in self.itemIdToItemIndex:
                self.itemIdToItemIndex[itemId] = basicItemIndex
                basicItemIndex += 1
            userIndex = userId
            itemIndex = itemId

            sentences = sent_tokenize(raw_review)
            reviewString = ''
            reviewString_detail = ''
            for sentence in sentences:
                sentenceString = ''
                sentenceString_detail = ''
                words = word_tokenizer.tokenize(sentence)
                for word in words:
                    word = word.lower()
                    if word in self.vocab_8000:  # check special words
                        if word not in self.wordToWordIndex:
                            self.wordToWordIndex[word] = basicWordIndex
                            self.wordIndexToWord[basicWordIndex] = word
                            self.wordIndexToVector[basicWordIndex] = self.model[str(word)]
                            basicWordIndex += 1
                        wordIndex = self.wordToWordIndex[word]
                        sentenceString = sentenceString + str(wordIndex) + ':'
                        sentenceString_detail = sentenceString_detail + str(word) + ':'
                    else:
                        if word not in self.specialWords:
                            self.specialWords[word] = 1
                        else:
                            self.specialWords[word] += 1

                reviewString = reviewString + sentenceString + '%'
                reviewString_detail = reviewString_detail + sentenceString_detail + '%'

            self.wholeDataSet.append([userIndex, itemIndex, rating, reviewString, timeStamp])
            self.userSet.add(userIndex)
            self.itemSet.add(itemIndex)

            eachline = str(userId) + ',' + str(itemId) + ',' + rating + ',' + reviewString + ',' + str(timeStamp) + '\n'
            detailLine = str(userId) + ',' + str(itemId) + ',' + rating + ',' + reviewString_detail + ',' + str(timeStamp) + '\n'

            # newlines.append(eachline)
            f2.write(eachline)
            # contentDetail.append(detailLine)
            f3.write(detailLine)

        # f2.writelines(newlines)
        # f3.writelines(contentDetail)

        # print special words
        # self.printSpeicalWords()
        self.numUser = basicUserIndex
        self.numItem = basicItemIndex
        self.numWord = basicWordIndex
        self.numRatingBeforeSplit = len(self.wholeDataSet)

        self.logger.info('numUser:' + str(basicUserIndex))
        self.logger.info('numItem:' + str(basicItemIndex))
        self.logger.info('numWord:' + str(basicWordIndex))
        self.logger.info('numRating:' + str(self.numRatingBeforeSplit))

    def split_UserRatio(self, trainRatio = 0.7, validRatio = 0.1):

        random.seed(123)
        # collect all user-items mappings
        user_items = {}
        # collect all user item - rating review timeStamp mappings
        userItemToRatingReviewTimeStamp = {}
        for record in self.wholeDataSet:
            userIdx, itemIdx, rating, reviewString, timeStamp = record
            if userIdx not in user_items.keys():
                user_items[userIdx] = []
            user_items[userIdx].append(itemIdx)
            userItemToRatingReviewTimeStamp[userIdx, itemIdx] = [rating, reviewString, timeStamp]

        # split the data
        trainSet = []
        validSet = []
        testSet = []
        itemsInTrainSet = set()
        usersInTrainSet = set()
        for userIndex in range(self.numUser):
            items = user_items[userIndex]
            for itemIndex in items:
                rating, reviewString, timeStamp = userItemToRatingReviewTimeStamp[userIndex, itemIndex]
                threshold = random.uniform(0, 1)
                if threshold <= trainRatio:
                    trainSet.append([userIndex, itemIndex, rating, reviewString, timeStamp])
                    itemsInTrainSet.add(itemIndex)
                    usersInTrainSet.add(userIndex)
                elif trainRatio < threshold <= (trainRatio + validRatio):
                    validSet.append([userIndex, itemIndex, rating, reviewString, timeStamp])
                else:
                    testSet.append([userIndex, itemIndex, rating, reviewString, timeStamp])

        trainSetToWrite = []
        validSetToWrite = []
        testSetToWrite = []

        delete_count = 0
        for record in trainSet:
            userIndex, itemIndex, rating, reviewString, timeStamp = record
            trainSetToWrite.append(str(userIndex) + " " + str(itemIndex) + " " + str(rating) + " " + reviewString + " " + str(timeStamp) + '\n')

        for record in validSet:
            userIndex, itemIndex, rating, reviewString, timeStamp = record
            if userIndex in usersInTrainSet and itemIndex in itemsInTrainSet:
                validSetToWrite.append(str(userIndex) + " " + str(itemIndex) + " " + str(rating) + " " + reviewString + " " + str(timeStamp) + '\n')
            else:
                delete_count += 1

        for record in testSet:
            userIndex, itemIndex, rating, reviewString, timeStamp = record
            if userIndex in usersInTrainSet and itemIndex in itemsInTrainSet:
                testSetToWrite.append(str(userIndex) + " " + str(itemIndex) + " " + str(rating) + " " + reviewString + " " + str(timeStamp) + '\n')
            else:
                delete_count += 1

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

        self.numRatingAfterSplit = self.numRatingBeforeSplit - delete_count
        assert self.numRatingAfterSplit == self.trainSize + self.validSize + self.testSize

        fullOutputPath = self.outputPath + '/' + self.fileName + '/userRatio'
        if not os.path.exists(fullOutputPath):
            os.makedirs(fullOutputPath)

        with open(fullOutputPath + '/train.txt', 'w') as trainFile:
            trainFile.writelines(trainSetToWrite)

        with open(fullOutputPath + '/valid.txt', 'w') as validFile:
            validFile.writelines(validSetToWrite)

        with open(fullOutputPath + '/test.txt', 'w') as testFile:
            testFile.writelines(testSetToWrite)

        with open(fullOutputPath + '/README.txt', 'w') as readMeFile:
            readMeFile.writelines(self.generateReadMe())



    def split_UserTimeRatio(self, trainRatio = 0.7, validRatio = 0.1):

        random.seed(123)
        # collect all user-items mappings
        user_items = {}
        # collect all user item - rating review timeStamp mappings
        userItemToRatingReviewTimeStamp = {}
        for record in self.wholeDataSet:
            userIdx, itemIdx, rating, reviewString, timeStamp = record
            if userIdx not in user_items.keys():
                user_items[userIdx] = []
            user_items[userIdx].append([itemIdx, timeStamp])
            userItemToRatingReviewTimeStamp[userIdx, itemIdx] = [rating, reviewString, timeStamp]

        # split the data
        trainSet = []
        validSet = []
        testSet = []
        itemsInTrainSet = set()
        usersInTrainSet = set()
        for userIndex in self.userSet:
            # get and sort items
            items = sorted(user_items[userIndex], key=itemgetter(1))
            numItems = len(items)
            # split first n items into train set, except the last 2
            # items for trainSet
            for itemIndex, timeStamp in items[: -2]:
                rating, reviewString, timeStamp = userItemToRatingReviewTimeStamp[userIndex, itemIndex]
                trainSet.append([userIndex, itemIndex, rating, reviewString, timeStamp])
                itemsInTrainSet.add(itemIndex)
                usersInTrainSet.add(userIndex)
            # items for validSet
            for itemIndex, timeStamp in items[-2:-1]:
                rating, reviewString, timeStamp = userItemToRatingReviewTimeStamp[userIndex, itemIndex]
                validSet.append([userIndex, itemIndex, rating, reviewString, timeStamp])
                itemsInTrainSet.add(itemIndex)
                usersInTrainSet.add(userIndex)
            # items for testSet
            for itemIndex, timeStamp in items[-1:]:
                rating, reviewString, timeStamp = userItemToRatingReviewTimeStamp[userIndex, itemIndex]
                testSet.append([userIndex, itemIndex, rating, reviewString, timeStamp])

        trainSetToWrite = []
        validSetToWrite = []
        testSetToWrite = []
        trainSetToLibRec = []
        validSetToLibRec = []
        testSetToLibRec = []
        delete_count = 0

        for record in trainSet:
            userIndex, itemIndex, rating, reviewString, timeStamp = record
            trainSetToWrite.append(
                str(userIndex) + " " + str(itemIndex) + " " + str(rating) + " " + reviewString + " "
                + str(timeStamp) + '\n')
            trainSetToLibRec.append(str(userIndex) + " " + str(itemIndex) + " " + str(rating) + '\n')

        for record in validSet:
            userIndex, itemIndex, rating, reviewString, timeStamp = record
            validSetToWrite.append(
                str(userIndex) + " " + str(itemIndex) + " " + str(rating) + " " + reviewString + " "
                + str(timeStamp) + '\n')
            validSetToLibRec.append(str(userIndex) + " " + str(itemIndex) + " " + str(rating) + '\n')

        for record in testSet:
            userIndex, itemIndex, rating, reviewString, timeStamp = record
            if userIndex in usersInTrainSet and itemIndex in itemsInTrainSet:
                testSetToWrite.append(
                    str(userIndex) + " " + str(itemIndex) + " " + str(rating) + " " + reviewString + " "
                    + str(timeStamp) + '\n')
                testSetToLibRec.append(str(userIndex) + " " + str(itemIndex) + " " + str(rating) + '\n')
            else:
                delete_count += 1

        self.trainSize = len(trainSetToWrite)
        self.validSize = len(validSetToWrite)
        self.testSize = len(testSetToWrite)
        self.logger.info('trainSize: ' + str(self.trainSize))
        self.logger.info('validSize: ' + str(self.validSize))
        self.logger.info('testSize: ' + str(self.testSize))

        self.numRatingAfterSplit = self.numRatingBeforeSplit - delete_count
        # assert self.numRatingAfterSplit == self.trainSize + self.validSize + self.testSize
        self.logger.info('numRatingAfterSplit: ' + str(self.numRatingAfterSplit))
        self.numUserAfterSplit = len(usersInTrainSet)
        self.numItemAfterSplit = len(itemsInTrainSet)
        self.logger.info('numUser after split: ' + str(self.numUserAfterSplit))
        self.logger.info('numItem after aplit: ' + str(self.numItemAfterSplit))

        fullOutputPath = self.outputPath + '/userTimeRatio'
        if not os.path.exists(fullOutputPath):
            os.makedirs(fullOutputPath)

        with open(fullOutputPath + '/train.txt', 'w') as trainFile:
            trainFile.writelines(trainSetToWrite)

        with open(fullOutputPath + '/valid.txt', 'w') as validFile:
            validFile.writelines(validSetToWrite)

        with open(fullOutputPath + '/test.txt', 'w') as testFile:
            testFile.writelines(testSetToWrite)

        with open(fullOutputPath + '/librec_train.txt', 'w') as trainFile:
            trainFile.writelines(trainSetToLibRec)

        with open(fullOutputPath + '/librec_valid.txt', 'w') as validFile:
            validFile.writelines(validSetToLibRec)

        with open(fullOutputPath + '/librec_test.txt', 'w') as testFile:
            testFile.writelines(testSetToLibRec)

        with open(fullOutputPath + '/README.txt', 'w') as readMeFile:
            readMeFile.writelines(self.generateReadMe())


    def split_UserTimeLOO(self):
        random.seed(123)
        # collect all user-items mappings
        user_items = {}
        # collect all user item - rating review timeStamp mappings
        userItemToRatingReviewTimeStamp = {}
        for record in self.wholeDataSet:
            userIdx, itemIdx, rating, reviewString, timeStamp = record
            if userIdx not in user_items.keys():
                user_items[userIdx] = []
            user_items[userIdx].append([itemIdx, timeStamp])
            userItemToRatingReviewTimeStamp[userIdx, itemIdx] = [rating, reviewString, timeStamp]

        # split the data
        trainSet = []
        validSet = []
        testSet = []
        itemsInTrainSet = set()
        usersInTrainSet = set()
        for userIndex in range(self.numUser):
            # get and sort items
            items = sorted(user_items[userIndex], key=itemgetter(1))
            numItems = len(items)
            # if numItem less than 3, split them into trainSet
            if numItems <= 2:
                for itemIndex, timeStamp in items:
                    rating, reviewString, timeStamp = userItemToRatingReviewTimeStamp[userIndex, itemIndex]
                    trainSet.append([userIndex, itemIndex, rating, reviewString, timeStamp])
                    itemsInTrainSet.add(itemIndex)
                    usersInTrainSet.add(userIndex)
            # split items by {N-2, 1, 1}
            else:
                # items for trainSet
                for itemIndex, timeStamp in items[0: -2]:
                    rating, reviewString, timeStamp = userItemToRatingReviewTimeStamp[userIndex, itemIndex]
                    trainSet.append([userIndex, itemIndex, rating, reviewString, timeStamp])
                    itemsInTrainSet.add(itemIndex)
                    usersInTrainSet.add(userIndex)
                # item for validSet
                itemIndex, timeStamp = items[-2]
                rating, reviewString, timeStamp = userItemToRatingReviewTimeStamp[userIndex, itemIndex]
                validSet.append([userIndex, itemIndex, rating, reviewString, timeStamp])
                # item for testSet
                itemIndex, timeStamp = items[-1]
                rating, reviewString, timeStamp = userItemToRatingReviewTimeStamp[userIndex, itemIndex]
                testSet.append([userIndex, itemIndex, rating, reviewString, timeStamp])

        trainSetToWrite = []
        validSetToWrite = []
        testSetToWrite = []
        delete_count = 0
        for record in trainSet:
            userIndex, itemIndex, rating, reviewString, timeStamp = record
            trainSetToWrite.append(
                str(userIndex) + " " + str(itemIndex) + " " + str(rating) + " " + reviewString + " "
                + str(timeStamp) + '\n')

        for record in validSet:
            userIndex, itemIndex, rating, reviewString, timeStamp = record
            if userIndex in usersInTrainSet and itemIndex in itemsInTrainSet:
                validSetToWrite.append(
                    str(userIndex) + " " + str(itemIndex) + " " + str(rating) + " " + reviewString + " "
                    + str(timeStamp) + '\n')
            else:
                delete_count += 1

        for record in testSet:
            userIndex, itemIndex, rating, reviewString, timeStamp = record
            if userIndex in usersInTrainSet and itemIndex in itemsInTrainSet:
                testSetToWrite.append(
                    str(userIndex) + " " + str(itemIndex) + " " + str(rating) + " " + reviewString + " "
                    + str(timeStamp) + '\n')
            else:
                delete_count += 1

        self.trainSize = len(trainSetToWrite)
        self.validSize = len(validSetToWrite)
        self.testSize = len(testSetToWrite)
        self.logger.info('trainSize: ' + str(self.trainSize))
        self.logger.info('validSize: ' + str(self.validSize))
        self.logger.info('testSize: ' + str(self.testSize))

        self.numRatingAfterSplit = self.numRatingBeforeSplit - delete_count
        assert self.numRatingAfterSplit == self.trainSize + self.validSize + self.testSize

        self.numUserAfterSplit = len(usersInTrainSet)
        self.numItemAfterSplit = len(itemsInTrainSet)
        self.logger.info('numUser after split: ' + str(self.numUserAfterSplit))
        self.logger.info('numItem after aplit: ' + str(self.numItemAfterSplit))

        fullOutputPath = self.outputPath + '/' + self.fileName + '/userTimeLOO'
        if not os.path.exists(fullOutputPath):
            os.makedirs(fullOutputPath)

        with open(fullOutputPath + '/train.txt', 'w') as trainFile:
            trainFile.writelines(trainSetToWrite)

        with open(fullOutputPath + '/valid.txt', 'w') as validFile:
            validFile.writelines(validSetToWrite)

        with open(fullOutputPath + '/test.txt', 'w') as testFile:
            testFile.writelines(testSetToWrite)

        with open(fullOutputPath + '/README.txt', 'w') as readMeFile:
            readMeFile.writelines(self.generateReadMe())

    def init_for_split(self):
        path = '../dataset/' + self.fileName + '/' + self.fileName + '_review.txt'
        userIndices = set()
        itemIndices = set()
        wordIndices = set()
        bufsize = 65536
        with open(path) as infile:
            while True:
                lines = infile.readlines(bufsize)
                if not lines:
                    break
                for line in lines:
                    userIndex, itemIndex, rating, reviewString, timeStamp = line.split(',')
                    userIndices.add(userIndex)
                    itemIndices.add(itemIndex)
                    for sentences in reviewString.split("%"):
                        for wordIdx in sentences.split(":"):
                            wordIndices.add(wordIdx)
                    self.wholeDataSet.append([int(userIndex), int(itemIndex), rating, reviewString, int(timeStamp)])
        self.numRatingBeforeSplit = len(self.wholeDataSet)
        self.numUser = len(userIndices)
        self.numItem = len(itemIndices)
        self.numWord = len(wordIndices)

    def printSpeicalWords(self):
        specialWordFile = open('../dataset/raw_datasets/' + self.fileName + '/' + 'special_words.txt', 'w')
        specialWordLines = []
        specialWordKeys = self.specialWords.keys()
        for specialWord in specialWordKeys:
            specialWordLines.append(str(specialWord) + '    ' + str(self.specialWords[specialWord]) + '\n')
        specialWordFile.writelines(specialWordLines)

    def printWordVectorsAndMappings(self, vec_size):
        if self.model == None:
            self.model = gensim.models.KeyedVectors.load_word2vec_format(self.word2VecFile)
        wordMappingFile = open('../dataset/raw_datasets/' + self.fileName + '/' + 'word_mapping.txt', 'w')
        wordVectorDirPath = self.outputPath
        if not os.path.exists(wordVectorDirPath):
            os.makedirs(wordVectorDirPath)
        wordVectorFile = open(self.outputPath + '/' + 'word2Vectors.txt', 'w')
        vector_lines = []
        mapping_lines = []
        # append 0 word
        vector_lines.append('0' + ' ' + 'null' + ' ' + self.listToString([0.0] * vec_size) + '\n')
        mapping_lines.append('0' + ' ' + 'null' + '\n')
        # append other words
        for wordIdx in range(1, self.numWord):
            wordVector = self.wordIndexToVector[wordIdx]
            word = self.wordIndexToWord[wordIdx]
            vector_lines.append(str(wordIdx) + ' ' + str(word) + ' ' + self.listToString(wordVector) + '\n')
            mapping_lines.append(str(wordIdx) + ' ' + str(word) + '\n')
            # check the vectors
            realVector = self.model[word]
            # assert len(wordVector) == len(realVector) and len(wordVector) == 300
            for i in range(len(wordVector)):
                assert wordVector[i] == self.model[word][i]

        wordVectorFile.writelines(vector_lines)
        wordMappingFile.writelines(mapping_lines)

    # def printWrongCorrectSpellings(self):
    #     spellCheckFile = open('../dataset/' + self.fileName + '/' + 'spellCheck.txt', 'w')
    #     wrongWordSet = self.wrongToCorrectSpelling.keys()
    #     lines = []
    #     for wrongWord in wrongWordSet:
    #         correctWord = self.wrongToCorrectSpelling[wrongWord]
    #         line = wrongWord + ' -> ' + correctWord + '\n'
    #         lines.append(line)
    #     spellCheckFile.writelines(lines)


    def listToString(self, list):
        listString = ''
        for i in range(len(list)):
            if i == len(list)-1:
                listString = listString + str(list[i])
            else:
                listString = listString + str(list[i]) + ','

        return  listString

    def sortListByTime(self, items):
        sorted(items, key=itemgetter(1))

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

    def generateReadMe(self):
        return 'numUser: ' + str(self.numUser) + '\n' \
               + 'numItem: ' + str(self.numItem) + '\n'\
               + 'numWord: ' + str(self.numWord) + '\n' \
               + 'word2vec dimension: 300\n' \
               + 'numRating_before_split: ' + str(self.numRatingBeforeSplit) + '\n' \
               + 'numRating_after_split: ' + str(self.numRatingAfterSplit) + '\n' \
               + 'trainSize: ' + str(self.trainSize) + '\n'\
               + 'validSize: ' + str(self.validSize) + '\n'\
               + 'testSize: ' + str(self.testSize) + '\n'\
               + 'numUserAfterSplit: ' + str(self.numUserAfterSplit) + '\n'\
               + 'numItemAfterSplit: ' + str(self.numItemAfterSplit) + '\n'




if __name__ == '__main__':
    # text = "If you are a serious violin student on a budget, this edition has it all: Piano accompaniment, low price, urtext solo parts, and annotations by Maestro David Oistrakh where Bach's penmanship is hard to decipher.  Additions (in dashes) are easily distinguishable from the original bowings.  This is a delightful concerto that all intermediate level violinists should perform with a violin buddy.  Get your copy, today, along with \"The Green Violin; Theory, Ear Training, and Musicianship for Violinists\" book to prepare for this concerto and for more advanced playing!"
    # from nltk.tokenize import sent_tokenize
    # from nltk.tokenize import RegexpTokenizer
    # word_tokenizer = RegexpTokenizer(r'\w+')
    # sentences = sent_tokenize(text)
    # print(sentences)
    # for sentence in sentences:
    #     print(word_tokenizer.tokenize(sentence))

    # # musical_instruments_raw
    # instant_video_raw
    # 'digital_music_5_core_100', 'apps_for_android_5'. 'sports_outdoors_raw'
    # 'cd_vinyl_5', 'kindle_5', 'movies_tv_5', 'electronics_5', 'video_games_5'
    fileNames = ['books_5']
    for fileName in fileNames:
        processor = DataPreprocessor(fileName=fileName)
        processor.pre_filter_active_users_for_seq(20)
        # processor.pre_filter_important_words(word_num=30000)
        # processor.pre_filter_active_users(wordNum=1, itemNum=15)
        # processor.pre_process()
        # processor.printSpeicalWords()
        # processor.printWordVectorsAndMappings(vec_size=200)
        # processor.split_UserRatio()
        # processor.split_UserTimeRatio()
        # processor.split_UserTimeLOO()


    # word_lemmizer = WordNetLemmatizer()
    # print(word_lemmizer.lemmatize('sfsdfea'))

    # processor.printWrongCorrectSpellings()
    # items = [[0, 32], [1, 12], [2, 45], [3, 5], [4, 5], [5, 5]]
    # print(str(sorted(items, key=itemgetter(1))))
    # items = sorted(items, key=itemgetter(1))
    # trainItems = items[0:2]
    #
    # validItems = items[2:4]
    # testItems = items[4:len(items)]
    # print(trainItems)
    # print(validItems)
    # print(testItems)

