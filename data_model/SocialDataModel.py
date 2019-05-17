
import random
import logging
import os.path
import time
from data_model.BasicDataModel import BasicDataModel

class SocialDataModel(BasicDataModel):

    def __init__(self, config):
        super(SocialDataModel, self).__init__(config)

        self.user_trustees = {}
        self.item_pad_num = 0
        self.pad_mask = {}

    def read_social_data(self):
        f = open(self.inputPath + '/' + self.splitterType + '/social.txt')
        lines = f.readlines()
        truster_set = set()
        trustee_set = set()
        numTrust = 0
        # check if there is time column
        for line in lines:
            trusterId, trusteeId, value = line.strip().split(' ')

            if trusterId not in self.userIdToUserIdx or trusteeId not in self.userIdToUserIdx:
                continue

            truster_idx = self.userIdToUserIdx[trusterId]
            trustee_idx = self.userIdToUserIdx[trusteeId]
            truster_set.add(truster_idx)
            trustee_set.add(trustee_idx)

            self.user_trustees.setdefault(truster_idx, [])
            self.user_trustees[truster_idx].append(trustee_idx)
            numTrust += 1

        self.numRatingBeforeSplit = len(self.wholeDataSet)

        self.logger.info('numTrust: ' + str(numTrust))
        self.logger.info('numTruster: ' + str(len(truster_set)))
        self.logger.info('numTrustee: ' + str(len(trustee_set)))
        self.logger.info('social_density: ' + str(numTrust / (len(truster_set) * len(trustee_set))))

    def calcu_item_pad_num(self):
        # calculate the max item number per user
        for userIdx, itemList in self.user_items_train.items():
            length = len(itemList)
            if length > self.item_pad_num:
                self.item_pad_num = length
        # pad each user's rating list with item 0
        for userIdx, itemList in self.user_items_train.items():
            copied_item_list = list(itemList)
            maskList = [1] * len(itemList)
            while len(copied_item_list) < self.item_pad_num:
                copied_item_list.append(0)
                maskList.append(0)
            self.user_items_train_paded[userIdx] = copied_item_list
            self.pad_mask[userIdx] = maskList




    def buildModel(self):
        self.pre_process()
        self.split_user_loo()
        self.logger.info("\n###### information of DataModel ######\n")
        self.readData()
        self.read_social_data()
        self.calcu_item_pad_num()

        self.generateEvalItemsForEachUser()

        self.printInfo()
