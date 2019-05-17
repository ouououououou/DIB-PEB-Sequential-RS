import time
import tensorflow as tf

class RatingEvaluator:

    def __init__(self, r_label):
        '''
        :param r_label: the rating label from test set
        '''

        self.r_pred = None
        self.r_label = r_label
        self.rmse = None
        self.mae = None

    def set_r_pred(self, r_pred):
        self.r_pred = r_pred



    '''RMSE and MAE'''
    def cal_RMSE_and_MAE(self):
        self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.r_label - self.r_pred)))
        self.mae = tf.reduce_mean(tf.abs(self.r_pred - self.r_label))
        start = time.clock()
        rmseValue = self.rmse.eval()
        maeValue = self.mae.eval()

        end = time.clock()
        return rmseValue, maeValue, (end - start)


if __name__ == '__main__':
    user_items_train = {1: [0, 1, 2, 3, 4, 5],
                        0: [0, 1, 3, 6]}
    groundTruthList = {
                       1: [7, 8, 9]}
    topK = 3
    numItem = set()
    numItem.add(4)
    numItem.add(5)
    numItem.add(7)
    numItem.add(8)
    numItem.add(9)

    testMatrix = {(1, 7): 5,
                  (1, 8): 4,
                  (1, 9): 3,
                  (0, 4): 5,
                  (0, 5): 4,
                  (0, 7): 3,
                  }
    predLists = {1: [7, 8, 9],
                 }












