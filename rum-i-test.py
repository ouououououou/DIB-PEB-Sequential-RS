
from recommender.RUM_I import RUMIRecommender
from data_model.SequenceDataModel import SequenceDataModel
import gc
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"


if __name__ == '__main__':

    # instant_video_5_core, digital_music_5_core_100, baby_5_core_100  apps_for_android_5 video_games_5

    config = {
        'generate_seq': True,
        'splitterType': 'userTimeRatio',
        'fileName': 'ml-100k',
        'trainType': 'test',
        'threshold': 0,
        'learnRate': 0.001,
        'maxIter': 1000,
        'trainBatchSize': 512,
        'testBatchSize': 512,
        'numFactor': 128,
        'topN': 10,
        'factor_lambda': 0.01,
        'goal': 'ranking',
        'verbose': False,
        'input_length': 5,
        'target_length': 1,
        'hor_filter_num': 16,
        'ver_filter_num': 4,
        'dropout_keep': 0.8,
        'item_fc_dim': [32],
        'capsule_num': 1,
        'dynamic_routing_iter': 3,
        'eval_item_num': 1000,
        'filter_num': 5,
        'early_stop': True,
        'loss': 'bpr',
        'loss_type': 'bpr',
        'random_seed': 123,
        'useRating': True,
        'layers': 1,
        'negative_numbers': 25,
        'familiar_user_num': 5,
        'need_process_data': False,
        'csv': True,
        'test_sparse_user': True,
        'merge_sparse_user': False,
        'khsoft': False,
        'save_path': 'saved_model',
        'save_model': True,
        'load_model': False,
        'using_model': 'RUMI'
    }

    for fileName in ['newkin-seq']:
        config['fileName'] = fileName

        dataModel = SequenceDataModel(config)
        dataModel.buildModel()

        for input_len in [5]:
            for target_len in [1]:

                dataModel.generate_sequences_hor(input_len, target_len)

                for T in [3]:

                    for filter_num in [5]:
                        for capsule_num in [5]:
                            for loss in ['bpr']:
                                config['input_length'] = input_len
                                config['target_length'] = target_len
                                config['loss'] = loss

                                recommender = RUMIRecommender(dataModel, config)
                                recommender.run()



























