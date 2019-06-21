
from recommender.Caser import CaserRecommender
from data_model.SequenceDataModel import SequenceDataModel
import gc


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
        'factor_lambda': 0.01,
        'goal': 'ranking',
        'verbose': False,
        'input_length': 5,
        'target_length': 3,
        'hor_filter_num': 16,
        'ver_filter_num': 4,
        'dropout_keep': 0.65,
        'dropout_user': 0.45,
        'dropout_item': 0.45,
        'item_fc_dim': [32],
        'early_stop': True,
        'random_seed': 123,
        'useRating': True,
        'familiar_user_num': 5,
        'negative_numbers': 25,
        'eval_item_num': 500,
        'trainBatchSize': 512,
        'testBatchSize': 512,
        'numFactor': 128,
        'topN': 10,
        'numK': 10,
        'need_process_data': False,
        'csv': True,
        'test_sparse_user': True,
        'merge_sparse_user': False,
        'khsoft': False,
        'save_path': 'saved_model',
        'save_model': True,
        'load_model': True,
        'using_model': 'Caser'
    }

    for fileName in ['ml-100k']:
        config['fileName'] = fileName

        dataModel = SequenceDataModel(config)
        dataModel.buildModel()

        for input_length in [5]:
            for target_length in [1]:
                for drop_keep in [0.8]:

                    config['input_length'] = input_length
                    config['target_length'] = target_length
                    config['dropout_keep'] = drop_keep

                    dataModel.generate_sequences_hor(input_length, target_length)

                    recommender = CaserRecommender(dataModel, config)
                    recommender.run()















