
from recommender.GRU4Rec import GRU4RecRecommender
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
        'learnRate': 0.002,
        'maxIter': 1000,
        'trainBatchSize': 512,
        'testBatchSize': 512,
        'numFactor': 100,
        'topN': 10,
        'factor_lambda': 0.01,
        'goal': 'ranking',
        'verbose': False,
        'seq_length': 5,
        'input_length': 5,
        'dropout_keep': 0.45,
        'dropout_item': 0.5,
        'dropout_context1': 0.5,
        'dropout_context2': 0.5,
        'rnn_unit_num': 128,
        'rnn_layer_num': 1,
        'rnn_cell': 'GRU',
        'eval_item_num': 1000,
        'seq_direc': 'hor',
        'early_stop': True,
        'random_seed': 123,
        'useRating': True,
        'loss_type': 'bpr',
        'target_weight': 0.8,
        'numK': 1,
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
        'using_model': 'GRU4Rec'
    }

    for fileName in ['newkin-seq']:
        config['fileName'] = fileName
        seq_length = config['seq_length']

        dataModel = SequenceDataModel(config)
        dataModel.buildModel()

        for seq_length in [5]:
            config['seq_length'] = seq_length
            recommender = GRU4RecRecommender(dataModel, config)
            recommender.run()

        # for rnn_unit_num in [32, 64, 128, 256]:
        #     for rnn_layer_num in [1]:
        #         for dp_kp in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        #             config['dropout_keep'] = dp_kp
        #             config['rnn_unit_num'] = rnn_unit_num
        #             config['rnn_layer_num'] = rnn_layer_num
























