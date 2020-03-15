from data_model.SequenceDataModel import SequenceDataModel
from recommender.RUM_Ksoft_mulcha import RUMIRecommender_Ksoft
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
        'factor_lambda': 0.01,
        'goal': 'ranking',
        'verbose': False,
        'hor_filter_num': 16,
        'ver_filter_num': 4,
        'dropout_keep': 0.5,
        'item_fc_dim': [32],
        'capsule_num': 1,
        'dynamic_routing_iter': 3,
        'filter_num': 5,
        'early_stop': True,
        'loss': 'bpr',
        'random_seed': 123,
        'useRating': True,
        'input_length': 5,
        'target_length': 1,
        'learnRate': 0.002,
        'maxIter': 2000,
        'trainBatchSize': 512,
        'testBatchSize': 512,
        'numFactor': 128,
        'cell_numbers': 128,
        'topN': 10,
        'layers': 1,
        'dropout_user': 0.45,
        'dropout_memory': 0.45,
        'dropout_userL': 0.5,
        'familiar_user_num': 5,
        'khsoft': False,
        'gru_model': False,
        'decrease soft': True,
        'loss_type': 'PEB',
        'negative_numbers': 25,
        'eval_item_num': 1000,
        'numK': 15,
        'old_loss': True,
        'target_weight': 0.5,
        'dynamic_item_type': 'user',
        'merge_type': 'add',
        'need_process_data': False,
        'csv': False,
        'test_sparse_user': True,
        'merge_sparse_user': False,
        'save_path': 'saved_model',
        'save_model': True,
        'load_model': False,
        'using_model': 'DIB-PEB-ASS',
        'online_learning': False
    }

    for fileName in ['newkin-seq']:
        config['fileName'] = fileName

        "(5,1)"
        dataModel = SequenceDataModel(config)
        dataModel.buildModel()

        for input_len in [5]:
            for target_len in [1]:
                "(5,1)"
                dataModel.generate_sequences_hor(input_len, target_len)

                for T in [3]:

                    for filter_num in [5]:
                        for capsule_num in [5]:
                            for loss in ['cross_entropy']:
                                config['input_length'] = input_len
                                config['target_length'] = target_len
                                config['loss'] = loss
                                "(5,1)"
                                recommender = RUMIRecommender_Ksoft(dataModel, config)
                                recommender.run()



























