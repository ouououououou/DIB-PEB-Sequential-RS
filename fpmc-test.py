
from recommender.FPMC import FPMCRecommender
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
        'factor_lambda': 0.1,
        'goal': 'ranking',
        'verbose': False,
        'input_length': 15,
        'target_length': 1,
        'dropout_keep': 0.8,
        'item_fc_dim': [32],
        'capsule_num': 1,
        'dynamic_routing_iter': 3,
        'filter_num': 5,
        'early_stop': True,
        'capsule_num_1': 10,
        'capsule_num_2': 5,
        'capsule_num_3': 2,
        'pose_len': 16,
        'capsule_lambda': 0.01,
        'cnn_lam': 0.001,
        'mlp_lam': 0.001,
        'capsule_lam': 0.001,
        'add_time': False,
        'time_user': False,
        'routing_type': 'em',
        'at_b': True,
        'at_c': True,
        'useRating': True,
        'share_user_para_b': True,
        'share_user_para_c': True,
        'user_as_miu_bias': False,
        'user_reg_cost': False,
        'cluster_pretrain': False,
        'num_cluster': 10,
        'random_seed': 123,
        'rating_threshold': 100,
        'trainBatchSize': 512,
        'testBatchSize': 512,
        'numFactor': 128,
        'topN': 10,
        'familiar_user_num': 5,
        'negative_numbers': 25,
        'eval_item_num': 1000,
        'numK': 15,
        'need_process_data': False,
        'csv': True,
        'test_sparse_user': True,
        'merge_sparse_user': False,
        'khsoft': False,
        'save_path': 'saved_model',
        'save_model': True,
        'load_model': False,
        'using_model': 'FPMC'
    }

    for fileName in ['newkin-seq']:
        config['fileName'] = fileName

        dataModel = SequenceDataModel(config)
        dataModel.buildModel()

        for input_len in [15]:
            for target_len in [1]:

                dataModel.generate_sequences_hor(input_len, target_len)
                config['input_length'] = input_len
                config['target_length'] = target_len

                recommender = FPMCRecommender(dataModel, config)
                recommender.run()


































