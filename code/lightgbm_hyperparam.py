import sys
import numpy as np
import argparse
from model import Lightgbm_CV as M
from feature import FeatureExtractor

choice = np.random.choice
def random_params():
    params = []
    for learning_rate in [0.01][::-1]:
        for max_bin in [10]:
            for max_depth in [5]:
                for feature_fraction in [1]:
                    for bagging_fraction in [1]:
                        param = {
                                #'metrics': 'auc', 
                                'learning_rate': learning_rate, 
                                #'max_depth': max_depth, 
                                #'num_leaves': 2**(max_depth),  # see here: https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
                                'max_bin': max_bin,  
                                'objective': 'binary', 
                                #'feature_fraction': feature_fraction,
                                #'bagging_fraction': bagging_fraction,
                                #'bagging_freq':10,  
                                #'min_data': 500,
                                'n_splits': 5,
                                'n_jobs': 4,
                                'subsample': 0.8,
                                'subsample_freq': 10,
                                'colsample_bytree': 0.8,
                                #'min_child_samples': 500, 
                                'n_estimators': 1250,
                                'random_state': 1025}
                        params.append(param)
    return params

def optimal_param():
    return {
        'colsample_bytree': 0.8,
        'learning_rate': 0.05,
        'max_bin': 10,
        'max_depth': 5,
        'n_estimators': 2000,
        'n_jobs': 4,
        'n_splits': 5,
        'num_leaves': 32,
        'objective': 'binary',
        'random_state': 1025,
        'subsample': 0.8,
        'subsample_freq': 10,
    }
#        '''
#        'excluded_features': [
#            'mean_range_ps_1nd_14',
#            'med_range_ps_1nd_14',
#            'oh_ps_car_10_cat_2',
#            'mean_range_ps_car_11',
#            'med_range_ps_car_12',
#            'oh_ps_ind_04_cat_-1',
#            'med_range_ro_ps_car_04_cat',
#            'med_range_ro_ps_car_06_cat',
#            'med_range_ps_ind_15',
#            'med_range_ro_ps_ind_05_cat',
#            'med_range_ps_ind_03',
#            'med_range_ps_ind_01',
#            'oh_ps_ind_14_2',
#        ]
#        '''
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tuning hyperparameter')
    parser.add_argument(
        '--data-dir',
        '-d',
        dest='data_dir',
        required=True,
        type=str
    )
    parser.add_argument(
        '--mode',
        '-m',
        dest='mode',
        help='random|best',
        required=True,
        type=str
    )
    args = parser.parse_args()
    if args.mode == 'random':
        for param in random_params():
            model = M(args.data_dir, param)
            model.train_predict_eval_and_log()
    elif args.mode == 'best':
        param = optimal_param()
        model = M(args.data_dir, param)
        model.train_predict_eval_and_log()
    else:
        raise Exception('wrong mode')
        model.train_predict_eval_and_log()
