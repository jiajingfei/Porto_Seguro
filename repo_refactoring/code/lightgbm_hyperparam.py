import sys
import numpy as np
import argparse
from model import lightgbm as M
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tuning hyperparameter')
    parser.add_argument(
        '--data-dir',
        '-d',
        dest='data_dir',
        required=True,
        type=str
    )
    '''
    parser.add_argument(
        '--num-runs',
        '-n',
        dest='num_runs',
        required=False,
        type=int
    )
    '''
    args = parser.parse_args()
    for param in random_params():
        model = M(args.data_dir, param)
        model.train_predict_eval_and_log()
