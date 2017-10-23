import sys
import numpy as np
import argparse
from model import XGBoost_CV as M
from feature import FeatureExtractor

choice = np.random.choice
def random_params():
    params = []
    for colsample_bytree in [0.8, 0.9]:
        for eta in [0.01, 0.03, 0.05]:
            for max_depth in [4, 5, 6]:
                for subsample in [0.8, 0.9]:
                    for scale_pos_weight in [5]:
                        param = {
                            'colsample_bytree': colsample_bytree,
                            'eta': eta,
                            'max_depth': max_depth,
                            'n_splits': 5,
                            'n_jobs': 4,
                            'num_boost_round': 5000,
                            'objective': 'binary:logistic',
                            'random_state': 135,
                            'seed': 246,
                            'silent': True,
                            'subsample': subsample,
                            'scale_pos_weight': scale_pos_weight
                        }
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
