import sys
import numpy as np
import argparse
from model import Sklearn_gradientboosting as M
from feature import FeatureExtractor

def random_params():
    params = []
    for learning_rate in [0.01, 0.03, 0.05]:
        for n_estimators in [500, 1000]:
            for max_depth in [4, 6]:
                for subsample in [0.8]:
                    param = {
                        'learning_rate': learning_rate,
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'subsample': subsample,
                        'n_splits': 5,
                        'random_state': 456,
                        'verbose': 1
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
