import sys
import numpy as np
import argparse
from model import Catboost_CV as M
from feature import FeatureExtractor

def random_params():
    params = []
    for learning_rate in [0.02, 0.05]:
        for iterations in [500, 1000]:
            for depth in [4, 6]:
                for l2_leaf_reg in [20]:
                    for subsample in [0.8]:
                        param = {
                            'learning_rate': learning_rate,
                            'iterations': iterations,
                            'depth': depth,
                            'l2_leaf_reg': l2_leaf_reg,
                            'loss_function': 'Logloss',
                            'verbose': False,
                            'n_splits': 5,
                            'random_state': 456,
                            'optimize_rounds': True
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
