import sys
import numpy as np
import argparse
from model import XGBoost_CV as M
from feature import FeatureExtractor

choice = np.random.choice
def random_params():
    return {
        'colsample_bytree': choice([0.5, 0.6, 0.7, 0.8, 0.9]),
        'eta': choice([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]),
        'features': FeatureExtractor.recommended_features(),
        'max_depth': choice([3, 4, 5, 6]),
        'n_splits': choice([3, 4, 5]),
        'num_boost_round': 5000,
        'objective': 'rank:pairwise',
        'random_state': 1234,
        'seed': 1234,
        'silent': True,
        'subsample': choice([0.5, 0.6, 0.7, 0.8, 0.9]),
    }

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
        '--num-runs',
        '-n',
        dest='num_runs',
        required=False,
        type=int
    )
    args = parser.parse_args()
    for i in xrange(args.num_runs):
        model = M(args.data_dir, random_params())
        model.train_predict_eval_and_log()
