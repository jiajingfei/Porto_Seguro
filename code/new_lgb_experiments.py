import sys
import numpy as np
import argparse
from new_model import Lightgbm_CV as M
from new_feature import FeatureExtractor as FE
from new_feature import Feature as F
from new_feature import Action_type as A

choice = np.random.choice
def rand():
    return np.random.uniform()

def choose_action_type(feature, action_types, p=None, drop_calc=True, drop_with_p=0):
    if rand() < p:
        return A.dropping
    elif drop_calc and feature.startswith('ps_calc'):
        return A.dropping
    else:
        return choice(action_types, p=p)

def random_run(
        action_types,
        feature_dirs,
        drop_with_p = 0,
        learning_rate = [0.04, 0.05, 0.06],
        max_bin = [8, 9, 10],
        max_depth = [4, 5, 6],
        feature_fraction = [0.8, 0.9, 1],
        bagging_fraction = [0.8, 0.9, 1],
        subsample = [0.7, 0.8],
        subsample_freq = [10],
        colsample_bytree = [0.7, 0.8],
        n_estimators = [1500]
):
    feature_dir = choice(feature_dirs)
    param = {}
    param['learning_rate'] = choice(learning_rate)
    param['max_bin'] = choice(max_bin)
    param['max_depth'] = choice(max_depth)
    param['feature_fraction'] = choice(feature_fraction)
    param['bagging_fraction'] = choice(bagging_fraction)
    param['subsample'] = choice(subsample)
    param['subsample_freq'] = choice(subsample_freq)
    param['colsample_bytree'] = choice(colsample_bytree)
    param['n_estimators'] = choice(n_estimators)
    def get_action_type(feature):
        return choose_action_type(feature, action_types, drop_with_p=drop_with_p)
    model = M(feature_dir, param, get_action_type)
    model.train_predict_eval_and_log()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tuning hyperparameter')
    parser.add_argument(
        '--feature-dirs',
        '-f',
        dest='feature_dirs',
        help='list of feature dirs',
        required=True,
        type=str
    )
    parser.add_argument(
        '--action-types',
        '-a',
        dest='action_types',
        help='list of {}|{}|{}'.format(A.raw, A.prefer_one_hot, A.prefer_reorder),
        required=True,
        type=str
    )
    parser.add_argument(
        '--drop-with-p',
        '-d',
        dest='drop_with_p',
        help='drop feature with probability',
        default=0.
        type=float
    )
    parser.add_argument(
        '--num-runs',
        '-r',
        dest='num_runs',
        help='num of runs, default is no limit',
        type=int,
        default=10000000
    )
    args = parser.parse_args()
    feature_dirs = args.feature_dirs.split(',')
    for _ in arange(args.num_runs):
        random_run(
            args.action_types,
            feature_dirs,
            drop_with_p = args.drop_with_p
        )
