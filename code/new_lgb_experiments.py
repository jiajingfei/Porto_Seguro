import sys
import numpy as np
import argparse
import timeit
from new_model import Lightgbm_CV as M
from new_feature import FeatureExtractor as FE
from new_feature import Feature as F
from new_feature import Action_type as A

choice = np.random.choice
def rand():
    return np.random.uniform()

def choose_action_type(feature, action_types, p=None, drop_calc=True, drop_with_p=0):
    if rand() < drop_with_p:
        return A.dropping
    elif drop_calc and feature.startswith('ps_calc'):
        return A.dropping
    else:
        return choice(action_types, p=p)

def random_run(
        action_types,
        feature_dirs,
        suffix=None,
        drop_with_p = 0,
        action_types_p = None,
        learning_rate = [0.04, 0.05, 0.06],
        max_bin = [8, 10, 12],
        max_depth = [5],
        subsample = [0.8],
        subsample_freq = [10],
        colsample_bytree = [0.8],
        lambda_l1 = [1, 3, 9, 27],
        lambda_l2 = [1, 3, 9, 27],
        verbose = [-1],
        n_estimators = [1500]
):
    feature_dir = choice(feature_dirs)
    param = {}
    param['objective'] = 'binary'
    param['learning_rate'] = choice(learning_rate)
    param['max_bin'] = choice(max_bin)
    param['max_depth'] = choice(max_depth)
    param['num_leaves'] = 2 ** param['max_depth']
    param['subsample'] = choice(subsample)
    param['subsample_freq'] = choice(subsample_freq)
    param['colsample_bytree'] = choice(colsample_bytree)
    param['n_estimators'] = choice(n_estimators)
    param['lambda_l1'] = choice(lambda_l1)
    param['lambda_l2'] = choice(lambda_l2)
    param['verbose'] = choice(verbose)
    def get_action_type(feature):
        return choose_action_type(
            feature, action_types, drop_with_p=drop_with_p, p=action_types_p
        )
    model = M(feature_dir, param, get_action_type)
    model.train_predict_eval_and_log(suffix)

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
        help='list of {}'.format(A.all()),
        required=True,
        type=str
    )
    parser.add_argument(
        '--action-types-ps',
        '-ap',
        dest='action_types_p',
        help='list of floats',
        type=str
    )
    parser.add_argument(
        '--drop-with-p',
        '-d',
        dest='drop_with_p',
        help='drop feature with probability',
        default=0.,
        type=float
    )
    parser.add_argument(
        '--num-runs',
        '-n',
        dest='num_runs',
        help='num of runs, default is no limit',
        type=int,
        default=10000000
    )
    parser.add_argument(
        '--suffix',
        '-s',
        dest='suffix',
        help='feature directory suffix',
        type=str,
    )
    args = parser.parse_args()
    feature_dirs = args.feature_dirs.split(',')
    action_types = args.action_types.split(',')
    if args.action_types_p is not None:
        action_types_p = [float(s) for s in args.action_types_p.split(',')]
    else:
        action_types_p = None
    start_time = timeit.default_timer()
    for _ in xrange(args.num_runs):
        random_run(
            action_types,
            feature_dirs,
            suffix=args.suffix,
            drop_with_p = args.drop_with_p,
            action_types_p = action_types_p
        )
    elapsed = timeit.default_timer() - start_time
    print '{} runs finished, elapsed time = {}'.format(args.num_runs, elapsed)
