import os
import sys
import pandas as pd
import datetime as dt
import argparse
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from configs.base_config import *

class Training_data(object):
    def __init__(self, directory, id_col, label_col):
        self._id_col = id_col
        self._label_col = label_col
        self._train = pd.read_csv(os.path.join(directory, train_file))
        return None

    def output_small_data_for_sanity_check(self, output_directory, nrows=500):
        if os.path.exists(output_directory):
            output_directory = os.path.abspath(output_directory)
            print ('WARNING: {} already exists, will overwrite it'.format(output_directory))
        else:
            os.system('mkdir -p {}'.format(output_directory))

        df = shuffle(self._train)
        train = df[:nrows].copy()
        test = df[nrows:2*nrows].copy()
        test_target = test[[self._label_col]]
        del test[self._label_col]
        train.to_csv(os.path.join(output_directory, train_file), index=False)
        test.to_csv(os.path.join(output_directory, test_file), index=False)
        test_target.to_csv(os.path.join(output_directory, test_target_file), index=False)

    # Default output_directory is base_data_dir + today's date
    def output_split(self, validation_proportion=0.3, output_directory=None):
        if output_directory is None:
            output_directory = os.path.join(base_data_dir, '{}'.format(dt.date.today ()))

        if os.path.exists(output_directory):
            output_directory = os.path.abspath(output_directory)
            raise Exception('{} already exists'.format(output_directory))
        else:
            df = self._train.copy()
            test_size = int(df.shape[0] * validation_proportion)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
            indices = sss.split(X=df, y=df.target).next()
            train = df.iloc[indices[0]]
            test = df.iloc[indices[1]]
            test_target = test[[self._label_col]]
            del test[self._label_col]
            # create directory right before output files
            os.system('mkdir -p {}'.format(output_directory))
            train.to_csv(os.path.join(output_directory, train_file), index=False)
            test.to_csv(os.path.join(output_directory, test_file), index=False)
            test_target.to_csv(os.path.join(output_directory, test_target_file), index=False)

    def kfold(self, n_splits, random_state=3223):
        results = []
        sss = StratifiedKFold(n_splits, random_state=random_state)
        for (train_ind, valid_ind) in sss.split(df, df.target):
            results.append((df.iloc[train_ind], df.iloc[valid_idn]))
        return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        '-m',
        dest='mode',
        required=True,
        type=str,
        help=' mode (sanity|split)'
    )
    args = parser.parse_args()
    training_data = Training_data(raw_data_dir, id_col, label_col)
    if args.mode == 'sanity':
        training_data.output_small_data_for_sanity_check(output_directory=sanity_data_dir)
    elif args.mode == 'split':
        training_data.output_split()
    else:
        raise Exception('unknown mode {}'.format(args.mode))
