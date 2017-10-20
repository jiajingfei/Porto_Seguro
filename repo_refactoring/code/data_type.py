from __future__ import print_function
import os
import pandas as pd
import datetime as dt
import argparse
import getpass
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from utils import gini_normalized, save_df_to_file, save_to_file
import config

class Training_data(object):
    def __init__(self, data_dir):
        self._train = pd.read_csv(config.data_train_file(data_dir))

    def __output_generated_data(self, train, test, output_data_dir):
        if not os.path.exists(config.get_data_dir(output_data_dir)):
            os.system('mkdir -p {}'.format(config.get_data_dir(output_data_dir)))
        test_target = test[[config.id_col, config.label_col]]
        del test[config.label_col]
        train_file = config.data_train_file(output_data_dir)
        test_file = config.data_test_file(output_data_dir)
        if os.path.exists(train_file):
            raise Exception('{} already exists'.format(train_file))
        if os.path.exists(test_file):
            raise Exception('{} already exists'.format(test_file))
        train.to_csv(train_file, index=False)
        test.to_csv(test_file, index=False)
        train[config.id_col].to_csv(train_file+'.id', index=False)
        test[config.id_col].to_csv(test_file+'.id', index=False)
        test_target.to_csv(config.data_test_target_file(output_data_dir), index=False)
        with open(config.data_readme_file(output_data_dir), 'w') as f:
            readme_lines = [
                'this directory is for generated training data and validation data\n',
                'it should contain a train file with labels and a test file without labels\n',
                'and it may contains a hidden file which is the label for the test file\n'
            ]
            f.writelines(readme_lines)

    def output_small_data_for_sanity_check(self, nrows=500):
        data_dir = config.get_data_dir(config.data_sanity_dir)
        if not os.path.exists(data_dir):
            os.system('mkdir -p {}'.format(data_dir))
        if os.path.exists(data_dir):
            data_dir = os.path.abspath(data_dir)
            print('WARNING: {} already exists, will overwrite it'.format(data_dir))

        df = shuffle(self._train)
        train = df[:nrows].copy()
        test = df[nrows:2*nrows].copy()
        test_target = test[[config.label_col]]
        self.__output_generated_data(train, test, config.data_sanity_dir)

    # Default output_data_dir is base_data_dir + today's date
    def output_split(self, validation_proportion=0.3, output_data_dir=None):
        if output_data_dir is None:
            output_data_dir = '{}'.format(dt.date.today ())

        data_dir = config.get_data_dir(output_data_dir)
        if os.path.exists(data_dir):
            data_dir = os.path.abspath(data_dir)
            raise Exception('{} already exists'.format(data_dir))
        else:
            df = self._train.copy()
            test_size = int(df.shape[0] * validation_proportion)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
            indices = sss.split(X=df, y=df.target).next()
            train = df.iloc[indices[0]]
            test = df.iloc[indices[1]]
            self.__output_generated_data(train, test, output_data_dir)

    def complete_data_dir_with_ids(self, data_dir):
        '''
        This code is to complete the data, from the id

        Input
        ---------
        data_dir: <str>, 
            what kind of data we try to generate, 
            in {'sanity_data', '2017-10-15', ...}

        Return
        ---------
        Nothing, but create 3 files under the same folder.
        If the files already exist, we will overwrite
        '''

        # get indices
        if data_dir == 'raw_data':
            raise Exception('Should never complete the raw data')

        ids_train_file = config.data_train_file(data_dir) + '.id'
        ids_test_file = config.data_test_file(data_dir) + '.id'
        if not os.path.exists(config.get_data_dir(data_dir)):
            raise Exception('The data dir {} does not exist.'.format(data_dir))

        def load_ids(filename):
            with open(filename, 'r') as f:
                return list(map(int, f.readlines()))

        train_ids = load_ids(ids_train_file)
        test_ids = load_ids(ids_test_file)

        train = self._train[self._train[config.id_col].isin(train_ids)]
        test = self._train[self._train[config.id_col].isin(test_ids)]
        self.__output_generated_data(train, test, output_data_dir=data_dir)

    def kfold(self, n_splits, random_state):
        df = self._train.copy()
        if n_splits is None:
            return [(df, None)]
        else:
            assert (n_splits > 1)
            results = []
            sss = StratifiedKFold(n_splits, random_state=random_state)
            for (train_ind, valid_ind) in sss.split(df, df.target):
                results.append((df.iloc[train_ind], df.iloc[valid_ind]))
            return results


class Prediction(object):

    def __init__(self, df, data_dir, identifier):
        self._dir = data_dir
        self._identifier = identifier
        self._df = df.sort_values(config.id_col).reset_index(drop=True)

    def _eval(self):
        filename = config.data_test_target_file(self._dir)
        if not os.path.isfile(filename):
            return None

        test_target = pd.read_csv(filename)
        test_target = test_target.sort_values(config.id_col).reset_index(drop=True)
        if not test_target[config.id_col].equals(self._df[config.id_col]):
            raise Exception('id cols are not consistent')
        else:
            return gini_normalized(test_target[config.label_col], self._df[config.label_col])

    def eval_and_save(self, filename_key):
        gini = self._eval()
        save_df_to_file(
            self._df,
            config.pred_filename(self._dir, filename_key, self._identifier),
            overwrite=False
        )
        return self._eval()

def test_predcition():
    if not os.path.exists(config.get_data_dir(config.data_sanity_dir)):
        training_data = Training_data(config.data_raw_dir)
        training_data.output_small_data_for_sanity_check(config.data_sanity_dir)
    df = pd.read_csv(config.data_test_target_file(config.data_sanity_dir))
    prediction = Prediction(df, config.data_sanity_dir, identifier="sanity_check")
    normalized_gini = prediction._eval()
    assert (np.isclose(normalized_gini, 1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='split training data into training and validation dataset')
    parser.add_argument(
        '--mode',
        '-m',
        dest='mode',
        required=True,
        type=str,
        help='mode (sanity|split|complete)'
    )
    parser.add_argument(
        '--data-dir',
        '-d',
        dest='data_dir',
        required=False,
        type=str,
        help='data dir, must be none if mode = sanity'
    )
    args = parser.parse_args()
    training_data = Training_data(config.data_raw_dir)
    if args.mode == 'sanity':
        assert (args.data_dir is None)
        training_data.output_small_data_for_sanity_check()
    elif args.mode == 'split':
        training_data.output_split(output_data_dir=args.data_dir)
    elif args.mode == 'complete':
        training_data.complete_data_dir_with_ids(args.data_dir)
    else:
        raise Exception('unknown mode {}'.format(args.mode))
