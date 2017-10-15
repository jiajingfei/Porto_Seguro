import os
import sys
import pandas as pd
import datetime as dt
import argparse
import getpass
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils import gini_normalized
from configs.base_config import (
    id_col,
    label_col,
    get_data_dir,
    data_train_file,
    data_test_file,
    data_test_target_file,
    data_readme,
    data_raw_dir,
    data_sanity_dir,
    get_pred_dir,
    pred_filename,
    pred_log_file,
)

class Training_data(object):
    def __init__(self, dir_or_df):
        if os.path.isdir(get_data_dir(dir_or_df)):
            self._train = pd.read_csv(data_train_file(dir_or_df))
        else:
            self._train = dir_or_df.copy()
        return None

    def __output_generated_data(self, train, test, output_directory):
        if not os.path.exists(get_data_dir(output_directory)):
            os.system('mkdir -p {}'.format(get_data_dir(output_directory)))
        test_target = test[[id_col, label_col]]
        del test[label_col]
        train.to_csv(data_train_file(output_directory), index=False)
        test.to_csv(data_test_file(output_directory), index=False)
        test_target.to_csv(data_test_target_file(output_directory), index=False)
        with open(data_readme(output_directory), 'w') as f:
            readme_lines = [
                'this directory is for generated training data and validation data\n',
                'it should contain a train file with labels and a test file without labels\n',
                'and it may contains a hidden file which is the labels for the test file'
            ]
            f.writelines(readme_lines)

    def output_small_data_for_sanity_check(self, nrows=500):
        data_dir = get_data_dir(data_sanity_dir)
        if not os.path.exists(data_dir):
            os.system('mkdir -p {}'.format(data_dir))
        if os.path.exists(data_dir):
            data_dir = os.path.abspath(data_dir)
            print ('WARNING: {} already exists, will overwrite it'.format(data_dir))

        df = shuffle(self._train)
        train = df[:nrows].copy()
        test = df[nrows:2*nrows].copy()
        test_target = test[[label_col]]
        self.__output_generated_data(train, test, data_sanity_dir)

    # Default output_directory is base_data_dir + today's date
    def output_split(self, validation_proportion=0.3, output_directory=None):
        if output_directory is None:
            output_directory = '{}'.format(dt.date.today ())

        data_dir = get_data_dir(output_directory)
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
            self.__output_generated_data(train, test, output_directory)

    def kfold(self, n_splits, random_state):
        results = []
        sss = StratifiedKFold(n_splits, random_state=random_state)
        for (train_ind, valid_ind) in sss.split(df, df.target):
            results.append((df.iloc[train_ind], df.iloc[valid_idn]))
        return results


class Prediction(object):

    def __init__(self, df, data_dir, identifier):
        self._dir = data_dir
        self._identifier = identifier
        self._df = df.sort_values(id_col).reset_index(drop=True)

    def eval(self):
        filename = data_test_target_file(self._dir)
        if not os.path.isfile(filename):
            return None

        test_target = pd.read_csv(filename)
        test_target = test_target.sort_values(id_col).reset_index(drop=True)
        if not test_target[id_col].equals(self._df[id_col]):
            raise Exception('id cols are not consistent')
        else:
            return gini_normalized(test_target[label_col], self._df[label_col])

    def eval_output_and_register(self, filename):
        gini = self.eval()
        pred_dir = get_pred_dir(self._dir)
        pred_log_file = pred_log_file(self._dir)
        pred_filename = pred_filename(self._dir, filename, self._identifier)
        if not os.path.exists(pred_dir):
            f = open(pred_log_file, 'ab')
            f.write('data_dir,gini,user,time,identifier')
        else:
            f = open(pred_log_file, 'ab')
        new_line = '{},{},{},{},{}'.format(
            self._dir,
            gini,
            getpass.getuser(),
            dt.datetime.now(),
            self._identifier
        )

def test_predcition():
    if not os.path.exists(get_data_dir(data_sanity_dir)):
        training_data = Training_data(data_raw_dir)
        training_data.output_small_data_for_sanity_check(data_sanity_dir)
    df = pd.read_csv(data_test_target_file(data_sanity_dir))
    prediction = Prediction(df, data_sanity_dir, identifier="sanity_check")
    normalized_gini = prediction.eval()
    assert (np.isclose(normalized_gini, 1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='split training data into training and validation dataset')
    parser.add_argument(
        '--mode',
        '-m',
        dest='mode',
        required=True,
        type=str,
        help=' mode (sanity|split)'
    )
    args = parser.parse_args()
    training_data = Training_data(data_raw_dir)
    if args.mode == 'sanity':
        training_data.output_small_data_for_sanity_check()
    elif args.mode == 'split':
        training_data.output_split()
    else:
        raise Exception('unknown mode {}'.format(args.mode))
