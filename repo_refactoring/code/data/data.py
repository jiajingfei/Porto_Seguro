import os
import sys
import pandas as pd
import datetime as dt
import argparse
import getpass
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils import gini_normalized, save_df_to_file
import configs.base_config as config
'''
from configs.base_config import config
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
    pred_log_file
)
'''

class Training_data(object):
    def __init__(self, dir_or_df):
        if os.path.isdir(config.get_data_dir(dir_or_df)):
            self._train = pd.read_csv(config.data_train_file(dir_or_df))
        else:
            self._train = dir_or_df.copy()
        return None

    def __output_generated_data(self, train, test, output_directory):
        if not os.path.exists(config.get_data_dir(output_directory)):
            os.system('mkdir -p {}'.format(config.get_data_dir(output_directory)))
        test_target = test[[config.id_col, config.label_col]]
        del test[config.label_col]
        train.to_csv(config.data_train_file(output_directory), index=False)
        test.to_csv(config.data_test_file(output_directory), index=False)
        test_target.to_csv(config.data_test_target_file(output_directory), index=False)
        with open(config.data_readme(output_directory), 'w') as f:
            readme_lines = [
                'this directory is for generated training data and validation data\n',
                'it should contain a train file with labels and a test file without labels\n',
                'and it may contains a hidden file which is the labels for the test file'
            ]
            f.writelines(readme_lines)

    def output_small_data_for_sanity_check(self, nrows=500):
        data_dir = config.get_data_dir(config.data_sanity_dir)
        if not os.path.exists(data_dir):
            os.system('mkdir -p {}'.format(data_dir))
        if os.path.exists(data_dir):
            data_dir = os.path.abspath(data_dir)
            print ('WARNING: {} already exists, will overwrite it'.format(data_dir))

        df = shuffle(self._train)
        train = df[:nrows].copy()
        test = df[nrows:2*nrows].copy()
        test_target = test[[config.label_col]]
        self.__output_generated_data(train, test, config.data_sanity_dir)

    # Default output_directory is base_data_dir + today's date
    def output_split(self, validation_proportion=0.3, output_directory=None):
        if output_directory is None:
            output_directory = '{}'.format(dt.date.today ())

        data_dir = config.get_data_dir(output_directory)
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

    def eval(self):
        filename = config.data_test_target_file(self._dir)
        if not os.path.isfile(filename):
            return None

        test_target = pd.read_csv(filename)
        test_target = test_target.sort_values(config.id_col).reset_index(drop=True)
        if not test_target[config.id_col].equals(self._df[config.id_col]):
            raise Exception('id cols are not consistent')
        else:
            return gini_normalized(test_target[config.label_col], self._df[config.label_col])

    def eval_output_and_register(self, filename):
        gini = self.eval()
        pred_dir = config.get_pred_dir(self._dir)
        log_file = config.pred_log_file(self._dir)
        filename = config.pred_filename(self._dir, filename, self._identifier)
        if not os.path.exists(log_file):
            os.system('mkdir -p {}'.format(pred_dir))
            f = open(log_file, 'w')
            f.write('data_dir,gini,user,time,identifier\n')
        else:
            f = open(log_file, 'ab')
        new_line = '{},{},{},{},{}'.format(
            self._dir,
            gini,
            getpass.getuser(),
            dt.datetime.now(),
            self._identifier
        )
        f.write(new_line)
        save_df_to_file(self._df, filename, overwrite=False)

def test_predcition():
    if not os.path.exists(config.get_data_dir(config.data_sanity_dir)):
        training_data = Training_data(config.data_raw_dir)
        training_data.output_small_data_for_sanity_check(config.data_sanity_dir)
    df = pd.read_csv(config.data_test_target_file(config.data_sanity_dir))
    prediction = Prediction(df, config.data_sanity_dir, identifier="sanity_check")
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
    training_data = Training_data(config.data_raw_dir)
    if args.mode == 'sanity':
        training_data.output_small_data_for_sanity_check()
    elif args.mode == 'split':
        training_data.output_split()
    else:
        raise Exception('unknown mode {}'.format(args.mode))
