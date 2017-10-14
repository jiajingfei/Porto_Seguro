import os
import sys
import pandas as pd
import datetime as dt
import argparse
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils import gini_normalized
from configs.base_config import (
    id_col,
    label_col,
    base_data_dir,
    train_file,
    test_file,
    test_target_file,
    raw_data_dir,
    sanity_data_dir
)

class Training_data(object):
    def __init__(self, directory):
        self._train = pd.read_csv(os.path.join(directory, train_file))
        return None

    def __output_generated_data(self, train, test, output_directory):
        if not os.path.exists(output_directory):
            os.system('mkdir -p {}'.format(output_directory))
        test_target = test[[id_col, label_col]]
        del test[label_col]
        train.to_csv(os.path.join(output_directory, train_file), index=False)
        test.to_csv(os.path.join(output_directory, test_file), index=False)
        test_target.to_csv(os.path.join(output_directory, test_target_file), index=False)
        with open(os.path.join(output_directory, 'readme.txt'), 'w') as f:
            readme_lines = [
                'this directory is for generated training data and validation data\n',
                'including:\n',
                train_file + '\n',
                test_file + '\n',
                test_target_file + '\n'
            ]
            f.writelines(readme_lines)

    def output_small_data_for_sanity_check(self, output_directory, nrows=500):
        if os.path.exists(output_directory):
            output_directory = os.path.abspath(output_directory)
            print ('WARNING: {} already exists, will overwrite it'.format(output_directory))

        df = shuffle(self._train)
        train = df[:nrows].copy()
        test = df[nrows:2*nrows].copy()
        test_target = test[[label_col]]
        self.__output_generated_data(train, test, output_directory)

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
            self.__output_generated_data(train, test, output_directory)

    def kfold(self, n_splits, random_state=3223):
        results = []
        sss = StratifiedKFold(n_splits, random_state=random_state)
        for (train_ind, valid_ind) in sss.split(df, df.target):
            results.append((df.iloc[train_ind], df.iloc[valid_idn]))
        return results


class Prediction(object):
    def __init__(self, df):
        self._df = df.sort_values(id_col).reset_index(drop=True)

    def eval(self, data_dir):
        filename = os.path.join(data_dir, test_target_file)
        if not os.path.isfile(filename):
            raise Exception('{} does not exist, is this raw file'.format(filename))
        test_target = pd.read_csv(filename)
        test_target = test_target.sort_values(id_col).reset_index(drop=True)
        if not test_target[id_col].equals(self._df[id_col]):
            raise Exception('id cols are not consistent')
        else:
            return gini_normalized(test_target[label_col], self._df[label_col])

def test_predcition():
    if not os.path.exists(sanity_data_dir):
        training_data = Training_data(raw_data_dir)
        training_data.output_small_data_for_sanity_check(sanity_data_dir)
    df = pd.read_csv(os.path.join(sanity_data_dir, test_target_file))
    print df.head()
    prediction = Prediction(df)
    normalized_gini = prediction.eval(sanity_data_dir)
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
    training_data = Training_data(raw_data_dir)
    if args.mode == 'sanity':
        training_data.output_small_data_for_sanity_check(output_directory=sanity_data_dir)
    elif args.mode == 'split':
        training_data.output_split()
    else:
        raise Exception('unknown mode {}'.format(args.mode))
