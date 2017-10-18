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

    def __output_generated_data(self, train, test, output_directory):
        if not os.path.exists(config.get_data_dir(output_directory)):
            os.system('mkdir -p {}'.format(config.get_data_dir(output_directory)))
        test_target = test[[config.id_col, config.label_col]]
        del test[config.label_col]
        train_file = config.data_train_file(output_directory)
        test_file = config.data_test_file(output_directory)
        train.to_csv(train_file, index=False)
        test.to_csv(test_file, index=False)
        train[config.id_col].to_csv(train_file+'.id', index=False)
        test[config.id_col].to_csv(test_file+'.id', index=False)
        test_target.to_csv(config.data_test_target_file(output_directory), index=False)
        with open(config.data_readme_file(output_directory), 'w') as f:
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

    def eval_output_and_register(self, filename, time):

        gini = self.eval()

        def write_log(log_file):
            if os.path.isfile(log_file):
                f = open(log_file, 'ab')
            else:
                f = open(log_file, 'w')
                f.write('data_dir,gini,user,time,identifier\n')
            new_line = '{},{},{},{},{}\n'.format(
                self._dir,
                gini,
                getpass.getuser(),
                time,
                self._identifier
            )
            f.write(new_line)
            f.close()

        save_to_file(
            config.pred_log_file(self._dir),
            save_fn=write_log,
            allow_existing=True
        )

        save_df_to_file(
            self._df,
            config.pred_filename(self._dir, filename, self._identifier),
            overwrite=False
        )

def test_predcition():
    if not os.path.exists(config.get_data_dir(config.data_sanity_dir)):
        training_data = Training_data(config.data_raw_dir)
        training_data.output_small_data_for_sanity_check(config.data_sanity_dir)
    df = pd.read_csv(config.data_test_target_file(config.data_sanity_dir))
    prediction = Prediction(df, config.data_sanity_dir, identifier="sanity_check")
    normalized_gini = prediction.eval()
    assert (np.isclose(normalized_gini, 1))

def generate_data_from_ids(data_folder='2017-10-15'):
    '''
    This code is to read the data, from the id
    
    Input
    ---------
    data_type: <str>, default: '2017-10-15'
        what kind of data we try to generate, 
        in {'sanity_data', '2017-10-15', ...}
    
    Return
    ---------
    Nothing, but create 3 files under the same folder.
    If the files already exist, we will overwrite
    '''
    
    # get indices
    if data_folder == 'raw_data':
        print('We do not want to modify the raw data. Will exit.')
        return
        
    ids_train_file = config.data_train_file(data_folder) + '.id'
    ids_test_file = config.data_test_file(data_folder) + '.id'
    # update the data_folder
    data_folder = config.get_data_dir(data_folder)
    
    try:
        with open(ids_train_file, 'r') as f:
            train_ids = f.readlines()
        train_ids = list(map(int, train_ids))
        with open(ids_test_file, 'r') as f:
            test_ids = f.readlines()
        test_ids = list(map(int, test_ids))
    except:  # no such folder
        print('The folder {} does not exist.'.format(data_folder))
        return
    
    # get the full, raw data
    raw_data_file = config.data_train_file(config.data_raw_dir)
    raw_df = pd.read_csv(raw_data_file)
    
    train_df = raw_df[raw_df[config.id_col].isin(train_ids)]
    test_df = raw_df[raw_df[config.id_col].isin(test_ids)]
    # create the target file
    target_df = test_df[[config.id_col, config.label_col]]
    del test_df[config.label_col]
    
    # save files
    train_df.to_csv(os.path.join(data_folder, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(data_folder, 'test.csv'), index=False)
    target_df.to_csv(os.path.join(data_folder, '.test_target.csv'), index=False)
    
    return 

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
    
    # create the 'real' data 
    generate_data_from_ids(data_folder='sanity_data')
