import os
import sys
import cPickle as pickle
import getpass
import datetime as dt
import pandas as pd
from abc import ABCMeta, abstractmethod
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from data_type import Training_data as T
from data_type import Prediction as P
from feature.feature_extraction import FeatureExtractor as F
from utils import random_word, save_to_file
import config

'''
This module is designed to generate deterministic results and keep tracking on the prediction's
metrics.
'''
class Model():

    __metaclass__ = ABCMeta

    '''
    data_dir: the directory name of the data, data is not allowed to passed in by dataframe
    param: a dictionary. It must contain 'features' and 'random_state' if in n_splits != None,
    its other values may depend on the model
    '''
    def __init__(self, data_dir, param, identifier=None):
        self._identifier = random_word(10) if identifier is None else identifier
        self._dir = data_dir
        self._training_data = T(data_dir)
        self._testing_data = pd.read_csv(config.data_test_file(self._dir))
        self._param = param
        return None

    def _save_param(self, filename):
        pickle.dump(self._param, open(filename, 'wb'))

    '''
    We may need df_test to determine whether to stop, I can imagine for some classifier,
    we don't need df_valie (for example, linear regression), the users should check
    df_valid=None by themselves for this case
    '''

    @abstractmethod
    def _train(self, df_train, df_valid):
        raise Exception('Unimplemented in abstract class')

    @abstractmethod
    def _pred(self):
        raise Exception('Unimplemented in abstract class')

    '''
    input data_dir should be a standard datadir
    '''
    def kfold_train_predict_eval(self, n_splits):
        training_data = T(self._dir)
        testing_data = config.data_test_file(self._dir)

        def get_relevant_df(df):
            features = F(df).peek_df()
            relevant_cols = [config.id_col] + self._param['features']
            if config.label_col in features.columns:
                relevant_cols += [config.label_col]
            return features[relevant_cols]

        self._sum_pred = 0
        if n_splits is None:
            random_state = None
        else:
            random_state = self._param['random_state']
        for i, (df_train, df_valid) in enumerate(training_data.kfold(n_splits, random_state)):
            # Doing this once before split is more efficient, but this is easier
            df_train = get_relevant_df(df_train)
            df_valid = get_relevant_df(df_valid)
            self._train(df_train, df_valid)
            pred = self._pred()
            self._sum_pred += pred[config.label_col]
            p = P(pred, self._dir, self._identifier)
            p.eval_output_and_register('fold{}'.format(i))

        sum_pred = pd.DataFrame(data={
            config.id_col: self._testing_data[config.id_col],
            config.label_col: self._sum_pred
        })

        p = P(sum_pred, self._dir, self._identifier)
        p.eval_output_and_register('sum')

        def write_log(log_file):
            if os.path.isfile(log_file):
                f = open(log_file, 'w')
                f.write('data_dir,user,time,identifier,model_name\n')
            else:
                f = open(log_file, 'ab')
            new_line = '{},{},{},{},{}'.format(
                self._dir,
                getpass.getuser(),
                dt.datetime.now(),
                self._identifier,
                self.__class__.__name__
            )
            f.write(new_line)
            f.close()

        save_to_file(
            filename=config.model_log_file(self._dir),
            save_fn=write_log,
            allow_existing=True
        )

        save_to_file(
            filename=config.model_filename(self._dir, filename='param', identifier=self._identifier),
            save_fn=lambda filename: self._save_param(filename),
            allow_existing=False
        )

class Toy_model(Model):
    def _train(self, df_train, df_valid):
        return None

    def _pred(self):
        ids = self._testing_data[config.id_col]
        return pd.DataFrame(data={
            config.id_col: ids,
            config.label_col: ids%5
        })
