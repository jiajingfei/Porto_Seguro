import os
import sys
import cPickle as pickle
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import pandas as pd
from abc import ABCMeta, abstractmethod
from data.data import Training_data as T
from data.data import Prediction as P
from feature.feature_extractor import FeatureExtractor as F
from utils import random_word
from config.base_config import (
    data_train_file,
    data_test_file
)

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
    def __init__(self, data_dir, param):
        self._identifier = random_word(10)
        self._dir = data_dir
        self._training_data = T(data_dir)
        self._testing_data = pd.read_csv(data_test_file(self._dir))
        self._param = param
        return None

    def save_param(self, filename):
        pickle.dump(self._param, open(filename, 'wb'))

    '''
    We may need df_test to determine whether to stop, I can imagine for some classifier,
    we don't need df_valie (for example, linear regression), the users should check
    df_valid=None by themselves for this case
    '''

    @abstractmethod
    def _train(self, df_train, df_valid):
        failwith Exception('Unimplemented in abstract class')

    @abstractmethod
    def _pred(self):
        failwith Exception('Unimplemented in abstract class')

    '''
    input data_dir should be a standard datadir
    '''
    def kfold_train_predict_eval(self, n_splits):
        data_dir = os.path.join(base_data_dir, data_sub_dir)
        training_data = Training_data(data_dir)
        testing_data = os.path.join(data_dir, test_file)

        def get_relevant_df(df):
            features = F(df).peek_df()
            relevant_cols = [id_col] + self._param['features']
            if label_col in features.columns:
                relevant_cols += [label_col]
            return features[relevant_cols]

        self._sum_pred = 0
        random_state = self._param.get_value('random_state')
        for i, (df_train, df_valid) in enumerate(training_data.kfold(n_splits, random_state)):
            # Doing this once before split is more efficient, but this is easier
            df_train = get_relevant_df(df_train)
            df_valid = get_relevant_df(df_valid)
            self._train(df_train, df_valid)
            pred = self._pred(df_test)
            self._sum_pred += pred
            p = P(pred, self._dir, self._identifier)
            p.eval_output_and_register('fold{}'.format(i))

        p = P(self._sum_pred, self._dir, self._identifier)
        p.eval_output_and_register('sum')
