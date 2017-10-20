import os
import getpass
import datetime as dt
import pandas as pd
from abc import ABCMeta, abstractmethod
from data_type import Training_data as T
from data_type import Prediction as P
from feature import FeatureExtractor as F
from utils import gini_normalized, unique_identifier, save_to_file
from sklearn.ensemble import RandomForestClassifier
import config

try:  # for python 2.x
    import cPickle as pickle
except:  # for python 3.x
    import pickle


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
        self._identifier = unique_identifier() if identifier is None else identifier
        self._dir = data_dir
        self._training_data = T(data_dir)
        self._df_test = pd.read_csv(config.data_test_file(self._dir))
        self._param = param
        return None

    def _save_param(self, filename):
        pickle.dump(self._param, open(filename, 'wb'))

    '''
    We may need df_test to determine whether to stop, I can imagine for some classifier,
    we don't need df_valie (for example, linear regression), the users should check
    df_valid=None by themselves for this case
    '''

    # This will not modify the original dataframe
    def _remove_id_and_label(self, df):
        cols = [c for c in df.columns if c not in [config.id_col, config.label_col]]
        return df[cols]

    @abstractmethod
    def _train(self, df_features_train, df_features_valid):
        raise Exception('Unimplemented in abstract class')

    @abstractmethod
    # df_features is features dataframe, it may contain or not contain label col
    def _pred(self, df_features):
        raise Exception('Unimplemented in abstract class')

    '''
    input data_dir should be a standard datadir
    '''
    def train_predict_eval_and_log(self):
        '''
        Main worker function. 
        Here we will do the k-fold CV on the training data
        and save the results and the parameters to respective log files

        The number of folds, n_splits, should be defined in self._params

        Input
        ---------
        Nothing

        Return
        ---------
        Nothing, save / modify 2 log files.
        '''

        time  = dt.datetime.now()
        n_splits = self._param.get('n_splits')
        if n_splits is None:
            random_state = None
        else:
            random_state = self._param['random_state']

        def to_save_fn(train_gini, valid_gini, test_gini, fold):
            header = ','.join([
                'data_dir',
                'user',
                'time',
                'identifier',
                'model_name',
                'train_gini',
                'valid_gini',
                'test_gini',
                'fold_str'
            ])
            new_row = [
                self._dir,
                getpass.getuser(),
                time,
                self._identifier,
                self.__class__.__name__,
                train_gini,
                valid_gini,
                test_gini,
                fold
            ]
            def write_log(log_file):
                if os.path.exists(log_file):
                    f = open(log_file, 'a')
                else:
                    f = open(log_file, 'w')
                    f.write(header + '\n')
                f.write(','.join([str(x) for x in new_row]) + '\n')
                f.close()
            return write_log

        # df_features here must be None or a features dataframe with label col
        def get_gini(df_features):
            if df_features is None:
                return None
            else:
                return gini_normalized(
                    df_features[config.label_col],
                    self._pred(df_features)[config.label_col]
                )

        training_data = T(self._dir)
        self._sum_pred = 0

        # k fold CV
        for i, (df_train, df_valid) in enumerate(training_data.kfold(n_splits, random_state)):
            # Doing this once before split is more efficient, but this is easier
            f = F()
            # In the current code, we do this over and over again, which is not efficient,
            # but I suspect the bottleneck is still model training. And if later we decide to
            # freeze the feature set we want to use, we should do this feature conversion on
            # the rawdata.
            df_features_train, df_features_valid, df_features_test = f.convert(
                df_train, df_valid, self._df_test, self._param['features']
            )
            self._train(df_features_train, df_features_valid)
            fold = 'fold{}'.format(i)
            pred = self._pred(df_features_test)
            p = P(pred, self._dir, self._identifier)
            test_gini = p.eval_and_save(fold)
            self._sum_pred += pred[config.label_col]
            
            if n_splits is not None:
                train_gini = get_gini(df_features_train)
                valid_gini = get_gini(df_features_valid)
                save_to_file(
                    filename=config.model_log_file(self._dir),
                    save_fn=to_save_fn(train_gini, valid_gini, test_gini, fold),
                    allow_existing=True
                )

        sum_pred = pd.DataFrame(data={
            config.id_col: self._df_test[config.id_col],
            config.label_col: self._sum_pred
        })
        if n_splits is None:
            fold = 'all'
        else:
            fold = 'sum'
        p = P(sum_pred, self._dir, self._identifier)
        test_gini = p.eval_and_save(fold)
        # save the result log
        save_to_file(
            filename=config.model_log_file(self._dir),
            save_fn=to_save_fn(None, None, test_gini, fold),
            allow_existing=True
        )
        param_file = config.model_filename(
            self._dir,
            filename='param',
            identifier=self._identifier
        )
        save_to_file(
            filename=param_file,
            save_fn=lambda filename: self._save_param(filename),
            allow_existing=False
        )

class Toy_model(Model):

    def _train(self, df_features_train, df_features_valid):
        return None

    def _pred(self, df_features):
        ids = df_features[config.id_col]
        return pd.DataFrame(data={
            config.id_col: ids,
            config.label_col: ids%5
        })

class RandomForest(Model):

    def _train(self, df_features_train, df_features_valid):
        # train a simple random forest (RF)
        # unlike xgboost, 
        # for RF, we don't need a validation set for training
        # prepare the data
        y = df_features_train[config.label_col]
        X = self._remove_id_and_label(df_features_train)
        # prepare the model
        model_param = {
            k : v for (k, v) in self._param.items()
            if k not in ['features', 'random_state', 'n_splits']
        }
        self.__clf = RandomForestClassifier(**model_param)
        # fit!
        self.__clf.fit(X, y)
        return None

    def _pred(self, df_features):
        ids = df_features[config.id_col]
        features = self._remove_id_and_label(df_features)
        y_proba = self.__clf.predict_proba(features)[:,1]  # the proba of being 1
        return pd.DataFrame(data={
            config.id_col: ids,
            config.label_col: y_proba
        })
