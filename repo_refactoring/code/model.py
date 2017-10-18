import os
import getpass
import datetime as dt
import pandas as pd
from abc import ABCMeta, abstractmethod
from data_type import Training_data as T
from data_type import Prediction as P
from feature import FeatureExtractor as F
from utils import random_word, save_to_file
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
        self._identifier = random_word(10) if identifier is None else identifier
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

    @abstractmethod
    # Train, this function may modify the inputs
    def _train(self, features_train, features_valid):
        raise Exception('Unimplemented in abstract class')

    @abstractmethod
    def _pred(self, features_test):
        raise Exception('Unimplemented in abstract class')

    '''
    input data_dir should be a standard datadir
    '''
    def kfold_train_predict_eval(self, n_splits):
        '''
        Main worker function. 
        Here we will do the k-fold CV on the training data
        and save the results and the parameters to respective log files
        
        Input
        ---------
        n_splits: <int>
            number of folds
            
        Return
        ---------
        Nothing, save / modify 2 log files.
        '''
        training_data = T(self._dir)
        self._sum_pred = 0
        if n_splits is None:
            random_state = None
        else:
            random_state = self._param['random_state']
        time  = dt.datetime.now()
        for i, (df_train, df_valid) in enumerate(training_data.kfold(n_splits, random_state)):
            # Doing this once before split is more efficient, but this is easier
            feature = F()
            # In the current code, we do this over and over again, which is not efficient,
            # but I suspect the bottleneck is still model training. And if later we decide to
            # freeze the feature set we want to use, we should do this feature conversion on
            # the rawdata.
            features_train, features_valid, features_test = feature.convert(
                df_train, df_valid, self._df_test, self._param['features']
            )
            self._train(features_train, features_valid)
            pred = self._pred(features_test)
            self._sum_pred += pred[config.label_col]
            p = P(pred, self._dir, self._identifier)
            p.eval_output_and_register('fold{}'.format(i), time)

        sum_pred = pd.DataFrame(data={
            config.id_col: self._df_test[config.id_col],
            config.label_col: self._sum_pred
        })

        p = P(sum_pred, self._dir, self._identifier)
        p.eval_output_and_register('sum', time)

        def write_log(log_file):
            if os.path.isfile(log_file):
                f = open(log_file, 'ab')
            else:
                f = open(log_file, 'w')
                f.write('data_dir,user,time,identifier,model_name\n')
            new_line = '{},{},{},{},{}\n'.format(
                self._dir,
                getpass.getuser(),
                time,
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

    def _pred(self, df_test):
        ids = df_test[config.id_col]
        return pd.DataFrame(data={
            config.id_col: ids,
            config.label_col: ids%5
        })
    
class RandomForest(Model):
    
    def _train(self, df_train, df_valid):
        # train a simple random forest (RF)
        # unlike xgboost, 
        # for RF, we don't need a validation set for training
        # prepare the data
        y = df_train[config.label_col]
        del df_train[config.label_col]
        X = df_train
        # prepare the model
        model_param = { 
            k : v for (k, v) in self._param.item()
            if k not in ['features', 'random_state'] 
        }
        self.__clf = RandomForestClassifier(**model_param)
        # fit!
        self.__clf.fit(X, y)
        return None

    def _pred(self, df_test):
        ids = df_test[config.id_col]
        y_proba = self.__clf.predict_proba(df_test)
        return pd.DataFrame(data={
            config.id_col: ids,
            config.label_col: y_proba
        })

    def kfold_train_predict_eval(self, n_splits):
        if n_splits is not None:
            raise Exception(
                'No cross-validation has been implemented in {}, n_splits must be None'.format(self.__class__.__name__)
            )
        super(RandomForest, self).kfold_train_predict_eval(n_splits)
