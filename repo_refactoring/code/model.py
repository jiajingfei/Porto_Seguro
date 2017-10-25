import os
import getpass
import datetime as dt
import pandas as pd
from abc import ABCMeta, abstractmethod
from data_type import Training_data as T
from data_type import Prediction as P
from feature import FeatureExtractor as F
from utils import gini_normalized, unique_identifier, save_to_file
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
import config
import xgboost as xgb

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
    param: a dictionary. It must contain 'random_state' if in n_splits != None, its other
    values may depend on the model.
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

        param_file = config.model_filename(
            self._dir,
            filename='param',
            identifier=self._identifier
        )
        # k fold CV
        for i, (df_train, df_valid) in enumerate(training_data.kfold(n_splits, random_state)):
            df_features_train, df_features_valid, df_features_test = F().convert(
                df_train, df_valid, self._df_test, self._param.get('features')
            )
            self._train(df_features_train, df_features_valid)
            fold = 'fold{}'.format(i)
            if df_features_valid is not None:
                valid_pred = self._pred(df_features_valid)
                P.save(valid_pred, self._dir, '{}-valid-fold{}'.format(self._identifier, i))
            test_pred = self._pred(df_features_test)
            P.save(test_pred, self._dir, '{}-test-fold{}'.format(self._identifier, i))
            test_gini = P.eval(test_pred, self._dir)
            self._sum_pred += test_pred[config.label_col]
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
        P.save(sum_pred, self._dir, '{}-test-{}'.format(self._identifier, fold))
        test_gini = P.eval(sum_pred, self._dir)
        save_to_file(
            filename=config.model_log_file(self._dir),
            save_fn=to_save_fn(None, None, test_gini, fold),
            allow_existing=True
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


class XGBoost_CV(Model):
    @staticmethod
    def example_param():
        return {
            # Trainer related params
            'eta': 0.05,
            'max_depth': 4,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'seed': 123,
            'silent': True,
            # Control params
            'num_boost_round': 5000,
            # General parameters
            'n_splits': 5,
            'random_state': 456,
        }

    def _train(self, df_features_train, df_features_valid):
        assert (self._param['n_splits'] > 1)
        def gini_xgb(preds, dtrain):
            labels = dtrain.get_label()
            gini_score = gini_normalized(labels, preds)
            return [('gini', gini_score)]
        train_X = self._remove_id_and_label(df_features_train)#.values
        valid_X = self._remove_id_and_label(df_features_valid)#.values
        train_y = df_features_train[config.label_col]#.values
        valid_y = df_features_valid[config.label_col]#.values
        d_train = xgb.DMatrix(train_X, train_y)
        d_valid = xgb.DMatrix(valid_X, valid_y)
        self._model = xgb.train(
            self._param,
            d_train,
            num_boost_round = self._param.get('num_boost_round'),
            evals=[(d_train, 'train'), (d_valid, 'valid')],
            feval=gini_xgb,
            maximize=True,
            verbose_eval=50,
            early_stopping_rounds=100
        )

    def _pred(self, df_features):
        ids = df_features[config.id_col]
        d_test = xgb.DMatrix(self._remove_id_and_label(df_features))
        return pd.DataFrame(data = {
            config.id_col: ids,
            config.label_col: self._model.predict_proba(d_test)
        })


class Sklearn_gradientboosting(Model):
    @staticmethod
    def example_param():
        return {
            # Trainer related params
            'learning_rate': 0.05,
            'n_estimators': 500,
            'max_depth': 4,
            'subsample': 0.8,
            'n_splits': 5,
            'random_state': 456,
            'min_impurity_decrease': 0.001
        }

    def _train(self, df_features_train, df_features_valid):
        assert (self._param['n_splits'] > 1)
        param = {
            k:v for (k, v) in self._param.items() if k not in ['features', 'n_splits']
        }
        self._model = GradientBoostingClassifier(**param)
        train_X = self._remove_id_and_label(df_features_train)
        train_y = df_features_train[config.label_col]
        self._model.fit(train_X, train_y)

    def _pred(self, df_features):
        ids = df_features[config.id_col]
        features = self._remove_id_and_label(df_features)
        return pd.DataFrame(data={
            config.id_col: ids,
            config.label_col: self._model.predict_proba(features)[:, 1]
        })

class Catboost_CV(Model):
    @staticmethod
    def example_param():
        return {
            # Trainer related params
            'learning_rate': 0.05,
            'iterations': 1000,
            'depth': 6,
            'l2_leaf_reg': 6,
            'loss_function': 'Logloss',
            'verbose': True,
            'n_splits': 5,
            'random_state': 456,
            'optimize_rounds': True
        }

    def _train(self, df_features_train, df_features_valid):
        assert (self._param['n_splits'] > 1)
        param = {
            k:v for (k, v) in self._param.items()
            if k not in ['features', 'n_splits', 'random_state', 'optimize_rounds']
        }
        model = CatBoostClassifier(**param)
        train_X = self._remove_id_and_label(df_features_train)
        train_y = df_features_train[config.label_col]
        valid_X = self._remove_id_and_label(df_features_valid)
        valid_y = df_features_valid[config.label_col]
        self._fit_model = model.fit(
            train_X,
            train_y,
            eval_set=[valid_X, valid_y],
            use_best_model=self._param.get('optimize_rounds')
        )
        print( "  N trees = ", model.tree_count_ )

    def _pred(self, df_features):
        ids = df_features[config.id_col]
        features = self._remove_id_and_label(df_features)
        pred_labels = self._fit_model.predict_proba(features)[:, 1]
        return pd.DataFrame(data={
            config.id_col: ids,
            config.label_col: pred_labels
        })

