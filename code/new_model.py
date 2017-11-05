import os
import gc
import getpass
import datetime as dt
import pandas as pd
from abc import ABCMeta, abstractmethod
from data_type import Training_data as T
from data_type import Prediction as P
from new_feature import Feature as F
from utils import gini_normalized, unique_identifier, save_to_file, remove_id_and_label
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
import config
import xgboost as xgb
import lightgbm as lgb


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

    # Default is using raw features
    def __init__(self, feature_dir, param, identifier=None, actions_dict={}):
        self._identifier = unique_identifier() if identifier is None else identifier
        self._actions_dict = actions_dict
        self._feature_dir = feature_dir
        self._param = param
        return None

    def _save_param(self, filename):
        all_param = {
            'actions': self._actions_dict,
            'param': self._param,
            'feature_dir': self._feature_dir
        }
        pickle.dump(self._param, open(filename, 'wb'))

    @abstractmethod
    def _train(self, df_train, df_valid):
        raise Exception('Unimplemented in abstract class')

    @abstractmethod
    def _pred(self, df):
        raise Exception('Unimplemented in abstract class')

    def train_predict_eval_and_log(self):
        time  = dt.datetime.now()
        def to_save_fn(train_gini, valid_gini, test_gini, fold):
            header = ','.join([
                'feature_dir',
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
                self._feature_dir,
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

        def get_gini(df):
            if df is None:
                return None
            else:
                return gini_normalized(
                    df[config.label_col],
                    self._pred(df)[config.label_col]
                )

        param_file = config.model_filename(
            self._feature_dir,
            filename='param',
            identifier=self._identifier
        )

        fold_num = config.get_num_folds(self._feature_dir)
        def prepare_data(df_file):
            df = pd.read_pickle(df_file)
            orig_features = config.get_orig_features(df)
            cols = [config.id_col]
            if config.label_col in df.columns:
                cols += [config.label_col]
            for f in orig_features:
                actions = self._actions_dict.get(f, [])
                f = F(f)
                cols += f.get_features(df, actions)
            df = df[cols].copy()
            gc.collect()
            return df

        sum_pred = 0
        for i in range(fold_num):
            df_train = prepare_data(config.get_feature_file(self._feature_dir, i, 'train'))
            df_valid = prepare_data(config.get_feature_file(self._feature_dir, i, 'valid'))
            df_test = prepare_data(config.get_feature_file(self._feature_dir, i, 'test'))
            self._train(df_train, df_valid)
            fold = 'fold{}'.format(i)
            valid_pred = self._pred(df_valid)
            P.save(valid_pred, self._feature_dir, '{}-valid-fold{}'.format(self._identifier, i))
            test_pred = self._pred(df_test)
            P.save(test_pred, self._feature_dir, '{}-test-fold{}'.format(self._identifier, i))

            target_file = config.get_feature_file(self._feature_dir, None, 'test_label')
            test_gini = P.eval(test_pred, target_file)
            sum_pred += test_pred[config.label_col]

            train_gini = get_gini(df_train)
            valid_gini = get_gini(df_valid)
            save_to_file(
                filename=config.model_log_file(self._feature_dir),
                save_fn=to_save_fn(train_gini, valid_gini, test_gini, fold),
                allow_existing=True
            )

        sum_pred = pd.DataFrame(data={
            config.id_col: df_test[config.id_col],
            config.label_col: sum_pred
        })
        P.save(sum_pred, self._feature_dir, '{}-test-{}'.format(self._identifier, 'sum'))
        target_file = config.get_feature_file(self._feature_dir, None, 'test_label')
        test_gini = P.eval(sum_pred, target_file)
        save_to_file(
            filename=config.model_log_file(self._feature_dir),
            save_fn=to_save_fn(None, None, test_gini, fold),
            allow_existing=True
        )
        save_to_file(
            filename=param_file,
            save_fn=lambda filename: self._save_param(filename),
            allow_existing=False
        )

class Toy_model(Model):

    def _train(self, df_train, df_valid):
        return None

    def _pred(self, df):
        ids = df[config.id_col]
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
        X = remove_id_and_label(df_train)
        # prepare the model
        model_param = {
            k : v for (k, v) in self._param.items()
            if k not in ['excluded_features', 'random_state', 'n_splits']
        }
        self.__clf = RandomForestClassifier(**model_param)
        # fit!
        self.__clf.fit(X, y)
        return None

    def _pred(self, df):
        ids = df[config.id_col]
        features = remove_id_and_label(df)
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

    def _train(self, df_train, df_valid):
        assert (self._param['n_splits'] > 1)
        def gini_xgb(preds, dtrain):
            labels = dtrain.get_label()
            gini_score = gini_normalized(labels, preds)
            return [('gini', gini_score)]
        train_X = remove_id_and_label(df_train)#.values
        valid_X = remove_id_and_label(df_valid)#.values
        train_y = df_train[config.label_col]#.values
        valid_y = df_valid[config.label_col]#.values
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

    def _pred(self, df):
        ids = df[config.id_col]
        d_test = xgb.DMatrix(remove_id_and_label(df))
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

    def _train(self, df_train, df_valid):
        assert (self._param['n_splits'] > 1)
        param = {
            k:v for (k, v) in self._param.items() if k not in ['features', 'n_splits']
        }
        self._model = GradientBoostingClassifier(**param)
        train_X = remove_id_and_label(df_train)
        train_y = df_train[config.label_col]
        self._model.fit(train_X, train_y)

    def _pred(self, df):
        ids = df[config.id_col]
        features = remove_id_and_label(df)
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

    def _train(self, df_train, df_valid):
        assert (self._param['n_splits'] > 1)
        param = {
            k:v for (k, v) in self._param.items()
            if k not in ['features', 'n_splits', 'random_state', 'optimize_rounds']
        }
        model = CatBoostClassifier(**param)
        train_X = remove_id_and_label(df_train)
        train_y = df_train[config.label_col]
        valid_X = remove_id_and_label(df_valid)
        valid_y = df_valid[config.label_col]
        self._fit_model = model.fit(
            train_X,
            train_y,
            eval_set=[valid_X, valid_y],
            use_best_model=self._param.get('optimize_rounds')
        )
        print( "  N trees = ", model.tree_count_ )

    def _pred(self, df):
        ids = df[config.id_col]
        features = remove_id_and_label(df)
        return pd.DataFrame(data={
            config.id_col: ids,
            config.label_col: self._fit_model.predict_proba(features)[:, 1]
        })

class Lightgbm_CV(Model):
    def example_param():
        return {
            'metric': 'auc',
            'learning_rate' : 0.01,
            'max_depth': 10,
            'max_bin': 10,
            'objective': 'binary',
            'feature_fraction': 0.8,
            'bagging_fraction': 0.9,
            'bagging_freq': 10,
            'min_data': 500,
            'n_splits': 5,
            'random_state': 1025,
        }

    def _train(self, df_train, df_valid):
        assert (self._param['n_splits'] > 1)

        def gini_lgb(preds, dtrain):
            y = list(dtrain.get_label())
            score = gini(y, preds) / gini(y, y)
            return 'gini', score, True

        param = {
            k:v for (k, v) in self._param.items() if
            k not in ['features', 'n_splits', 'random_state', 'excluded_features']
        }
        train_X = remove_id_and_label(df_train)
        valid_X = remove_id_and_label(df_valid)
        train_y = df_train[config.label_col]
        valid_y = df_valid[config.label_col]

        self._model = lgb.LGBMClassifier(**param)
        self._model.fit(
            train_X,
            train_y,
            eval_set=[(valid_X, valid_y)],
            early_stopping_rounds=100,
            eval_metric='auc',
            verbose=True)

    def _pred(self, df):
        ids = df[config.id_col]
        d_test = remove_id_and_label(df)
        return pd.DataFrame(data={
            config.id_col: ids,
            config.label_col: self._model.predict_proba(d_test)[:,1]
            })
