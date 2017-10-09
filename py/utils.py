import os
import numpy as np
import pandas as pd
import datetime as dt
import getpass
import xgboost
import cPickle as pickle
from abc import ABCMeta, abstractmethod

this_path = os.path.dirname(os.path.realpath(__file__))

''' Metrics related '''

def gini_normalized(a, p):
    def gini(actual, pred, cmpcol=0, sortcol=1):
        assert( len(actual) == len(pred) )
        all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
        all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
        totalLosses = all[:,0].sum()
        giniSum = all[:,0].cumsum().sum() / totalLosses
        giniSum -= (len(actual) + 1) / 2.
        return giniSum / len(actual)
    return gini(a, p) / gini(a, a)

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return [('gini', gini_score)]


''' Feature related '''


def get_features(df):
    cols = list(df.columns)
    if 'id' in cols: cols.remove('id')
    if 'target' in cols: cols.remove('target')
    return df[cols]

def get_labels(df):
    return df['target']


''' General class type '''

class C():

    __metaclass__ = ABCMeta
    def __init__(self, param):
        self._param = param
    def load_param(self, filename):
        self._param = pickle.load(open(filename, 'rb'))
    def save_param(self, filename):
        pickle.dump(self._param, open(filename, 'wb'))
    def get_value(self, key):
        return self._param.get('{}:{}'.format(self.__class__.__name__, key))


''' Model for private leaderboard '''
class Model(C):

    _private_lb = os.path.join(this_path, '../data/private_leaderboard/lb.csv')
    _params_dir = os.path.join(this_path, '../data/private_leaderboard/params')
    _data_dir = os.path.join(this_path, '../data')

    __metaclass__ = ABCMeta

    @abstractmethod
    # This function should return the predict score for df_test
    def train_and_predict(self, df_train, df_test):
        return

    def train_and_get_score(self, data_dir=None):
        if data_dir is None:
            dirs = [s for s in os.listdir(self._data_dir) if s.startswith('2017')] 
            dirs.sort()
            data_dir = os.path.join(self._data_dir, dirs[-1])
        df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
        test_target = pd.read_csv(os.path.join(data_dir, '.test_target.csv'))
        test_pred = self.train_and_predict(df_train, df_test)
        self._this_data_dir = data_dir
        self._test_pred = test_pred
        return gini_normalized(test_target, test_pred)

    def train_and_write_on_lb(self, data_dir=None, description=None):
        time = dt.datetime.now()
        if hasattr(self, _name):
            model_name = self._name
        else:
            model_name = self.__class__.__name__
        score = self.train_and_get_score(data_dir)
        param_file = os.path.join(self._params_dir, 'param_{}'.format(time))
        pred_file = os.path.join(self._params_dir, 'pred_{}.csv'.format(time))
        self._save_param(param_file)
        self._test_pred.to_csv(pred_file, index=False)
        data_set = self._this_data_dir
        user = getpass.getuser()
        new_line = '{},{},{},{},{},{},{}'.format(
            user,
            time,
            data_set,
            model_name,
            param_file,
            score,
            description
        )
        f = open(self._private_lb,'ab')
        f.write(new_line)
        f.close()


''' gradient boosting trees
num_boost_round
early_stopping_rounds
and all other xgb_params
'''
class Xgb_trees(C):
    def train(self, df_train, df_valid):
        train_x = get_features(df_train).values
        valid_x = get_features(df_valid).values
        train_y = get_labels(df_train).values
        valid_y = get_labels(df_valid).values
        d_train = xgb.DMatrix(train_x, train_y)
        d_valid = xgb.DMatrix(valid_x, valid_y)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        xgb_params = {
            'objective': self.get_value('objective'),
            'eta': self.get_value('eta'),
            'silent': self.get_value('silent'),
            'max_depth': self.get_value('max_depth'),
            'subsample': self.get_value('subsample'),
            'colsample_bytree': self.get_value('colsample_bytree'),
            'colsample_bylevel': self.get_value('colsample_bylevel')
        }
        self._model = xgb.train(
            xgb_params,
            d_train,
            self.get_value('num_boost_round'),
            watchlist,
            feval=gini_xgb,
            maximize=True,
            verbose_eval=50,
            early_stopping_rounds=self.get_value('early_stopping_rounds')
        )

    def pred(self, df):
        df = get_features(df)
        preds = self._model.predict(xgb.DMatrix(df.values))
        return pd.Dataframe(data={'target': preds})

'''
params:
n_splits
is_stratified
random_state
'''
class Kfold(C):
    def split_data(self, df):
        inds = []
        n_splits=self.get_value('n_splits')
        random_state=self.get_value('random_state')
        if self.get_value('is_stratified'):
            sss = StratifiedKFold(n_splits, random_state=random_state)
            for ind in sss.split(df, df.target):
                inds.append(ind)
        else:
            kf = KFold(n_splits, random_state=random_state)
            for ind in kf.split(df):
                inds.append(ind)
        return [(df.iloc[train_ind], df.iloc[test_ind]) for (train_ind, test_ind) in inds]


'''
params:
features_to_drop
features_to_reorder
features_to_revert
'''
class Feature_transformation(C):

    def drop_features(self, df, features=None):
        if features is None:
            features = self.get_value('features_to_drop')
        if features is None:
            return df
        else:
            df = df.copy()
            for f in features:
                if f in df.columns:
                    del df[f]
            return df

    # only reorder given features if features is not None,
    # otherwise reorder all the binary and categorical features
    def reorder_feature(self, df, df_test, features=None):
        if features is None:
            features = self.get_value('features_to_reorder')
        if features is None:
            return df, df_test
        elif features == 'all':
            features = []
            for c in get_features(df).columns:
                if c.endswith(('bin', 'cat')):
                    features.append(c)

        transforms = {}
        for c in features:
            tmp = df[[c, 'target']].groupby(c).mean()
            tmp = tmp.sort_values('target').reset_index()
            tmp.loc[:, 'new_'+c] = tmp.index
            del tmp['target']
            transforms[c] = tmp
        def apply(df):
            df = df.copy()
            for c, tmp in transforms.items():
                if c.endswith(('bin', 'cat')):
                    df = df.merge(tmp[[c, 'new_'+c]], on=c)
                    df.loc[:, c] = df['new_'+c]
                    del df['new_'+c]
                    print 'feature {} has been transformed'.format(c)
            return df
        return apply(df), apply(df_test)

    def revert_one_hot(self, df, features=None):
        if features is None:
            features = self.get_value('features_to_revert')
        if features is None:
            return df
        else:
            df = df.copy()
            for new_feas, feas in features:
                to_revert = df[feas]
                new_feas = '{}_cat'.format(new_feas)
                to_revert.loc[:, new_feas] = -1
                for i, f in enumerate(feas):
                    if not (to_revert[to_revert[f]==1][new_feas] == -1).all():
                        raise Exception('Error in revert one hot encoding, check {}'.format(feas))
                    to_revert.loc[:, new_feas] = (i+1) * (to_revert[f]==1) + to_revert[new_feas]
                df.loc[:, new_feas] = to_revert[new_feas]
                for f in feas:
                    del df[f]
            return df

    def transform(self, df_train, df_test):
        df_train = self.drop_features(df_train)
        df_test = self.drop_features(df_test)
        df_train = self.revert_one_hot(df_train)
        df_test = self.revert_one_hot(df_test)
        return self.reorder_feature(df_train, df_test)

def combine_params(params_list):
    params = {}
    for class_name, params in params_list:
        for key, data in params.items():
            params['{}:{}'.format(class_name, key)] = data
    return params
