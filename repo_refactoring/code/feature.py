import os
import pandas as pd
import numpy as np
import config
import gc
from data_type import Training_data as T
from utils import save_to_file

class Feature_type(object):
    binary = 'bin'
    categorical = 'cat'
    continuous = 'continuous'

class Action(object):
    dropping = 'dp'
    one_hot = 'oh'
    above_mean = 'mean'
    above_median = 'med'
    reorder = 'ro'
    reorder_above_mean = 'ro_mean'
    reorder_above_median = 'ro_med'

class Feature(object):

    def __init__(self, name):
        if not name.startswith('ps_'):
            raise Exception('{} is not a valid feature'.format(name))
        self._name = name
        suffix = name.split('_')[-1]
        if suffix == 'bin':
            self._type = Feature_type.binary
        elif suffix == 'cat':
            self._type = Feature_type.categorical
        else:
            self._type = Feature_type.continuous

    def __add_features_if_not_exist(self, df, col, feature_extraction):
        self._features.append(col)
        if col not in df.columns:
            df.loc[:, col] = feature_extraction(df)

    # This won't modify the input dataframe
    def load_df_train(self, df_train):
        self._df_train = df_train[[config.label_col, self._name]]
        self._label_group_mean = self._df_train.groupby(self._name, as_index=False).mean()
        self._unique_values = self._label_group_mean[self._name].values
        self._num_unique_values = len(self._unique_values)
        self._mean = df_train[self._name].mean()
        self._median = df_train[self._name].median()
        self._reordered_group = self._label_group_mean.sort_values(config.label_col).reset_index(drop=True)
        self._ro_name = '{}_{}'.format(Action.reorder, self._name)
        self._reordered_group.loc[:, self._ro_name] = self._reordered_group.index
        self._reordered_mean = self._reordered_group[self._ro_name].mean()
        self._reordered_median = self._reordered_group[self._ro_name].median()

    # From here there will be a bunch functions that will modify the input dataframe
    def _dropping(self, df):
        self._features.remove(self._name)

    def _one_hot(self, df):
        for val in self._unique_values:
            self.__add_features_if_not_exist(
                df,
                '{}_{}_{}'.format(Action.one_hot, self._name, val),
                lambda df: (df[self._name].values==val).astype(int)
            )

    def _reorder(self, df):
        def get_reordered_value(df):
            tmp = self._df_train.merge(
                self._reordered_group[[self._name, self._ro_name]], on=self._name
            )
            return tmp[self._ro_name]
        self.__add_features_if_not_exist(df, self._ro_name, get_reordered_value)

    def _reorder_above_mean(self, df):
        self.__add_features_if_not_exist(
            df,
            '{}_{}'.format(Action.reorder_above_mean, self._name),
            lambda df: (df[self._ro_name] > self._reordered_mean).astype(int)
        )

    def _reorder_above_median(self, df):
        self.__add_features_if_not_exist(
            df,
            '{}_{}'.format(Action.reorder_above_median, self._name),
            lambda df: (df[self._ro_name] > self._reordered_median).astype(int)
        )

    def _above_mean(self, df):
        self.__add_features_if_not_exist(
            df,
            '{}_{}'.format(Action.above_mean, self._name),
            lambda df: (df[self._name] > self._mean).astype(int)
        )

    def _above_median(self, df):
        self.__add_features_if_not_exist(
            df,
            '{}_{}'.format(Action.above_median, self._name),
            lambda df: (df[self._name] > self._median).astype(int)
        )

    def all_actions_without_dropping(self):
        actions = []
        # For between unique values between 3 and 6, we consider one hot encoding
        if self._num_unique_values > 2 and self._num_unique_values < 7:
            actions.append(Action.one_hot)

        # For categorical and binary features or continuous features with fewer
        # than 7 unique values, use reorder actions
        if self._type != Feature_type.continuous or self._num_unique_values < 7:
            actions.append(Action.reorder)
            actions.append(Action.reorder_above_mean)
            actions.append(Action.reorder_above_median)

        # For continuous features, consider mean and median actions
        if self._type == Feature_type.continuous:
            actions.append(Action.above_mean)
            actions.append(Action.above_median)

        return actions

    # This function will modify df
    def get_features(self, df, actions):
        if len(actions) != len(set(actions)):
            raise Exception('found duplicate actions')

        all_available_actions = set(self.all_actions_without_dropping())
        for action in actions:
            if action != Action.dropping and action not in all_available_actions:
                raise Exception('{} not allowed for {}'.format(action, self._name))

        self._features = [self._name]
        action_dict = {
            Action.dropping: self._dropping,
            Action.one_hot: self._one_hot,
            Action.above_mean: self._above_mean,
            Action.above_median: self._above_median,
            Action.reorder: self._reorder,
            Action.reorder_above_mean: self._reorder_above_mean,
            Action.reorder_above_median: self._reorder_above_median
        }
        for action in actions:
            action_dict[action](df)
        return self._features


# This is a replacement for the old feature extractor
class FeatureExtractorNew():
    def __init__(self):
        return

    def __hand_design(self, df):
        df.loc[:, 'ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']

    def __revert_one_hot(self, df):
        def hlp(features, new_feature):
            df.loc[:, new_feature] = -1
            for i, f in enumerate(features):
                if not (df[df[f]==1][new_feature] == -1).all():
                    raise Exception(
                        'Error in revert one hot encoding, check {}'.format(features)
                    )
                df.loc[:, new_feature] = (i+1) * (df[f]==1) + df[new_feature]
            for c in features:
                del df[c]
        hlp(
            ['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin'],
            'ps_ind_06_to_09_cat'
        )
        hlp(['ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin'], 'ps_ind_16_18_cat')

    def __count_nas(self, df):
        df.loc[:, 'ps_num_nas'] = (df==-1).sum(axis=1)

    def freeze_all_features_for_kfold(self, data_dir, n_splits, random_state, feature_dir):
        train_data = T(data_dir)
        test = pd.read_csv(config.data_test_file(data_dir))
        def preprocess(df):
            self.__hand_design(df)
            self.__revert_one_hot(df)
            self.__count_nas(df)
            return df
        for fold_num, (train, valid) in enumerate(train_data.kfold(n_splits, random_state)):
            gc.collect()
            df_train = preprocess(train.copy())
            df_valid = preprocess(valid.copy())
            df_test = preprocess(test.copy())
            features = []
            for f in df_train.columns:
                if f not in [config.id_col, config.label_col]:
                    f = Feature(f)
                    f.load_df_train(df_train)
                    actions = f.all_actions_without_dropping()
                    features += (f.get_features(df_train, actions))
                    _ = f.get_features(df_valid, actions)
                    _ = f.get_features(df_test, actions)

            save_to_file(
                config.get_feature_train_file(feature_dir, fold_num),
                lambda filename: df_train[
                    [config.id_col, config.label_col] + features
                ].to_pickle(filename)
            )
            save_to_file(
                config.get_feature_valid_file(feature_dir, fold_num),
                lambda filename: df_valid[
                    [config.id_col, config.label_col] + features
                ].to_pickle(filename)
            )
            save_to_file(
                config.get_feature_test_file(feature_dir, fold_num),
                lambda filename: df_test[[config.id_col] + features].to_pickle(filename)
            )
            print '{} fold {} is finished'.format(feature_dir, fold_num)

class FeatureExtractor():

    def __dropping_calc_features(self, df):
        for c in df.columns:
            if c.startswith('ps_calc'):
                del df[c]

    def __hand_design(self, df):
        df.loc[:, 'hd_ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']

    def __one_hot(self, df, c):
        for val in self.__unique_vals[c]:
            df.loc[:, 'oh_{}_{}'.format(c, val)] = (df[c].values==val).astype(int)

    def __reorder(self, df, c):
        new_f = 'ro_' + c
        if self.__reorder_map.get(c) is None:
            tmp = df[[config.label_col, c]].groupby(c, as_index=False).mean()
            tmp = tmp.sort_values(config.label_col).reset_index(drop=True)
            tmp.loc[:, new_f] = tmp.index
            del tmp[config.label_col]
            self.__reorder_map[c] = tmp.copy()
        merged = df[[c]].merge(self.__reorder_map[c], how='left', on=c).fillna(-1)
        df.loc[:, new_f] = merged[new_f].values

    def get_reorder_map(self, c):
        return self.__reorder_map[c]

    def __count_missing(self, df):
        df_orig_features = df[[c for c in df.columns if c.startswith('ps')]] 
        df.loc[:, 'missing_count'] = (df_orig_features==-1).sum(axis=1)

    def __mean_range(self, df, c):
        df.loc[:, 'mean_range_'+c] = (df[c].astype(float).values > self.__df_mean[c]).astype(int)

    def __median_range(self, df, c):
        df.loc[:, 'med_range_'+c] = (df[c].astype(float).values > self.__df_median[c]).astype(int)

    def __revert_one_hot(self, df, features, new_feature):
        new_f = 'roh_{}'.format(new_feature)
        df.loc[:, new_f] = -1
        for i, f in enumerate(features):
            if not (df[df[f]==1][new_f] == -1).all():
                raise Exception('Error in revert one hot encoding, check {}'.format(features))
            df.loc[:, new_f] = (i+1) * (df[f]==1) + df[new_f]
        for c in features:
            del df[c]

    def __drop_cols_hard_coded(self, df):
        cols_to_drop = [
            'ps_ind_10_bin',
            'ps_ind_11_bin',
            'ps_ind_12_bin',
            'ps_ind_13_bin',
        ]
        for c in cols_to_drop:
            if c in df.columns:
                del df[c]

    def __init__(self):
        self.__retired = False
        self.__handled_training_data = False
        self.__reorder_map = {}
        return None

    def _convert(self, df):
        df = df.copy()
        self.__dropping_calc_features(df)
        self.__hand_design(df)
        self.__revert_one_hot(df, ['ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin'], 'ps_ind_16_18')
        self.__drop_cols_hard_coded(df)
        self.__count_missing(df)

        if not self.__handled_training_data:
            self.__unique_vals = {
                c: list(df[c].unique()) for c in df.columns
                if c not in [config.id_col, config.label_col]
            }
            self.__num_unique_vals = {
                c: len(df[c].unique()) for c in df.columns
                if c not in [config.id_col, config.label_col]
            }
            self.__one_hot_cols = []
            self.__freq_cols = []
            self.__drop_cols = []
            self.__reorder_cols = []
            for c in df.columns:
                if c not in [config.id_col, config.label_col]:
                    if self.__num_unique_vals[c] > 2 and self.__num_unique_vals[c] < 7:
                        self.__one_hot_cols.append(c)
                        if c.endswith(('bin', 'cat')):
                            self.__drop_cols.append(c)
                    if c.endswith('cat'):
                        if self.__num_unique_vals[c] > 7 and self.__num_unique_vals[c] < 20:
                            self.__drop_cols.append(c)
                            self.__reorder_cols.append(c)
                        elif self.__num_unique_vals[c] >= 20:
                            self.__drop_cols.append(c)
                    '''
                    if not c.endswith(('cat', 'bin')):
                        if self.__num_unique_vals[c] < 5:
                            self.__reorder_cols.append(c)
                    '''

        for c in self.__one_hot_cols:
            self.__one_hot(df, c)
        for c in self.__reorder_cols:
            self.__reorder(df, c)
        for c in self.__drop_cols:
            del df[c]

        if not self.__handled_training_data:
            self.__df_median = df.median(axis=0)
            self.__df_mean = df.mean(axis=0)

        for c in df.columns:
            if c not in [config.id_col, config.label_col]:
                if c.startswith('ro_'):
                    self.__mean_range(df, c)
                    self.__median_range(df, c)

                if c.startswith('ps_') and (not c.endswith(('cat', 'bin'))): 
                    self.__mean_range(df, c)
                    self.__median_range(df, c)

        self.__handled_training_data = True
        return df

    '''
    if excluded features are not specified, then don't do any filtering. Otherwise, filter out
    all the excluded features
    '''
    def convert(self, df_train, df_valid, df_test, excluded_features=None):
        assert (not self.__retired)
        # order matters here, must convert df_train first
        df_train = self._convert(df_train)
        print ('train data is converted')
        df_valid = None if df_valid is None else self._convert(df_valid)
        print ('validation data is converted')
        df_test = None if df_test is None else self._convert(df_test)
        print ('test data is converted')
        self.__retired = True
        def filter_features(df, excluded_features):
            if excluded_features is None:
                return df
            else:
                cols = [c for c in df.columns if c not in excluded_features]
                return df[cols]
        return filter_features(df_train, excluded_features), \
            filter_features(df_valid, excluded_features), \
            filter_features(df_test, excluded_features)
