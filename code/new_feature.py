import os
import pandas as pd
import config
import gc
import argparse
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

class Action_type(object):
    dropping = 'dropping'
    raw = 'raw'
    raw_more = 'raw_more'
    one_hot = 'one_hot'
    reorder = 'reorder'
    reorder_more = 'reorder_more'
    @staticmethod
    def all():
        return [
            Action_type.dropping,
            Action_type.raw,
            Action_type.raw_more,
            Action_type.one_hot,
            Action_type.reorder,
            Action_type.reorder_more
        ]

class Feature(object):

    def __init__(self, name):
        if not name.startswith('ps_'):
            raise Exception('{} is not a valid feature'.format(name))
        self._name = name
        self._ro_name = '{}_{}'.format(Action.reorder, self._name)
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
        if self._num_unique_values > 2 and self._num_unique_values < 7:
            actions.append(Action.one_hot)

        if self._num_unique_values < 7:
            actions.append(Action.reorder)
            actions.append(Action.reorder_above_mean)
            actions.append(Action.reorder_above_median)

        if self._type == Feature_type.continuous:
            actions.append(Action.above_mean)
            actions.append(Action.above_median)
        return actions

    def _has_one_hot(self, df):
        return len([
            c for c in df.columns
            if c.startswith('{}_{}'.format(Action.one_hot, self._name))
        ]) > 0

    def _has_reorder(self, df):
        return len([
            c for c in df.columns
            if c.startswith('{}_{}'.format(Action.reorder, self._name))
        ]) > 0

    def _generate_actions(self, df, action_type):
        if action_type == Action_type.dropping:
            return [Action.dropping]
        elif action_type == Action_type.raw:
            return []
        elif action_type == Action_type.raw_more:
            if self._type == Feature_type.continuous:
                return [Action.above_mean, Action.above_median]
            else:
                return []
        elif action_type == Action_type.one_hot:
            if self._has_one_hot(df):
                return [Action.one_hot, Action.dropping]
            else:
                return self._generate_actions(df, Action_type.raw)
        elif action_type == Action_type.reorder:
            if self._has_reorder(df):
                return [Action.reorder]
            else:
                return self._generate_actions(df, Action_type.raw)
        elif action_type == action_type.reorder_more:
            if self._has_reorder(df):
                return [
                    Action.reorder,
                    Action.reorder_above_mean,
                    Action.reorder_above_median,
                ]
            else:
                return self._generate_actions(df, Action_type.raw_more)
        elif action_type == Action_type.prefer_one_hot:
            if self._has_one_hot(df):
                return [Action.one_hot, Action.dropping]
            elif self._has_reorder(df):
                return [
                    Action.reorder,
                    Action.reorder_above_mean,
                    Action.reorder_above_median,
                ]
            else:
                return self._generate_actions(df, Action_type.raw)
        else:
            raise Exception('Wrong action type {}')

    def get_features(self, df, action_type):
        features = []
        actions = self._generate_actions(df, action_type)
        if Action.dropping in actions:
            features = []
        else:
            features = [self._name]
        for action in actions:
            if action != Action.dropping:
                features += [
                    c for c in df.columns
                    if c.startswith('{}_{}'.format(action, self._name))
                ]
        return features

class FeatureExtractor():
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
        test = pd.read_csv(config.get_data_file(data_dir, 'test'))
        data_test_target_file = config.get_data_file(data_dir, 'test_label')
        print data_test_target_file
        if os.path.exists(data_test_target_file):
            test_target = pd.read_csv(data_test_target_file)
        else:
            test_target = None
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
                config.get_feature_file(feature_dir, fold_num, 'train'),
                lambda filename: df_train[
                    [config.id_col, config.label_col] + features
                ].to_pickle(filename)
            )
            save_to_file(
                config.get_feature_file(feature_dir, fold_num, 'valid'),
                lambda filename: df_valid[
                    [config.id_col, config.label_col] + features
                ].to_pickle(filename)
            )
            save_to_file(
                config.get_feature_file(feature_dir, fold_num, 'test'),
                lambda filename: df_test[[config.id_col] + features].to_pickle(filename)
            )
            print '{} fold {} is finished'.format(feature_dir, fold_num)

        if test_target is not None:
            save_to_file(
                config.get_feature_file(feature_dir, None, 'test_label'),
                lambda filename: test_target.to_pickle(filename)
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract all features and save to file')
    parser.add_argument('--data-dir', '-d', dest='data_dir', type=str, required=True)
    parser.add_argument('--feature-dir', '-f', dest='feature_dir', type=str, required=True)
    parser.add_argument('--n-splits', '-n', dest='n_splits', type=int, default=5)
    parser.add_argument('--random-seed', '-r', dest='seed', type=int, required=True)
    args = parser.parse_args()
    fe = FeatureExtractor()
    fe.freeze_all_features_for_kfold(args.data_dir, args.n_splits, args.seed, args.feature_dir)
