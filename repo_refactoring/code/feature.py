import os
import pandas as pd
import config

class FeatureExtractor():

    def __dropping_calc_features(self):
        df = self._df
        for c in df.columns:
            if c.startswith('ps_calc'):
                del df[c]

    def __revert_one_hot(self, features, new_feature):
        new_f = 'reverted_one_hot_{}_cat'.format(new_feature)
        df = self._df
        df.loc[:, new_f] = -1
        for i, f in enumerate(features):
            if not (df[df[f]==1][new_f] == -1).all():
                raise Exception('Error in revert one hot encoding, check {}'.format(features))
            df.loc[:, new_f] = (i+1) * (df[f]==1) + df[new_f]

    def __reorder(self, feature):
        # only reorder categorical and binary feature
        assert feature.endswith(('cat', 'bin'))
        feature_map = self.__reorder_map.get(feature)
        if feature_map is None:
            tmp = self._df[[config.label_col, feature]].groupby(feature, as_index=False).mean()
            tmp = tmp.sort_values(config.label_col).reset_index(drop=True)
            tmp.loc[:, 'reordered_'+feature] = tmp.index
            del tmp[config.label_col]
            self.__reorder_map[feature] = tmp
            feature_map = tmp
        self._df = self._df.merge(feature_map, on=feature)

    def __count_missing(self):
        df = self._df
        df_orig_features = df[[c for c in df.columns if c.startswith('ps')]] 
        df.loc[:, 'missing_count'] = (df_orig_features==-1).sum(axis=1)

    def __init__(self):
        self.__reorder_map = {}
        return None

    def load_df(self, df):
        self._df = df
        self.__dropping_calc_features()
        self.__revert_one_hot(
            ['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin'],
            'ps_ind_06_09'
        )
        self.__revert_one_hot(
            ['ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin'],
            'ps_ind_16_18'
        )
        for c in self._df.columns:
            if c.endswith(('bin', 'cat')):
                self.__reorder(c)
        self.__count_missing()

    def get_df(self, features=None):
        if features is None:
            return self._df
        else:
            cols = features + [config.id_col]
            if config.label_col in self._df.columns:
                cols += [config.label_col]
            return self._df[cols]
