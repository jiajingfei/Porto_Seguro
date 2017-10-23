import os
import pandas as pd
import config

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
            self.__one_hot_cols = []
            self.__drop_cols = []
            self.__reorder_cols = []
            for c in df.columns:
                if c not in [config.id_col, config.label_col]:
                    if len(self.__unique_vals[c]) > 2 and len(self.__unique_vals[c]) < 7: 
                        self.__one_hot_cols.append(c)
                        if c.endswith(('bin', 'cat')):
                            self.__drop_cols.append(c)
                    if c.endswith('cat'):
                        if len(self.__unique_vals[c]) > 7 and len(self.__unique_vals[c]) < 20:
                            self.__reorder_cols.append(c)
                            self.__drop_cols.append(c)
                        elif len(self.__unique_vals[c]) >= 20:
                            self.__drop_cols.append(c)

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
    if features are not specified, then don't do any filtering. Otherwise, filter out
    features that are not in given features
    '''
    def convert(self, df_train, df_valid, df_test, features=None):
        assert (not self.__retired)
        # order matters here, must convert df_train first
        df_train = self._convert(df_train)
        print ('train data is converted')
        df_valid = None if df_valid is None else self._convert(df_valid)
        print ('validation data is converted')
        df_test = None if df_test is None else self._convert(df_test)
        print ('test data is converted')
        self.__retired = True
        def filter_features(df, features):
            if features is None:
                return df
            else:
                if config.label_col in df.columns:
                    cols = features + [config.id_col, config.label_col]
                else:
                    cols = features + [config.id_col]
                return df[cols]
        return filter_features(df_train, features), \
            filter_features(df_valid, features), \
            filter_features(df_test, features)
