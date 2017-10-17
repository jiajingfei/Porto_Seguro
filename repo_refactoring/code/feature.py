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
        df = self._df
        new_f = 'reverted_one_hot_{}_cat'.format(new_feature)
        df.loc[:, new_f] = -1
        for i, f in enumerate(features):
            if not (df[df[f]==1][new_f] == -1).all():
                raise Exception('Error in revert one hot encoding, check {}'.format(features))
            df.loc[:, new_f] = (i+1) * (df[f]==1) + df[new_f]

    def __reorder(self, feature):
        df = self._df
        # only reorder categorical and binary feature
        assert feature.endswith(('cat', 'bin'))
        feature_map = self.__reorder_map.get(feature)
        if feature_map is None:
            tmp = df[[config.label_col, feature]].groupby(feature, as_index=False).mean()
            tmp = tmp.sort_values(config.label_col).reset_index(drop=True)
            tmp.loc[:, 'reordered_'+feature] = tmp.index
            del tmp[config.label_col]
            self.__reorder_map[feature] = tmp
            feature_map = tmp
        self._df = df.merge(feature_map, how='left', on=feature).fillna(-1)

    def __count_missing(self):
        df = self._df
        df_orig_features = df[[c for c in df.columns if c.startswith('ps')]] 
        df.loc[:, 'missing_count'] = (df_orig_features==-1).sum(axis=1)

    def __init__(self):
        self.__reorder_map = {}
        return None

    def _convert(self, df):
        self._df = df.copy()
        self.__dropping_calc_features()
        self.__revert_one_hot(
            ['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin'],
            'ps_ind_06_09'
        )
        self.__revert_one_hot(
            ['ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin'],
            'ps_ind_16_18'
        )
        self.__count_missing()
        for c in self._df.columns:
            if c.endswith(('bin', 'cat')):
                self.__reorder(c)
        return self._df

    def convert(self, df_train, df_valid, df_test, features):
        # order matters here, must convert df_train first
        df_train = self._convert(df_train)
        df_valid = None if df_valid is None else self._convert(df_valid)
        df_test = None if df_test is None else self._convert(df_test)
        def to_features(df, for_test):
            if df is None:
                return None
            elif for_test:
                return df[features+[config.id_col]]
            else:
                return df[features+[config.id_col, config.label_col]]
        return to_features(df_train, False), \
            to_features(df_valid, False), \
            to_features(df_test, True)

    @staticmethod
    def all_features():
        return [
            'ps_ind_01',
            'ps_ind_02_cat',
            'ps_ind_03',
            'ps_ind_04_cat',
            'ps_ind_05_cat',
            'ps_ind_06_bin',
            'ps_ind_07_bin',
            'ps_ind_08_bin',
            'ps_ind_09_bin',
            'ps_ind_10_bin',
            'ps_ind_11_bin',
            'ps_ind_12_bin',
            'ps_ind_13_bin',
            'ps_ind_14',
            'ps_ind_15',
            'ps_ind_16_bin',
            'ps_ind_17_bin',
            'ps_ind_18_bin',
            'ps_reg_01',
            'ps_reg_02',
            'ps_reg_03',
            'ps_car_01_cat',
            'ps_car_02_cat',
            'ps_car_03_cat',
            'ps_car_04_cat',
            'ps_car_05_cat',
            'ps_car_06_cat',
            'ps_car_07_cat',
            'ps_car_08_cat',
            'ps_car_09_cat',
            'ps_car_10_cat',
            'ps_car_11_cat',
            'ps_car_11',
            'ps_car_12',
            'ps_car_13',
            'ps_car_14',
            'ps_car_15',
            'reverted_one_hot_ps_ind_06_09_cat',
            'reverted_one_hot_ps_ind_16_18_cat',
            'missing_count',
            'reordered_ps_ind_02_cat',
            'reordered_ps_ind_04_cat',
            'reordered_ps_ind_05_cat',
            'reordered_ps_ind_06_bin',
            'reordered_ps_ind_07_bin',
            'reordered_ps_ind_08_bin',
            'reordered_ps_ind_09_bin',
            'reordered_ps_ind_10_bin',
            'reordered_ps_ind_11_bin',
            'reordered_ps_ind_12_bin',
            'reordered_ps_ind_13_bin',
            'reordered_ps_ind_16_bin',
            'reordered_ps_ind_17_bin',
            'reordered_ps_ind_18_bin',
            'reordered_ps_car_01_cat',
            'reordered_ps_car_02_cat',
            'reordered_ps_car_03_cat',
            'reordered_ps_car_04_cat',
            'reordered_ps_car_05_cat',
            'reordered_ps_car_06_cat',
            'reordered_ps_car_07_cat',
            'reordered_ps_car_08_cat',
            'reordered_ps_car_09_cat',
            'reordered_ps_car_10_cat',
            'reordered_ps_car_11_cat',
            'reordered_reverted_one_hot_ps_ind_06_09_cat',
            'reordered_reverted_one_hot_ps_ind_16_18_cat' 
        ]

    @staticmethod
    def recommended_features():
        return [
            'ps_ind_01',
            'ps_ind_03',
            'ps_ind_14',
            'ps_ind_15',
            'ps_reg_01',
            'ps_reg_02',
            'ps_reg_03',
            'ps_car_11',
            'ps_car_12',
            'ps_car_13',
            'ps_car_14',
            'ps_car_15',
            'missing_count',
            'reordered_ps_ind_02_cat',
            'reordered_ps_ind_04_cat',
            'reordered_ps_ind_05_cat',
            'reordered_ps_ind_10_bin',
            'reordered_ps_ind_11_bin',
            'reordered_ps_ind_12_bin',
            'reordered_ps_ind_13_bin',
            'reordered_ps_car_01_cat',
            'reordered_ps_car_02_cat',
            'reordered_ps_car_03_cat',
            'reordered_ps_car_04_cat',
            'reordered_ps_car_05_cat',
            'reordered_ps_car_06_cat',
            'reordered_ps_car_07_cat',
            'reordered_ps_car_08_cat',
            'reordered_ps_car_09_cat',
            'reordered_ps_car_10_cat',
            # 'reordered_ps_car_11_cat', # too many categories
            'reordered_reverted_one_hot_ps_ind_06_09_cat',
            'reordered_reverted_one_hot_ps_ind_16_18_cat' 
        ]

def test_features():
    from data_type import Training_data as Training_data
    if not os.path.exists(config.get_data_dir(config.data_sanity_dir)):
        training_data = Training_data(config.data_raw_dir)
        training_data.output_small_data_for_sanity_check(config.data_sanity_dir)
    F = FeatureExtractor
    f = F()
    df_train = pd.read_csv(config.data_train_file(config.data_sanity_dir))
    features_train = f._convert(df_train)
    assert (set(features_train.columns[2:]) == set(F.all_features()))
    assert (set(F.recommended_features()) < set(F.all_features()))
