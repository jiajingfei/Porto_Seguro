import pandas as pd
from utils import *

class Kfold_XGB(Model):
    def __init__(self, params):
        super(Kfold_XGB, self).__init__(params)
        self._kfold = Kfold(params)
        self._xgb = Xgb_trees(params)
        self._feature_transformation = Feature_transformantion(params)

    def train_and_predict(self, df_train, df_test):
        df_train, df_test = self._feature_transformation.transform(df_train, df_test)
        dfs = self._kfold.split_data(df_train)
        k = len(dfs)
        pred = 0
        for train, test in dfs:
            self._xgb.train(train, test)
            pred += self._xgb.pred(df_test)

        pred = (pred.target - pred.target.min()) / (pred.target.max() - pred.target.min()) / 1.1
        return pd.DataFrame(data = {'id': df_test['id'], 'target': pred})
