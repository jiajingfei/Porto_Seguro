from utils import *

class Kfold_XGB(Model):
    def __init__(self, params):
        super(Kfold_XGB, self).__init__(params)
        self._kfold = Kfold(params)
        self._xgb = Xgb_trees(params)
        self._feature_transformation = Feature_transformantion(param)

    def train_and_predict(self, df_train, df_test):
        raise Exception("to be done")
