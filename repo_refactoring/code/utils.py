import os
import numpy as np
import random, string
import datetime as dt
import config
import lightgbm as lgb
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def remove_id_and_label(df):
    cols = [c for c in df.columns if c not in [config.id_col, config.label_col]]
    return df[cols]

def save_to_file(filename, save_fn, allow_existing=False):
    if os.path.isfile(filename):
        if not allow_existing:
            raise Exception('{} already exists'.format(filename))
    path, _ = os.path.split(filename)
    if not os.path.isdir(path):
        os.system('mkdir -p {}'.format(path))
    save_fn(filename)

def save_df_to_file(df, filename, overwrite=False):
    save_to_file(filename, lambda f: df.to_csv(f, index=False), allow_existing=overwrite)

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

def unique_identifier():
    return str(int((dt.datetime.now()-dt.datetime(2017,10,17)).total_seconds()))

def test_gini_normalized():
    a = np.random.normal(size = 100)
    p = np.random.normal(size = 100)
    assert(gini_normalized(a, p) < 1)
    assert(np.isclose(gini_normalized(a, a), 1))

def join_model_and_params(model_dir):
    df = pd.read_csv(os.path.join(model_dir, 'model_log.csv'))
    identifiers = df.identifier.unique()
    params = []
    for id in identifiers:
        with open(
            os.path.join(model_dir, '{}-param.pickle'.format(id)), 'rb'
        ) as handle:
            param = pickle.load(handle)
            param['identifier'] = id
        params.append(param)
    p = pd.DataFrame(data = params)
    df = df.merge(p, on='identifier', how='left')
    df.to_csv(os.path.join(model_dir, 'model_log_with_param.csv'))

# To use the following function in jupyter notebook, you can run the following two lines
# %run "PATH TO THIS FILE"
# %matplotlib inline
def feature_importance(df, features=None, n_estimators=500):
    y = df[config.label_col]
    X= remove_id_and_label(df)
    if features is not None:
        X = X[features]
    model = lgb.LGBMClassifier(
        learning_rate=0.05,
        n_estimators=n_estimators,
        max_depth=5,
        num_leaves=32,
        max_bin=10,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=10
    )
    model.fit(X, y)
    lgb.plot_importance(model, figsize=(15,25))
    return model
