import os
import numpy as np
import random, string
import datetime as dt

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
