import pandas as pd
import numpy as np
import datetime as dt
import os
from sklearn.model_selection import StratifiedShuffleSplit

this_path = os.path.dirname(os.path.realpath(__file__))

base_dir = os.path.join(this_path, '../data/')
train_csv = os.path.join(base_dir, 'train.csv')

def split(df, test_ratio=0.3):
    target_dir = os.path.join(base_dir, str(dt.date.today()))
    if os.path.exists(target_dir):
        raise Exception('{} already exists'.format(target_dir))
    else:
        os.system('mkdir -p {}'.format(target_dir))
    test_size = int(df.shape[0] * test_ratio)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    indices = sss.split(X=df, y=df.target).next()
    df_train = df.iloc[indices[0]]
    df_test = df.iloc[indices[1]]
    df_train.to_csv(os.path.join(target_dir, 'train.csv'), index=False)
    df_test.target.to_csv(os.path.join(target_dir, '.test_target.csv'), index=False)
    del df_test['target']
    df_test.to_csv(os.path.join(target_dir, 'test.csv'), index=False)
    return None


if __name__ == '__main__':
    df = pd.read_csv(train_csv)
    split(df)
