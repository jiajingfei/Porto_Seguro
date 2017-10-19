#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 23:01:42 2017

@author: changyaochen
"""

import pandas as pd
import config
import model

data_dir = config.get_data_dir('sanity_data')

# is this the best way to get feature list?
df = pd.read_csv(data_dir+'/train.csv')
feature = df.columns.tolist()
to_remove = ['id','target']
for x in feature:
    if '_calc_' in x:
        to_remove.append(x)
for x in to_remove:
    feature.remove(x)

param = {'features':feature, 'random_state':42}
single_model = model.RandomForest(data_dir=data_dir, param=param)
single_model.kfold_train_predict_eval(None)