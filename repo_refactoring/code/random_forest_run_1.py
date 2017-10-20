#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 23:01:42 2017

@author: changyaochen
"""
import os
import pandas as pd
import config
import model
from data_type import Training_data

data_dir = config.get_data_dir('sanity_data')
data_dir = 'sanity_data'

# is this the best way to get feature list?
training_data = Training_data(config.data_raw_dir)
if not os.path.exists(config.data_train_file(data_dir)):
    training_data.complete_data_dir_with_ids('sanity_data')

df = pd.read_csv(config.data_train_file(data_dir))

feature = df.columns.tolist()
to_remove = ['id','target']
for x in feature:
    if '_calc_' in x:
        to_remove.append(x)
for x in to_remove:
    feature.remove(x)

param = {'features':feature, 'random_state':42, 'verbose':1, 'n_splits': 3}
single_model = model.RandomForest(data_dir=data_dir, param=param)
single_model.train_predict_eval_and_log()