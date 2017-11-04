#!/usr/bin/env python2
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
to_remove = ['id','target']
param = {'random_state':42}
single_model = model.RandomForest(data_dir=data_dir, param=param)
single_model.train_predict_eval_and_log()
