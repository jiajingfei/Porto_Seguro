#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 21:43:47 2017

@author: changyaochen
"""


def generate_data(id_filename, save_filename, 
                  keep_label=True):
    '''
    This code is to read the sanity data, from the id
    
    Input
    ---------
    id_filename: <str>
        filename for the id file, not including the path
    save_filename: <str>
        filename for the 'real' data file
    keep_label: <Boolean>, default: True
        If False, remove the label column
    
    Return
    ---------
    Nothing, but create file(s) under the same folder.
    If the file(s) already exist(s), we will overwrite
    '''
    
    import pandas as pd
    import sys
    # get config files
    sys.path.append('../../code')
    import config
    
    # get indices
    with open(id_filename, 'r') as f:
        idx = f.readlines()
    idx = list(map(int, idx))
    
    # get the data
    raw_data = '../raw_data/train.csv'
    raw_df = pd.read_csv(raw_data)
    df = raw_df[raw_df[config.id_col].isin(idx)]
    if not keep_label:
        df.loc[:, [config.id_col, config.label_col]].to_csv(
                '.'+save_filename.split('.')[0]+'_target.csv', index=False)
        df.drop(config.label_col, axis=1, inplace=True)
    df.to_csv(save_filename, index=False)
    
    return 
    
    
if __name__ == '__main__':
    generate_data('train.csv.id', 'train.csv')
    generate_data('test.csv.id', 'test.csv', keep_label=False)