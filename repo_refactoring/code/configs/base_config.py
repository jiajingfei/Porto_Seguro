import os
this_path = os.path.dirname(os.path.realpath(__file__))

##################################################################
# Data configs
##################################################################

base_data_dir = os.path.join(this_path, '../../data/')
raw_data_dir = os.path.join(base_data_dir, 'raw_data')
raw_train_file = os.path.join(raw_data_dir, 'train.csv')
raw_test_file = os.path.join(raw_data_dir, 'test.csv')

##################################################################
# Feature configs
##################################################################

base_feature_dir = os.path.join(this_path, '../../features/')
train_feature_dir = os.path.join(base_feature_dir, 'train')
test_feature_dir = os.path.join(base_feature_dir, 'test')
