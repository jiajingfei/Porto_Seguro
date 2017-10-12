import os
this_path = os.path.dirname(os.path.realpath(__file__))


##################################################################
# Data configs
##################################################################
id_col = 'id'
label_col = 'target'
train_file = 'train.csv'
test_file = 'test.csv'
test_target_file = '.test_target.csv'
base_data_dir = os.path.join(this_path, '../../data/')
raw_data_dir = os.path.join(base_data_dir, 'raw_data')
sanity_data_dir = os.path.join(base_data_dir, 'sanity_data')


##################################################################
# Feature configs
##################################################################
base_feature_dir = os.path.join(this_path, '../../features/')
