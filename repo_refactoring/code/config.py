import os
this_path = os.path.dirname(os.path.realpath(__file__))

##################################################################
# DataFrame configs
##################################################################
id_col = 'id'
label_col = 'target'

##################################################################
# Data configs
##################################################################
__data_base_dir = os.path.join(this_path, '../data/')

def get_data_dir(dir_name):
    return os.path.join(__data_base_dir, dir_name)

def data_train_file(dir_name):
    return os.path.join(__data_base_dir, dir_name, 'train.csv')

def data_test_file(dir_name):
    return os.path.join(__data_base_dir, dir_name, 'test.csv')

def data_test_target_file(dir_name):
    return os.path.join(__data_base_dir, dir_name, '.test_target.csv')

def data_readme_file(dir_name):
    return os.path.join(__data_base_dir, dir_name, 'readme.txt')

data_raw_dir = 'raw_data'
data_sanity_dir = 'sanity_data'

##################################################################
# prediction configs 
##################################################################
__pred_base_dir = os.path.join(this_path, '../pred/')

def get_pred_dir(dir_name):
    return os.path.join(__pred_base_dir, dir_name)

def pred_filename(dir_name, filename, identifier):
    return os.path.join(get_pred_dir(dir_name), '{}-{}.csv'.format(identifier, filename))

def pred_log_file(dir_name):
    return os.path.join(get_pred_dir(dir_name), 'pred_log.csv')

##################################################################
# Model param configs 
##################################################################
__base_model_dir = os.path.join(this_path, '../model/')  # heck!

def get_model_dir(dir_name):
    return os.path.join(__base_model_dir, dir_name)

def model_filename(dir_name, filename, identifier):
    return os.path.join(get_model_dir(dir_name), '{}-{}.pickle'.format(identifier, filename))

def model_log_file(dir_name):
    return os.path.join(get_model_dir(dir_name), 'model_log.csv')
