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
# feature configs
##################################################################
__feature_base_dir = os.path.join(this_path, '../features/')

def get_feature_dir(dir_name, fold_num=None):
    directory = os.path.join(__feature_base_dir, dir_name)
    if fold_num is None:
        return directory
    else:
        return os.path.join(directory, 'fold{}'.format(fold_num))

def get_feature_train_file(dir_name, fold_num):
    return os.path.join(get_feature_dir(dir_name, fold_num), 'train.pickle')

def get_feature_valid_file(dir_name, fold_num):
    return os.path.join(get_feature_dir(dir_name, fold_num), 'valid.pickle')

def get_feature_test_file(dir_name, fold_num):
    return os.path.join(get_feature_dir(dir_name, fold_num), 'test.pickle')

def get_num_folds(dir_name):
    folds = [d for d in os.listdir(get_feature_dir(dir_name)) if d.startswith('fold')]
    return len(folds)

##################################################################
# prediction configs 
##################################################################
__pred_base_dir = os.path.join(this_path, '../pred/')

def get_pred_dir(dir_name):
    return os.path.join(__pred_base_dir, dir_name)

def pred_filename(dir_name, filename):
    return os.path.join(get_pred_dir(dir_name), '{}.csv'.format(filename))

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
