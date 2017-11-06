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

def get_data_file(dir_name, mode):
    data_dir = get_data_dir(dir_name)
    filename_dict = {
        'train': 'train.csv',
        'test': 'test.csv',
        'test_label': '.test_target.csv',
        'readme': 'readme.txt'
    }
    filename = filename_dict.get(mode)
    if filename is None:
        raise Exception('Wrong mode {}, mode must be in {}'.format(mode, filename_dict.keys()))
    else:
        return os.path.join(data_dir, filename)

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

def get_feature_file(dir_name, fold_num, mode):
    if mode == 'test_label':
        fold_num = None
    feature_dir = get_feature_dir(dir_name, fold_num)
    filename_dict = {
        'train': 'train.pickle',
        'valid': 'valid.pickle',
        'test': 'test.pickle',
        'test_label': '.test_target.pickle',
    }
    filename = filename_dict.get(mode)
    if filename is None:
        raise Exception('wrong mode {}, mode must be train|test|valid|test_label')
    else:
        return os.path.join(feature_dir, filename)

def get_num_folds(dir_name):
    folds = [d for d in os.listdir(get_feature_dir(dir_name)) if d.startswith('fold')]
    return len(folds)

def get_orig_features(df):
    return [c for c in df.columns if c.startswith('ps_')]

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
