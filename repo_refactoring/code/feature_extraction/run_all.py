import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), './'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from configs.base_config import *
from raw_features import Raw_features


##################################################################
# Generate raw data (basically just copying)
##################################################################
raw_features = Raw_features()
raw_features.apply_on_file(
    input_file = raw_train_file,
    output_file = os.path.join(train_feature_dir, 'raw.csv')
)
raw_features.apply_on_file(
    input_file = raw_test_file,
    output_file = os.path.join(test_feature_dir, 'raw.csv')
)

