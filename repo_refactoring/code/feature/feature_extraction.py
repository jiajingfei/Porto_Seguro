import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import pandas as pd
from configs.base_config import *
from abc import ABCMeta, abstractmethod
from utils import save_df_to_file

class FeatureExtractor():
    def __init__(
            self,
            df_or_csv,
            id_col = 'id',
            cols = None # Cols to use for feature extraction
    ):
        if isinstance(df_or_csv, str):
            self._df = pd.read_csv(df_or_csv)
        else:
            self._df = df_or_csv 
        self._id_col = id_col
        self._cols = cols

    def PeekDF(self):
        return self._df

    def DoNothing(self):
        print "Do Nothing."

    # Other feature operations to manipulate self._df

    def SaveToFile(
            self,
            output_file,
            overwrite=False
    ):
        save_df_to_file(self._df, output_file, overwrite)
            
if __name__ == '__main__':
    extractor = FeatureExtractor('../../data/train.csv')
    extractor.DoNothing()
    extractor.SaveToFile('../../data/extractor_test.csv')
