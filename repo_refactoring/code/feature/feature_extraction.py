import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import pandas as pd
from utils import save_df_to_file

class FeatureExtractor():
    def __init__(self, df_or_csv):
        if isinstance(df_or_csv, str):
            self._df = pd.read_csv(df_or_csv)
        else:
            self._df = df_or_csv 

    def peek_df(self):
        return self._df

    def do_nothing(self):
        print "Do Nothing."

    # Other feature operations to manipulate self._df

    def save_to_file(
            self,
            output_file,
            overwrite=False
    ):
        save_df_to_file(self._df, output_file, overwrite)

if __name__ == '__main__':
    extractor = FeatureExtractor('../../data/train.csv')
    extractor.do_nothing()
    extractor.save_to_file('../../data/extractor_test.csv')
