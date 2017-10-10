import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import pandas as pd
from configs.base_config import *
from abc import ABCMeta, abstractmethod
from utils import outputfile_already_exists

class Proto():
    def __init__(
            self,
            id_col='id',
            cols=None # Cols to use for feature extraction, default is all but id_col
    ):
        self._id_col = id_col
        return None

    @abstractmethod
    def apply(self, df):
        # Return some feature dataframe
        return None

    def apply_on_file(
            self,
            input_file,
            output_file,
            overwrite=False
    ):
        df = pd.read_csv(input_file)
        # CR jjia: could check the last modified time to determine whether to rerun this
        if not outputfile_already_exists(output_file) or overwrite:
            self.apply(df).to_csv(output_file)
        else:
            print "WARNING: output file for {} already exists".format(self.__class__.__name__)
