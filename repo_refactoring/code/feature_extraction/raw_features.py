import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), './'))
from feature_extraction import Proto as P

# Example of raw features extraction
class Raw_features(P):
    def apply(self, df):
        return df
