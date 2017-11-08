import pandas as pd
import os
import config

class Ensembler():

    @staticmethod
    def load_file(dir_name, identifier, mode, file_type='pickle'):
        if mode == 'valid':
            prefix = 'valid'
        elif mode == 'test':
            prefix = 'test-sum'
        else:
            raise Exception('wrong mode {}, not in [valid|test]'.format(mode))
        pred_dir = config.get_pred_dir(dir_name)
        files = os.listdir(pred_dir)
        relevant_files = [f for f in files if f.startswith(identifier)]
        relevant_files = [
            f for f in relevant_files
            if f.startswith(identifier+'-'+prefix)
        ]
        if file_type == 'pickle':
            read_file = pd.read_pickle
        elif file_type == 'csv':
            read_file = pd.read_csv
        else:
            raise Exception('wrong file_type {}, not in [pickle|csv]'.format(file_type))

        relevant_files.sort()
        dfs = [read_file(os.path.join(pred_dir, f)) for f in relevant_files]
        return pd.concat(dfs)
