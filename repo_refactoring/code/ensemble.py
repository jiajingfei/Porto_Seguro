import pandas as pd
import os
import config

class Ensembler():

    @staticmethod
    def load_file(filename_with_dir, key=None):
        if key is None:
            _, filename = os.path.split(filename_with_dir)
            key = filename.replace('.csv', '')
        df = pd.read_csv(filename_with_dir)
        return df.rename(columns = {config.label_col: key})

    @staticmethod
    def load_directory(directory, mode):
        files = os.listdir(directory)
        keys = list(set([f.split('-')[0] for f in files]))
        # Hard coded threshold
        keys = [k for k in keys if len(k) == 6]

        def load_valid(directory, files, key):
            files_to_load = [f for f in files if f.startswith(key+'-valid')]
            dfs = [pd.read_csv(os.path.join(directory, f)) for f in files_to_load]
            df = pd.concat(dfs)
            return df.rename(columns={config.label_col: key})

        def load_test(directory, files, key):
            files_to_load = [f for f in files if f.startswith(key+'-test-fold')]
            files_to_load.sort()
            df = None
            for i, f in enumerate(files_to_load):
                df0 = pd.read_csv(os.path.join(directory, f))
                df0 = df0.rename(columns={ config.label_col: key + '_' + str(i) })
                if df is None:
                    df = df0
                else:
                    df = df.merge(df0, on=config.id_col)
            return df

        def load_test_sum(directory, files, key):
            files_to_load = [f for f in files if f.startswith(key+'-test-sum')]
            file_to_load = os.path.join(directory, files_to_load[0])
            df = pd.read_csv(file_to_load)
            return df.rename(columns={config.label_col: key})

        def load_key(directory, files, key, mode):
            if mode == 'valid':
                return load_valid(directory, files, key)
            elif mode == 'test-folds':
                return load_test(directory, files, key)
            elif mode == 'test-sum':
                return load_test_sum(directory, files, key)
            else:
                raise Exception('wrong mode')

        df = None

        for key in keys:
            df0 = load_key(directory, files, key, mode)
            if df is None:
                df = df0
            else:
                df = df.merge(df0, on=config.id_col)

        return df.sort_values(by=config.id_col).reset_index(drop=True)
