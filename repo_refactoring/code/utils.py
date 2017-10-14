import os
import numpy as np

'''
CR-soon jjia: consider generalize this function to more data types, for example, we can change the signature to
save_to_file(data, output_file, overwrite, save_fn)
Then the current save_df_to_file on a pandas dataframe can be achieved through calling

save_to_file(
    df,
    output_file,
    overwrite,
    lambda df, output_file: df.to_csv(output_file, index=False)
)

also I changed overwrite to an optional argument for the simplicity and safety reason
'''
def save_df_to_file(df, output_file, overwrite=False):
    if os.path.isfile(output_file):
        # CR jjia: could check the last modified time to determine whether to rerun this
        if not overwrite:
            raise Exception(
                'try to overwrite an existing file {} when overwrite is False'.format(
                    output_file
                )
            )
        else:
            print "{} already exists, but will overwrite it".format(output_file)

    path, filename = os.path.split(output_file)
    if not os.path.isdir(path):
        os.system('mkdir -p {}'.format(path))
    df.to_csv(output_file, index = False)

def gini_normalized(a, p):
    def gini(actual, pred, cmpcol=0, sortcol=1):
        assert( len(actual) == len(pred) )
        all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
        all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
        totalLosses = all[:,0].sum()
        giniSum = all[:,0].cumsum().sum() / totalLosses
        giniSum -= (len(actual) + 1) / 2.
        return giniSum / len(actual)
    return gini(a, p) / gini(a, a)
