import os

def save_df_to_file(df, output_file, overwrite):
    if os.path.isfile(output_file):
        print "Output file exists."
        # CR jjia: could check the last modified time to determine whether to rerun this
        if not overwrite:
            print "Not overwrite."
            # raise
            return
        else:
            print "Will overwirte."
    path, filename = os.path.split(output_file)
    if not os.path.isdir(path):
        os.system('mkdir -p {}'.format(path))
    df.to_csv(output_file, index = False)
