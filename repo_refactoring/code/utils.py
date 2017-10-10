import os

def outputfile_already_exists(filename, create_path_if_not_exist=True):
    if os.path.isfile(filename):
        return True
    elif not create_path_if_not_exist:
        return False
    else:
        path, filename = os.path.split(filename)
        if not os.path.isdir(path):
            os.system('mkdir {}'.format(path))
        return False
