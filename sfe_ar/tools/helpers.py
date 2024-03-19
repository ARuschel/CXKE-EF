import datetime
import os

def generate_timestamp():
    return datetime.datetime.now().strftime('%y%m%d%H%M')

def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(os.path.join(file_path))
        print('Creating folder: {}.'.format(file_path))
    else:
        print('Directory {} already exists!'.format(file_path))