
import os

def check_dir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)
