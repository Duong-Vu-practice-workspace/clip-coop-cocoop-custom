import os
import clip
import torch
import os
import json

from my_config import ROOT_DIR
def change_root_dir(path_name):
    parts = path_name.split(os.path.sep)
    new_path = os.path.join(ROOT_DIR, parts[-4], parts[-3], parts[-2], parts[-1])
    return new_path

