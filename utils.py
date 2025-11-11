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


def format_metadata_readable(obj, indent=0):
    """
    Convert nested dict/list metadata into human-friendly lines:
    key: value
      nested_key: value
    - item1
    - item2
    Returns list of lines.
    """
    lines = []
    space = "  " * indent
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{space}**{k}**:")
                lines.extend(format_metadata_readable(v, indent + 1))
            else:
                lines.append(f"{space}**{k}**: {v}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj, 1):
            if isinstance(item, (dict, list)):
                lines.append(f"{space}- [{i}]")
                lines.extend(format_metadata_readable(item, indent + 1))
            else:
                lines.append(f"{space}- {item}")
    else:
        lines.append(f"{space}{obj}")
    return lines