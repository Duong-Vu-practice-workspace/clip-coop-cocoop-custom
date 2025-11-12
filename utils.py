import os
import clip
import torch
import os
import json
import re
from my_config import ROOT_DIR
def change_root_dir(path_name):
    parts = path_name.split(os.path.sep)
    new_path = os.path.join(ROOT_DIR, parts[-4], parts[-3], parts[-2], parts[-1])
    return new_path



def format_metadata_readable(obj, indent=0):
    """
    Convert nested dict/list metadata into human-friendly lines:
    Key names are converted from camelCase / snake_case to Title Case,
    keys are bolded (Markdown) and top-level entries are separated by an extra blank line.
    Returns list of lines suitable for joining with '\n' and passing to st.markdown().
    """
    def humanize_key(k: str) -> str:
        if not isinstance(k, str):
            return str(k)
        # replace underscores, split camelCase, collapse multiple spaces
        s = k.replace("_", " ")
        s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s.title()

    lines = []
    space = "  " * indent
    if isinstance(obj, dict):
        first = True
        for k, v in obj.items():
            title = humanize_key(k)
            if isinstance(v, (dict, list)):
                lines.append(f"{space}**{title}**:")
                lines.extend(format_metadata_readable(v, indent + 1))
            else:
                val = "" if v is None else str(v)
                lines.append(f"{space}**{title}**: {val}")
            # add an extra blank line after top-level keys for readability
            if indent == 0:
                lines.append("")
            first = False
    elif isinstance(obj, list):
        for i, item in enumerate(obj, 1):
            if isinstance(item, (dict, list)):
                lines.append(f"{space}- [{i}]")
                lines.extend(format_metadata_readable(item, indent + 1))
            else:
                lines.append(f"{space}- {item}")
        if indent == 0:
            lines.append("")
    else:
        lines.append(f"{space}{obj}")
    return lines


def _read_json_with_fallback(path):
    """Read JSON with encoding fallbacks and return (obj, error_str)."""
    if not os.path.isfile(path):
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e_utf8:
        try:
            with open(path, "rb") as f:
                raw = f.read()
            for enc in ("utf-8-sig", "utf-8", "latin-1"):
                try:
                    s = raw.decode(enc)
                    return json.loads(s), None
                except Exception:
                    continue
            return None, f"failed to decode JSON file {path}: {e_utf8}"
        except Exception as e:
            return None, str(e)


def _get_class_metadata_for_path(img_path):
    """
    Look for metadata.json in the image's parent directory or grandparent (class folder).
    Returns dict or None. If a parse error occurs, returns dict with key '_metadata_error'.
    """
    try:
        parent = os.path.dirname(img_path)
        candidates = []
        if parent:
            candidates.append(os.path.join(parent, "metadata.json"))
        grandparent = os.path.dirname(parent) if parent else None
        if grandparent:
            candidates.append(os.path.join(grandparent, "metadata.json"))
        for cand in candidates:
            obj, err = _read_json_with_fallback(cand)
            if obj is not None:
                return obj
            if err:
                return {"_metadata_error": err}
    except Exception as e:
        return {"_metadata_error": str(e)}
    return None