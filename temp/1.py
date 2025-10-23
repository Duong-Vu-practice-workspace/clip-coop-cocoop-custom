# ...existing code...
import os
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "clip", "checkpoints")

@st.cache(allow_output_mutation=True)
def load_model_from_path(path: str):
    """
    Load a saved PyTorch object with torch.load(path).
    Returns whatever was saved (module, state_dict, dict, ...).
    """
    if not path:
        raise FileNotFoundError("No model path provided")
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")

# Sidebar: model path input (no model choice UI)
pt_candidates = []
if os.path.isdir(CHECKPOINT_DIR):
    pt_candidates = [os.path.join(CHECKPOINT_DIR, f) for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")]
default_model_path = pt_candidates[0] if pt_candidates else ""

model_path = st.sidebar.text_input("Model file path (torch.load)", value=default_model_path)
load_model_btn = st.sidebar.button("Load model (torch.load)")

if "model_loaded" not in st.session_state:
    st.session_state["model_loaded"] = False

if load_model_btn:
    try:
        with st.spinner("Loading model via torch.load(...)"):
            model_obj = load_model_from_path(model_path)
            st.session_state["model_obj"] = model_obj
            st.session_state["model_loaded"] = True
            st.success(f"Loaded model object from {model_path}")
    except Exception as e:
        st.session_state["model_loaded"] = False
        st.error(f"Failed to load model: {e}")

# show status
if st.session_state.get("model_loaded"):
    st.sidebar.write("Model loaded")
else:
    st.sidebar.write("Model not loaded")
# ...existing code...
```# filepath: /home/duongvct/Documents/workspace/pycharm/clip-coop-cocoop-custom/ui/main.py
# ...existing code...
import os
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "clip", "checkpoints")

@st.cache(allow_output_mutation=True)
def load_model_from_path(path: str):
    """
    Load a saved PyTorch object with torch.load(path).
    Returns whatever was saved (module, state_dict, dict, ...).
    """
    if not path:
        raise FileNotFoundError("No model path provided")
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")

# Sidebar: model path input (no model choice UI)
pt_candidates = []
if os.path.isdir(CHECKPOINT_DIR):
    pt_candidates = [os.path.join(CHECKPOINT_DIR, f) for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")]
default_model_path = pt_candidates[0] if pt_candidates else ""

model_path = st.sidebar.text_input("Model file path (torch.load)", value=default_model_path)
load_model_btn = st.sidebar.button("Load model (torch.load)")

if "model_loaded" not in st.session_state:
    st.session_state["model_loaded"] = False

if load_model_btn:
    try:
        with st.spinner("Loading model via torch.load(...)"):
            model_obj = load_model_from_path(model_path)
            st.session_state["model_obj"] = model_obj
            st.session_state["model_loaded"] = True
            st.success(f"Loaded model object from {model_path}")
    except Exception as e:
        st.session_state["model_loaded"] = False
        st.error(f"Failed to load model: {e}")

# show status
if st.session_state.get("model_loaded"):
    st.sidebar.write("Model loaded")
else:
    st.sidebar.write("Model not loaded")
# ...existing code...