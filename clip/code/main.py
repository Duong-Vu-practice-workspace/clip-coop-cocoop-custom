import io
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import torch
import clip
import os
import time
import json
from config import get_cfg_defaults
from torchvision.datasets import ImageFolder
from CustomCLIPCoOp import CustomCLIPCoOp

model_path = '/workspaces/clip-coop-cocoop-custom/clip/code/model_epoch_30.pt'
image_embed_path = '/workspaces/clip-coop-cocoop-custom/clip/code/vit_image_embs.pt'
st.set_page_config(page_title="CLIP Demo", layout="wide")
st.title("CLIP Image/Text Retrieval")

cfg = get_cfg_defaults()
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ViT-B/16"
DATA_DIR = "/workspaces/clip-coop-cocoop-custom/Dataset"  # change to your path mounted or uploaded

class Clip:
    def __init__(self):
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        try:
            self.model.to(device)
        except Exception:
            pass
        self.train_ds = ImageFolder(DATA_DIR, transform=self.preprocess)
        self.image_embeds = torch.load(image_embed_path, map_location=device).to(device)
        self.image_embeds = self.image_embeds / (self.image_embeds.norm(dim=-1, keepdim=True) + 1e-10)
@st.cache_resource()
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
    model = Clip()
    return model
    

try:
    with st.spinner("Loading model via torch.load(...)"):
        model_obj = load_model_from_path(model_path)
        st.session_state["model_obj"] = model_obj
        st.session_state["model_loaded"] = True
except Exception as e:
    st.session_state["model_loaded"] = False
    st.error(f"Failed to load model: {e}")

@st.cache_data()
def get_text_embedding(text: str, model_key: str = None):
    """
    Return normalized text embedding (list of floats).
    Accepts optional second argument model_key for compatibility with callers
    that pass (text, model_path).
    Uses CLIP model wrapper stored in st.session_state['model_obj'].
    """
    if not text:
        raise ValueError("text must be non-empty")

    model_wrapper = st.session_state.get("model_obj")
    if model_wrapper is None:
        raise RuntimeError("Model not loaded (st.session_state['model_obj'] missing)")

    clip_model = getattr(model_wrapper, "model", model_wrapper)
    if not hasattr(clip_model, "encode_text"):
        raise RuntimeError("Loaded object does not expose encode_text; expected Clip wrapper or CLIP model")
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        txt_emb = clip_model.encode_text(tokens).float()
        txt_emb = txt_emb / (txt_emb.norm(dim=-1, keepdim=True) + 1e-10)

    return txt_emb[0].cpu().numpy()

@st.cache_data()
def get_image_embedding(image_bytes: bytes):
    """
    Compute normalized CLIP image embedding from raw image bytes.
    Returns embedding as Python list (floats).
    """
    if not image_bytes:
        raise ValueError("no image bytes provided")

    # ensure model loaded
    model_wrapper = st.session_state.get("model_obj")
    if model_wrapper is None:
        raise RuntimeError("Model not loaded (st.session_state['model_obj'] missing)")

    clip_model = getattr(model_wrapper, "model", model_wrapper)
    preprocess = getattr(model_wrapper, "preprocess", None)
    if preprocess is None:
        raise RuntimeError("Loaded model wrapper does not provide preprocess function")

    if not hasattr(clip_model, "encode_image"):
        raise RuntimeError("Loaded object does not expose encode_image; expected CLIP model")

    # open image from bytes (PIL)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # preprocess -> tensor batch
    img_t = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        img_emb = clip_model.encode_image(img_t).float()
        img_emb = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-10)

    return img_emb[0].cpu().numpy()

def cal_similarity_text_with_pre_embed_image(text: str):
    text_embeds = get_text_embedding(text)  # numpy array (cpu)
    model_wrapper = st.session_state.get("model_obj") or model_obj
    image_embeds = model_wrapper.image_embeds  # tensor on device, possibly half

    # convert text embedding to a tensor with the same dtype & device as image_embeds
    text_t = torch.tensor(text_embeds, device=image_embeds.device, dtype=image_embeds.dtype)
    text_t = text_t.unsqueeze(-1)  # (D,1)

    # matrix multiply: (N, D) @ (D,1) -> (N,1) then squeeze
    similarity_scores = (image_embeds @ text_t).squeeze(-1)

    # ensure float32 on CPU for downstream use if desired
    return similarity_scores

# Sidebar: input controls
st.sidebar.header("Query")
text_query = st.sidebar.text_input("Text query")
uploaded_file = st.sidebar.file_uploader("Upload query image", type=["jpg", "jpeg", "png"])
top_k = st.sidebar.number_input("Top K", min_value=1, max_value=48, value=12, step=1)
combine = st.sidebar.checkbox("Combine text + image", value=False)
search_btn = st.sidebar.button("Search")




# Simple session state to hold placeholder results + feedback
if "results" not in st.session_state:
    st.session_state["results"] = []  # list of dict {idx, title, meta}
if "feedback" not in st.session_state:
    st.session_state["feedback"] = {}  # idx -> vote


def _make_placeholder_image(text: str, size=(320, 240)):
    img = Image.new("RGB", size, color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
    except Exception:
        try:
            # textlength gives width; derive height from font metrics if available
            w = int(draw.textlength(text, font=font))
            if font is not None and hasattr(font, "getmetrics"):
                ascent, descent = font.getmetrics()
                h = ascent + descent
            else:
                h = 11
        except Exception:
            # final fallback: estimate
            w = len(text) * 6
            h = 11

    draw.text(((size[0] - w) / 2, (size[1] - h) / 2), text, fill=(60, 60, 60), font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

# When Search pressed, create placeholder results (backend will replace)
if search_btn:
    try:
        similarity_scores = cal_similarity_text_with_pre_embed_image(text_query)
        k = min(int(top_k), similarity_scores.shape[0])
        top_k_indices = torch.topk(similarity_scores, k=k).indices.tolist()

        st.session_state["results"] = []
        model_wrapper = st.session_state.get("model_obj") or model_obj
        for rank, idx in enumerate(top_k_indices):
            # try to resolve image path and class from the ImageFolder
            meta: dict = {}
            try:
                samples = model_wrapper.train_ds.samples  # list of (path, class_idx)
                if 0 <= idx < len(samples):
                    img_path, cls_idx = samples[idx]
                    cls_name = model_wrapper.train_ds.classes[cls_idx]
                    meta["class"] = cls_name
                    meta["img_path"] = img_path
                    # attempt to load per-class metadata.json from DATA_DIR/<class>/metadata.json
                    meta_path = os.path.join(DATA_DIR, cls_name, "metadata.json")
                    if os.path.isfile(meta_path):
                        try:
                            with open(meta_path, "r", encoding="utf-8") as mf:
                                class_meta = json.load(mf)
                                meta["class_metadata"] = class_meta
                        except Exception as e:
                            meta["class_metadata_error"] = f"failed to load {meta_path}: {e}"
                else:
                    meta["note"] = "index out of range for train_ds.samples"
            except Exception as e:
                meta["train_ds_error"] = str(e)

            score_val = float(similarity_scores[idx].detach().cpu().item())
            meta["similarity_score"] = score_val
            meta["source"] = "calculated"

            st.session_state["results"].append({
                "idx": idx,
                "title": f"Result #{rank + 1} (img idx {idx})",
                "meta": meta
            })
    except Exception as e:
        st.error(f"Failed computing similarity / building results: {e}")

# Main area: show query summary and results grid
col_left, col_right = st.columns([6, 1])

with col_left:
    st.subheader("Query")
    if text_query:
        st.write("Text:", text_query)
    if uploaded_file:
        st.image(uploaded_file, caption="Query image", width=240)
    if not (text_query or uploaded_file):
        st.info("Enter a text query or upload an image, then press Search.")

    st.markdown("---")
    st.subheader("Results")
    results = st.session_state["results"]
    if not results:
        st.write("No results to show (backend not connected).")
    else:
        cols = st.columns(2)
        for i, item in enumerate(results):
            c = cols[i % 2]
            with c:
                # show image placeholder (or real thumbnail if you want)
                ph = _make_placeholder_image(item["title"])
                st.image(ph, use_container_width=True)
                st.markdown(f"**{item['title']}**")
                st.caption(f"idx: {item['idx']}")

                # show a "View metadata" button that expands inline when clicked
                view_key = f"view_meta_{item['idx']}"
                if st.button("View metadata", key=view_key):
                    st.session_state["view_meta"] = item["meta"]

                # if metadata for this item is the currently selected, show it prominently
                if st.session_state.get("view_meta") and st.session_state["view_meta"].get("img_path", None) == item["meta"].get("img_path", None):
                    st.markdown("**Class metadata**")
                    st.json(st.session_state["view_meta"])

                # Like / Dislike buttons store in session state
                like_key = f"like_{item['idx']}"
                dislike_key = f"dislike_{item['idx']}"
                lc, rc = st.columns([1,1])
                with lc:
                    if st.button("Like", key=like_key):
                        st.session_state["feedback"][str(item["idx"])] = 1
                        st.success("Liked")
                with rc:
                    if st.button("Dislike", key=dislike_key):
                        st.session_state["feedback"][str(item["idx"])] = -1
                        st.warning("Disliked")

with col_right:
    st.subheader("Session info")
    st.write(f"Top K selected: {top_k}")
    st.write("Combine text+image:", combine)
    st.markdown("**Feedback (session)**")
    st.json(st.session_state["feedback"])