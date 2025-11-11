from PIL import Image
import streamlit as st
import torch
import clip
import os
import json
import numpy as np
from peft import PeftModel
from faiss_index import create_faiss_index, load_faiss_index
from my_config import INDEX_FILE, LORA_MODEL_DIR, PROMPTS_FILE, ROOT_DIR, device
from utils import change_root_dir
import torch.nn.functional as F

# Load embeddings and image paths
embeddings = np.load(os.path.join(ROOT_DIR, 'original_embeddings.npy'))
img_paths_aug = np.load(os.path.join(ROOT_DIR, 'original_image_paths.npy'))
img_paths_aug = [change_root_dir(path) for path in img_paths_aug]

# Create / load index
create_faiss_index(embeddings, img_paths_aug, INDEX_FILE)
index, image_paths_index = load_faiss_index(INDEX_FILE)

# Load prompts file (to get image paths and labels if available)
with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
    prompts = json.load(f)

# Normalize prompt paths and build quick lookups
image_paths = [change_root_dir(p.get('img_path') or p.get('image') or p.get('path', '')) for p in prompts]
labels = [p.get('binomial_name') for p in prompts]

PROMPT_BY_PATH = {}
PROMPT_BY_BASENAME = {}
for p in prompts:
    ip = change_root_dir(p.get('img_path') or p.get('image') or p.get('path', ''))
    if ip:
        PROMPT_BY_PATH[ip] = p
        PROMPT_BY_BASENAME[os.path.basename(ip)] = p

def _get_class_metadata_for_path(img_path):
    """
    Try to find a class-level metadata.json in the parent folder of img_path.
    Returns dict or None.
    """
    try:
        parent = os.path.dirname(img_path)
        meta_path = os.path.join(parent, "metadata.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as mf:
                return json.load(mf)
    except Exception:
        return None
    return None

def _get_image_metadata(img_path):
    """
    Try to load a per-image JSON alongside the image (same basename .json).
    """
    try:
        jpath = os.path.splitext(img_path)[0] + ".json"
        if os.path.isfile(jpath):
            with open(jpath, "r", encoding="utf-8") as jf:
                return json.load(jf)
    except Exception:
        return None
    return None

def get_topk_results(query, model, index, image_paths, embeddings_matrix, k=6):
    model.eval()
    if isinstance(query, str) and os.path.exists(query):
        image_path = query
        preprocess = st.session_state.get("preprocess")
        if preprocess is None:
            raise RuntimeError("preprocess is not available in session state")
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(image)
            emb = F.normalize(emb, dim=-1).cpu().numpy().astype(np.float32)
    else:
        toks = clip.tokenize([query], truncate=True).to(device)
        with torch.no_grad():
            emb = model.encode_text(toks)
            emb = F.normalize(emb, dim=-1).cpu().numpy().astype(np.float32)

    D, I = index.search(emb, k)
    indices = [int(x) for x in I[0]]

    embs = np.asarray(embeddings_matrix, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms

    sims = [float(np.dot(emb[0], embs[idx])) for idx in indices]
    paths = [image_paths[idx] for idx in indices]

    return indices, sims, paths

st.set_page_config(page_title="CLIP Demo", layout="wide")
st.title("CLIP Image/Text Retrieval")

@st.cache_resource()
def load_model_from_path():
    base_model, preprocess = clip.load("ViT-B/32", device=device)
    model = PeftModel.from_pretrained(base_model, LORA_MODEL_DIR).to(device)
    model = model.merge_and_unload()
    model.eval()
    return model, preprocess

try:
    with st.spinner("Loading model via torch.load(...)"):
        model_obj, preprocess = load_model_from_path()
        st.session_state["model_obj"] = model_obj
        st.session_state["preprocess"] = preprocess
        st.session_state["model_loaded"] = True
except Exception as e:
    st.session_state["model_loaded"] = False
    st.error(f"Failed to load model: {e}")

# Sidebar controls
st.sidebar.header("Query")
text_query = st.sidebar.text_input("Text query")
uploaded_file = st.sidebar.file_uploader("Upload query image", type=["jpg", "jpeg", "png"])
top_k = st.sidebar.number_input("Top K", min_value=1, max_value=48, value=12, step=1)
combine = st.sidebar.checkbox("Combine text + image", value=False)
search_btn = st.sidebar.button("Search")

# Session state defaults
if "results" not in st.session_state:
    st.session_state["results"] = []
if "feedback" not in st.session_state:
    st.session_state["feedback"] = {}

# When Search pressed
if search_btn:
    try:
        model = st.session_state.get("model_obj") or model_obj
        preprocess = st.session_state.get("preprocess") or preprocess

        sim_text = None
        sim_img = None

        # compute text similarity (if text provided)
        indices, sims, paths = [], [], []
        if text_query:
            indices, sims, paths = get_topk_results(text_query, model, index, image_paths, embeddings_matrix=embeddings, k=top_k)
            st.success(f"Text query found {len(indices)} results.")
            st.success(f"Top text similarity scores: {[round(s, 4) for s in sims]}")
            sim_text = (indices, sims, paths)

        st.session_state["results"] = []

        # Build results from indices, sims and paths returned by get_topk_results
        for rank, (idx, sim, path) in enumerate(zip(indices, sims, paths)):
            meta = {
                "similarity_score": float(sim),
                "img_path": path,
                "source": "combined" if (combine and sim_text is not None and sim_img is not None) else ("text" if sim_text is not None else "image")
            }

            # Attach label if available in prompts/labels
            try:
                if image_paths and labels:
                    try:
                        pos = image_paths.index(path)
                        meta["label"] = labels[pos]
                    except ValueError:
                        meta["label"] = os.path.splitext(os.path.basename(path))[0]
                else:
                    meta["label"] = os.path.splitext(os.path.basename(path))[0]
            except Exception:
                meta["label"] = os.path.splitext(os.path.basename(path))[0]

            # Attach class-level metadata (metadata.json in parent folder)
            class_meta = _get_class_metadata_for_path(path)
            if class_meta is not None:
                meta["class_metadata"] = class_meta

            # Attach per-image metadata (image.json next to image)
            image_meta = _get_image_metadata(path)
            if image_meta is not None:
                meta["image_metadata"] = image_meta

            # Attach prompt / annotation entry if available
            prompt_entry = PROMPT_BY_PATH.get(path)
            if prompt_entry is None:
                prompt_entry = PROMPT_BY_BASENAME.get(os.path.basename(path))
            if prompt_entry is not None:
                meta["prompt_entry"] = prompt_entry
                if "binomial_name" in prompt_entry:
                    meta["label_from_prompts"] = prompt_entry["binomial_name"]
                elif "label" in prompt_entry:
                    meta["label_from_prompts"] = prompt_entry["label"]

            st.session_state["results"].append({
                "idx": int(idx),
                "title": f"Result #{rank + 1} {meta.get('label', '')}",
                "meta": meta
            })

    except Exception as e:
        st.error(f"Failed computing similarity / building results: {e}")

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
                img_path = item["meta"].get("img_path")
                score = item["meta"].get("similarity_score")
                title = f"{item['title']} — score: {score:.3f}" if score is not None else item["title"]
                shown = False
                if img_path:
                    try:
                        img = Image.open(img_path).convert("RGB")
                        img.thumbnail((640, 640))
                        st.image(img, caption=title, width='stretch')
                        shown = True
                    except Exception as e:
                        st.warning(f"Failed to open image {img_path}: {e}")

                st.markdown(f"**{item['title']}**")
                st.caption(f"idx: {item['idx']}")

                with st.expander("View metadata (click to expand)"):
                    class_meta = item["meta"].get("class_metadata")
                    if class_meta is not None:
                        st.markdown("**Class metadata**")
                        st.json(class_meta)

                    image_meta = item["meta"].get("image_metadata")
                    if image_meta is not None:
                        st.markdown("**Image metadata**")
                        st.json(image_meta)

                    prompt_meta = item["meta"].get("prompt_entry")
                    print(prompt_meta.get('text'))
                    if prompt_meta is not None:
                        st.markdown(prompt_meta.get("text"))

                like_key = f"like_{item['idx']}"
                dislike_key = f"dislike_{item['idx']}"
                lc, rc = st.columns([1, 1])
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