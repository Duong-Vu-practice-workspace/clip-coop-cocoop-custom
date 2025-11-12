from PIL import Image
import streamlit as st
import torch
import clip
import os
import json
import numpy as np
from peft import PeftModel
from faiss_index import create_faiss_index, load_faiss_index
from my_config import INDEX_FILE, LORA_MODEL_DIR, LORA_MODEL_DIR_DAI, PROMPTS_FILE, ROOT_DIR, INDEX_FILE_DAI, device
from utils import change_root_dir, _get_class_metadata_for_path
import torch.nn.functional as F
import io
from utils import format_metadata_readable

# Load embeddings and image paths (these are the indexed images)
embeddings = np.load(os.path.join(ROOT_DIR, 'original_embeddings.npy'))
img_paths_aug = np.load(os.path.join(ROOT_DIR, 'original_image_paths.npy'))
img_paths_aug = [change_root_dir(path) for path in img_paths_aug]

index = None
image_paths_index = None

# Load prompts file
_prompts = []
try:
    with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
        _prompts = json.load(f)
except Exception:
    _prompts = []

prompt_image_paths = [change_root_dir(p.get('img_path')) for p in _prompts]
prompt_labels = [p.get('binomial_name') for p in _prompts]




def get_topk_results(query, model, index, image_paths, embeddings_matrix, k=6):
    model.eval()
    if not isinstance(query, str):
      image = preprocess(query).unsqueeze(0).to(device)
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
def load_model_by_name(model_name: str):
    """
    Load base CLIP and optionally apply LoRA weights. Cached by model_name.
    Returns (model_or_wrapper_or_None, preprocess).
    """
    base_model, preprocess = clip.load("ViT-B/32", device=device)
    model = None
    if model_name == "CLIP + LoRA (Đ)":
        model = PeftModel.from_pretrained(base_model, LORA_MODEL_DIR_DAI)
    elif model_name == "CLIP + LoRA (P)":
        model = PeftModel.from_pretrained(base_model, LORA_MODEL_DIR)
    elif model_name == "CLIP + reranking":
        # use base CLIP without LoRA
        model = base_model
    else:
        # unknown model name -> return base
        model = base_model

    if model is not None:
        try:
            model = model.to(device)
        except Exception:
            pass
        # only LoRA wrappers have merge_and_unload; guard it
        try:
            model = model.merge_and_unload()
        except Exception:
            pass
        model.eval()

    return model, preprocess

# Sidebar controls
st.sidebar.header("Query")
text_query = st.sidebar.text_input("Text query")
uploaded_file = st.sidebar.file_uploader("Upload query image", type=["jpg", "jpeg", "png"])
top_k = st.sidebar.number_input("Top K", min_value=1, max_value=48, value=12, step=1)
search_btn = st.sidebar.button("Search")
# Add combobox (selectbox) for base CLIP model selection
model_options = ["CLIP + LoRA (Đ)", "CLIP + LoRA (P)", "CLIP + reranking"]
selected_model = st.sidebar.selectbox("Model", model_options, index=0)
st.session_state["selected_model"] = selected_model
# Create / load index
create_faiss_index(embeddings, img_paths_aug, INDEX_FILE_DAI)
create_faiss_index(embeddings, img_paths_aug, INDEX_FILE)
index, image_paths_index = load_faiss_index(INDEX_FILE)
prev_selected = st.session_state.get("_selected_model_cached")
if prev_selected != selected_model:
    # clear previous model keys to avoid stale references
    st.session_state.pop("model_obj", None)
    st.session_state.pop("preprocess", None)
    st.session_state["model_loaded"] = False

    try:
        with st.spinner(f"Loading model {selected_model} ..."):
            model_obj, preprocess = load_model_by_name(selected_model)
            st.session_state["model_obj"] = model_obj
            st.session_state["preprocess"] = preprocess
            st.session_state["model_loaded"] = model_obj is not None
            st.session_state["_selected_model_cached"] = selected_model
    except Exception as e:
        st.session_state["model_loaded"] = False
        st.error(f"Failed to load model {selected_model}: {e}")

# load appropriate FAISS index for the selected model (choose once)
if selected_model == "CLIP + LoRA (Đ)":
    index, image_paths_index = load_faiss_index(INDEX_FILE_DAI)
elif selected_model in ["CLIP + LoRA (P)"]:
    index, image_paths_index = load_faiss_index(INDEX_FILE)


# ensure local references used below come from session_state
model = st.session_state.get("model_obj")
preprocess = st.session_state.get("preprocess")
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
        img_loaded = None
        if uploaded_file is not None:
            img_bytes = uploaded_file.getvalue()
            img_loaded = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            
        query = text_query if text_query else img_loaded
        indices, sims, paths = get_topk_results(query, model, index, image_paths_index, embeddings, k=top_k)

        st.session_state["results"] = []

        # Build results from indices, sims and paths returned by get_topk_results
        for rank, (idx, sim, path) in enumerate(zip(indices, sims, paths)):
            meta = {
                "similarity_score": float(sim),
                "img_path": path
            }

            class_meta = _get_class_metadata_for_path(path)
            if class_meta and isinstance(class_meta, dict):
                meta["class_metadata"] = class_meta
                for key in ("binomial_name", "scientificName", "label", "name"):
                    if key in class_meta:
                        meta["label"] = class_meta[key]
                        break
            elif class_meta and isinstance(class_meta, str):
                meta["class_metadata_error"] = class_meta
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
                        st.markdown("\n\n".join(format_metadata_readable(class_meta)))
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
    st.markdown("**Feedback (session)**")
    st.json(st.session_state["feedback"])