import os
import clip
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from my_config import CLIP_PROMPT_TEMPLATES, device


def _encode_text_with_templates(model, qtext, device):
    texts = [t.format(qtext) for t in CLIP_PROMPT_TEMPLATES]
    toks = clip.tokenize(texts, truncate=True).to(device)
    with torch.no_grad():
        txt_emb = model.encode_text(toks)                     # [B, D]
        txt_emb = F.normalize(txt_emb, dim=-1)
        txt_emb = txt_emb.mean(dim=0, keepdim=True)
        txt_emb = F.normalize(txt_emb, dim=-1).cpu().numpy().astype(np.float32)
    return txt_emb
