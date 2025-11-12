import os
import clip
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from my_config import device
def get_topk_results(query, model, index, image_paths, embeddings_matrix, k=6):
    model.eval()
    if isinstance(query, str) and os.path.exists(query):
      image_path = query
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