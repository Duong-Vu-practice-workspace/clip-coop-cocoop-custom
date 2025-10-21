import clip
import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
from tqdm import tqdm
from config import get_cfg_defaults

TEMPLATES = "a photo of fungi {}"

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
    
class TextEncoder(nn.Module):

    def __init__(self, cfg, clip_model,device):
        super().__init__()
        self.cfg = cfg
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
        self.device = device
    
    def forward(self):
        temp = TEMPLATES
        prompts = [temp.format(c.replace('_', ' ')) for c in self.cfg.DATASET.CLASSNAMES]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        text_features = self.clip_model.encode_text(prompts)
        x = text_features
        return x


class CustomCLIPAdapter(nn.Module):

    def __init__(self, cfg, clip_model,device):
        super().__init__()
        # self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(cfg, clip_model,device)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.adapter = Adapter(512, 4).to(clip_model.dtype)

        # for name, param in self.image_encoder.named_parameters():
        #     if 'adapter' not in name:
        #         param.requires_grad_(False)

        for name, param in self.text_encoder.named_parameters():
            if 'adapter' not in name:
                param.requires_grad_(False)
            
    def forward(self, image_features,ratio = 0.2):
        # image_features = self.image_encoder(image.type(self.dtype))
        x = self.adapter(image_features)
        
        image_features = ratio * x + (1 - ratio) * image_features

        text_features = self.text_encoder()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits