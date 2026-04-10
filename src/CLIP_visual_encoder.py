import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPEncoderWrapper(nn.Module):
    def __init__(self, latent_dim, *args, **kwargs):
        super().__init__()

        from transformers import CLIPModel
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        for p in self.clip.parameters():
            p.requires_grad = False

        hidden_dim = self.clip.config.vision_config.hidden_size
        self.projection = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        mean = torch.tensor([0.481, 0.457, 0.408], device=x.device).view(1,3,1,1)
        std  = torch.tensor([0.268, 0.261, 0.275], device=x.device).view(1,3,1,1)
        x = (x - mean) / std

        outputs = self.clip.vision_model(pixel_values=x)
        features = outputs.pooler_output

        features = features / features.norm(dim=-1, keepdim=True)

        z = self.projection(features)
        return z