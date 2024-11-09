import torch
import torch.nn as nn
from torch.functional import F
from .preprocess import EmbedSequence
from .transformer import TransformerEncoder

class ViT(nn.Module):
    def __init__(self, d_model, n_head, n_layers, n_patch, n_class, max_len):
        super(ViT, self).__init__()
        self.n_patch = n_patch
        self.n_class = n_class
        self.max_len = max_len
        self.embed = EmbedSequence(max_len, d_model, n_patch)
        self.encoder = TransformerEncoder(d_model, n_head, n_layers, d_model*4)
        self.head = nn.Linear(d_model, n_class)

    def forward(self, x):
        # x: (batch_size, n_patch, d_patch)
        x = self.embed(x)
        x = self.encoder(x)
        x = x[:, 0] # extract class embedding: (batch_size, d_model)
        x = F.layer_norm(x, x.size()[1:])
        x = self.head(x)
        return x