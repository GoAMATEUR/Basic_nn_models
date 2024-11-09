import torch
import torch.nn as nn


class EmbedSequence(nn.Module):
    def __init__(self, max_len, d_model, d_patch):
        super(EmbedSequence, self).__init__()
        self.pe = nn.Parameter(torch.zeros(max_len+1, d_model), requires_grad=True) # positional encoding
        self.patch_to_embedding = nn.Linear(d_patch, d_model) # linear layer to embed patches
        self.cls_embedding = nn.Parameter(torch.zeros(d_model), requires_grad=True) # learnable class embedding

    def forward(self, x):
        # x: Image patch seq: (batch_size, n_patch, d_patch). d_patch = 16 * 16 * 3
        x = self.patch_to_embedding(x)  # (batch_size, n_patch, d_model)
        batch_size, n_patch, d_model = x.size()
        cls_embedding = self.cls_embedding.expand(batch_size, 1, d_model) # (batch_size, 1, d_model)
        x = torch.cat((cls_embedding, x), dim=1)
        x += self.pe[:n_patch+1]
        return x
        
        