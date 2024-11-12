import torch
import torch.nn as nn
import numpy as np
import math

class TimePosEmb(nn.Module):
    def __init__(self, dim):
        super(TimePosEmb, self).__init__()
        self.dim = dim
        self.time_emb = nn.Embedding(dim, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, dim, 1, 1))

    def forward(self, t):
        t = t.long()
        time_emb = self.time_emb(t)
        return time_emb + self.pos_emb


class DenoisingMLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, device, t_dim):
        super(DenoisingMLP, self).__init__()

        self.device = device
        self.t_dim = t_dim
        self.a_dim = action_dim
        
        self.time_emb = nn.Sequential(
            TimePosEmb(t_dim),
            nn.Linear(t_dim, t_dim*2),
            nn.Mish(),
            nn.Linear(t_dim*2, t_dim),
        )
        

    def forward(self, x, t):
        t = torch.ones_like(x) * t
        x = torch.cat([x, t], dim=-1)
        return self.net(x)

if __name__ == "__main__":
    pass