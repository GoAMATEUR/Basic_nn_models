import torch
import torch.nn as nn


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout_rate=0.1):
        super(MultiheadAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, mask=None):
        # q, k, v: (batch_size, seq_len, d_model)
        # mask: (seq_len, seq_len)
        q = self.w_q(q).view(q.size(0), q.size(1), self.n_head, self.d_head).transpose(1, 2) # (batch_size, n_head, seq_len, d_head)
        k = self.w_k(k).view(k.size(0), k.size(1), self.n_head, self.d_head).transpose(1, 2) # (batch_size, n_head, seq_len, d_head)
        v = self.w_v(v).view(v.size(0), v.size(1), self.n_head, self.d_head).transpose(1, 2) # (batch_size, n_head, seq_len, d_head)
        
        score = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5) # (batch_size, n_head, seq_len, seq_len)
        if mask is not None:
            # apply causal mask
            mask = mask.unsqueeze(0).unsqueeze(1)
            score = score + mask
        score = torch.softmax(score, dim=-1)
        score = self.dropout(score)
        
        x = torch.matmul(score, v) # (batch_size, n_head, seq_len, d_head)
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1, self.d_model) # (batch_size, seq_len, d_model)
        x = self.w_o(x)
       
        return x, score

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, feedforward_dim, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiheadAttention(d_model, n_head, dropout_rate=dropout_rate)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, d_model)
        # self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attention(x, x, x, mask)
        x = self.dropout(x)
        x += residual
        # feedforward
        residual = x
        x = self.norm2(x)
        x = self.feedforward(x)
        x = self.dropout(x)
        x += residual
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_head, n_layer, feedforward_dim, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        # self.d_model = d_model
        # self.n_head = n_head
        # self.feedforward_dim = feedforward_dim
        # self.n_layer = n_layer
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_head, feedforward_dim, dropout_rate)
            for _ in range(n_layer)
        ])

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, d_model)
        for layer in self.layers:
            x = layer(x, mask)
        return x


if __name__ == "__main__":
    # Example usage
    model = TransformerEncoder(d_model=768, n_head=12, n_layer=6, feedforward_dim=4*768)
    x = torch.randn(1, 14*14, 16*16*3)  # (batch_size, seq_len, d_model)
    print(model(x).shape)  # torch.Size([16, 20, 512])

