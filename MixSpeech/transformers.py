import torch
import torch.nn as nn
from torch.nn import functional as F
 

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        
    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: [batch_size, len_q, d_k]
            k: [batch_size, len_k, d_k]
            v: [batch_size, len_k, d_v]
            mask: [batch_size, len_q, len_k]
        Returns:
            context: [batch_size, len_q, d_v]
            attn: [batch_size, len_q, len_k]
        """
        attn = torch.matmul(q / self.temperature, k.transpose(2, 1))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))

        context = torch.matmul(attn, v)

        return context, attn




class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        self.fc = nn.Linear(n_head * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: [batch_size, len_q, d_model]
            k: [batch_size, len_k, d_model]
            v: [batch_size, len_v, d_model]
            mask: [batch_size, len_q, len_k]
        Returns:
            context: [batch_size, len_q, d_model]
            attn: [batch_size, n_head, len_q, len_k]
        """
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head
        residual = q

        batch_size, len_q, d_model = q.size()
        batch_size, len_k, d_model = k.size()
        batch_size, len_v, d_model = v.size()

        q = self.w_qs(q).view(batch_size, len_q, n_head, d_k)
        k = self.w_ks(k).view(batch_size, len_k, n_head, d_k)
        v = self.w_vs(v).view(batch_size, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        
        output, attn = ScaledDotProductAttention(temperature=d_k)(q, k, v, mask=mask)
        output = output.view(n_head, batch_size, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, len_q, -1)  # b x lq x (n*dv)
        output = self.dropout(self.fc(output))
        output = output + residual

        return output, attn




class LayerNorm(nn.Module):
    """ LayerNorm """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta



class PositionwiseFeedForward(nn.Module):
    """ PositionwiseFeedForward """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))




class EncoderLayer(nn.Module):
    """ EncoderLayer """
    def __init__(self, d_model, d_ff, n_head, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_head, d_model, d_model // n_head, d_model // n_head, dropout=dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.self_attn(x, x, x, mask=mask)[0]
        x = self.dropout(x)
        x = self.ff(x)
        x = self.dropout(x)
        return x



class Encoder(nn.Module):
    """ Encoder """
    def __init__(self, d_model, d_ff, n_layers, n_head, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, n_head, dropout=dropout) for _ in range(n_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


if __name__ == '__main__':
    encoder = Encoder(d_model=512, d_ff=2048, n_layers=6, n_head=8, dropout=0.1)
    x = torch.randn(1, 64, 512)
    y = encoder(x)
    print(y)
