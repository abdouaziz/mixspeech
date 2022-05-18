from matplotlib.pyplot import cla
import torch
from features import FeaturesEncoder
import torch.nn as nn
from torch.nn import functional as F
import math







class PositionalEncoding(nn.Module):
    """
    Implement the positional encoding (PE) function.
    PE_(pos, 2i)    =  sin(pos / 10000 ** (2i / d_model))
    PE_(pos, 2i+1)  =  cos(pos / 10000 ** (2i / d_model))
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.permute(2,0,1)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)




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
        #residual = q

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
        #output = output + residual

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
    """
    Implement position-wise feed forward layer.
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
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
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Compute self attention
        x_ = x 
        x = self.self_attn(x, x, x, mask=mask)[0]
        
        # Add residual and layer normalization
        x = self.layer_norm(x + x_)
        x = self.dropout(x)
      
        # Compute feed forward
        x_ = x
        x = self.ff(x)

        # Add residual and layer normalization
        x = self.layer_norm(x + x_)
        x = self.dropout(x)


        return x


class TransfomerMixSpeech(nn.Module):
    """ Encoder """
    def __init__(self, d_model, d_ff, n_layers, n_head, dropout=0.1):
        super().__init__()
       #   self.position_enc = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, n_head, dropout=dropout) for _ in range(n_layers)])

    def forward(self, x, mask=None):
       # x = self.position_enc(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class Model(nn.Module):
    def __init__(self, d_model, d_ff, n_layers, n_head, input_chanel ,  dropout=0.1):
        super().__init__()
        self.features = FeaturesEncoder(input_chanel)
        self.position_enc = PositionalEncoding(d_model, dropout=dropout)
        self.transformer = TransfomerMixSpeech(d_model, d_ff, n_layers, n_head, dropout=dropout)
       
    def forward(self, x, mask=None):
        x = self.features(x)  
        x = self.position_enc(x) 
        Y = self.transformer(x, mask=mask)
        
        output = x.mean(dim=1)
        
        return output , Y

        

