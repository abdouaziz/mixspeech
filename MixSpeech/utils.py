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






if __name__ == '__main__':

    attention = ScaledDotProductAttention(temperature=1.0)

    q = torch.randn(2, 3, 4)
    k = torch.randn(2, 3, 4)
    v = torch.randn(2, 3, 4)
    mask = torch.randn(2, 3, 3)

    context, attn = attention(q, k, v, mask)