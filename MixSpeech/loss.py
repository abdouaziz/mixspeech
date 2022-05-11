import torch
import torch.nn as nn
import torch.nn.functional as F




#Cosinus similarity

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Cosine similarity between x1 and x2 along dim.
    Args:
        x1: Tensor of shape [batch_size, d_1, ..., d_k].
        x2: Tensor of shape [batch_size, d_1, ..., d_k].
        dim: The dimension along which to apply the cosine similarity.
        eps: A small value to avoid numerical instability.
    Returns:
        Tensor of shape [batch_size, n_1, ..., n_k].
    """
    dot = torch.matmul(x1, x2.transpose(dim, -1))
    x1_norm = torch.norm(x1, 2, dim=dim, keepdim=True)
    x2_norm = torch.norm(x2, 2, dim=dim, keepdim=True)
    return dot / (x1_norm * x2_norm.transpose(dim, -1) + eps)







if __name__ == '__main__':
    x1 = torch.randn(1, 3, 4)
    x2 = torch.randn(2, 3, 4)
    print(cosine_similarity(x1, x2))
 



