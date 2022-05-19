import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 





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
    
    return nn.CosineSimilarity(dim, eps)(x1, x2)


def loss_fn (output , Y , fx_1 , fx_2 , alpha):
    """Cosine similarity between x1 and x2 along dim.
    Args:
        x1: Tensor of shape [batch_size, d_1, ..., d_k].
        x2: Tensor of shape [batch_size, d_1, ..., d_k].
        dim: The dimension along which to apply the cosine similarity.
        eps: A small value to avoid numerical instability.

    Returns:
        Tensor of shape [batch_size, n_1, ..., n_k].
    """
    Y = Y.permute(1,0,2)
    
    num_1 = alpha * torch.exp(cosine_similarity(output, fx_1))
    num_2 = (1 - alpha) * torch.exp(cosine_similarity(output, fx_2))

   
    denom = torch.exp(np.sum(cosine_similarity(output, i) for i in Y) )
    

  
    return torch.div((np.sum(num_1 , num_2)), denom).mean()






""" 

if __name__ == '__main__':

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    x1 = torch.randn(10, 10, 1000)
    x2 = torch.randn(10, 768)
    print(cosine_similarity(x1, x2))
    print(cosine_similarity(x1, x2).shape) """