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
    dot_product = torch.sum(x1 * x2, dim)
    x1_norm = torch.norm(x1, 2, dim)
    x2_norm = torch.norm(x2, 2, dim)
    return dot_product / (x1_norm * x2_norm + eps)


def loss_similarity(x1, x2):
    """Cosine similarity between x1 and x2 along dim.
    Args:
        x1: Tensor of shape [batch_size, d_1, ..., d_k].
        x2: Tensor of shape [batch_size, d_1, ..., d_k].
        mask: Tensor of shape [batch_size, d_1, ..., d_k].
        eps: A small value to avoid numerical instability.
    Returns:

    """
    num = cosine_similarity(x1, x2)
    print("the shape of num is: ", num.shape)
    som = []
    denom = torch.add(cosine_similarity(x1, i) for i in x2[0])
    print("the shape of denom is: ", denom)

    return num , denom





if __name__ == '__main__':

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    x1 = torch.randn(2, 3, 4)
    x2 = torch.randn(2, 3, 4)
    #print(cosine_similarity(x1, x2))
    print(loss_similarity(x1, x2))