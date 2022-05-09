import torch
import torch.nn as nn
import torch.nn.functional as F



#Contrastive loss function
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                        (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

    
#Cosine similarity loss function
class CosineSimilarityLoss(nn.Module):
    """
    Cosine similarity loss function.
    Based on: https://arxiv.org/pdf/1801.04381.pdf
    """
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, output1, output2, label):
        cosine_similarity = F.cosine_similarity(output1, output2)
        loss_cosine_similarity = torch.mean((1-label) * torch.pow(cosine_similarity, 2) +
                                                (label) * torch.pow(torch.clamp(1 - cosine_similarity, min=0.0), 2))
        return loss_cosine_similarity


