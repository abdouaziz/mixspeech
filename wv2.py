import torch


#constrastive loss function
def contrastive_loss(y_pred, y_true , kappa ):
    """
    Compute the contrastive loss between y_pred and y_true
    """
    # y_pred is the output of the model
    # y_true is the ground truth
    # kappa is the margin
    # y_pred is a tensor of shape (N, 2)
    # y_true is a tensor of shape (N,)
    # kappa is a scalar
    # N is the number of samples
    # Compute the distance between y_pred and y_true
    # dist is a tensor of shape (N, )
    dist = torch.sqrt(torch.sum((y_pred - y_true)**2, dim=1))
    # Compute the loss
    # loss is a scalar
    loss = torch.mean(torch.max(torch.zeros(dist.shape), kappa - dist))
    return loss