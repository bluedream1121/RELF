import torch
import torch.nn.functional as F

def mean_squared_error(A, B, p=2):
    loss = (A - B).pow(p).sum(1)
    return loss

def cross_entropy(A, B, eps=1e-6):
    loss = -(A * torch.log(B + eps)).sum(1)
    return loss


## reference : https://github.com/sthalles/SimCLR
def info_nce_contrastive_loss(desc1, desc2, temperature=0.07):

    # features = F.normalize(features, dim=1)
    B, K, C = desc1.shape
    desc1 = desc1.reshape(B*K, C)
    desc2 = desc2.reshape(B*K, C)

    similarity_matrix = torch.matmul(desc1, desc2.T)
    mask = torch.eye(similarity_matrix.shape[0], dtype=torch.bool).cuda()

    assert mask.shape == similarity_matrix.shape
    positives = similarity_matrix[mask].view(B*K, -1)
    negatives = similarity_matrix[~mask].view(B*K, -1)

    logits =torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(B*K).long().to(logits.device)  ## positives are the 0-th

    # logits == torch.softmax(logits / temperature, dim=1)
    logits = logits / temperature  
    loss = F.cross_entropy(logits, labels)

    return loss

def l2_loss(desc1, desc2):
    """ desc: [B, K, CG], """
    loss = (desc1 - desc2).pow(2)
    return loss

def l1_loss(desc1, desc2):
    loss = torch.abs(desc1 - desc2)
    return loss

def triplet_loss(desc1, desc2, margin=0.15):
    B, K, N = desc1.shape

    desc1 = desc1.reshape(B*K, N)
    desc2 = desc2.reshape(B*K, N)

    dist_matrix = torch.cdist(desc1, desc2, p=2)
    pos = torch.diagonal(dist_matrix, 0)
    neg = torch.max(dist_matrix, dim=1)[0]

    loss = torch.max(pos - neg + margin, torch.tensor([0.0]).cuda())

    return loss

def triplet_loss_v2(desc1, desc2, margin=0.15):
    B, K, N = desc1.shape

    desc1 = desc1.reshape(B*K, N)
    desc2 = desc2.reshape(B*K, N)

    corr = torch.matmul(desc1, desc2.t()) ## cosine similarity 
    # corr_dist = 1 / corr + 1e-6
    pos = torch.diagonal(corr, 0)
    neg = torch.min(corr, dim=1)[0]

    loss = torch.max( -pos +  neg + margin, torch.tensor([0.0]).cuda())

    return loss


def orientation_shift_loss(desc1, desc2, angle, G):
    """ desc: [B, K, CG],
        angle: [B] degree """

    assert desc1.shape == desc2.shape

    B, K, CG = desc1.shape
    desc1 = desc1.reshape(B,K,-1,G)
    desc2 = desc2.reshape(B,K,-1,G)

    ## shift the first dimension
    GT_shift = torch.round(G*angle/360).int()

    loss_batch = [] ## TODO: batchified torch roll (shifting vector)?
    for b, shift in enumerate(GT_shift):
        desc1_shift = torch.roll(desc1[b, :, 0, :], shifts=int(shift), dims=1)
        desc2_shift = desc2[b, :, 0, :]

        desc1_shift = torch.softmax(desc1_shift, dim=1)
        desc2_shift = torch.softmax(desc2_shift, dim=1)        

        # desc1_shift = torch.sigmoid(desc1_shift)
        # desc2_shift = torch.sigmoid(desc2_shift)  

        ## Gaussian smooth      
        # desc1_shift = gaussian(desc1_shift)
        # desc2_shift = gaussian(desc2_shift)

        loss = cross_entropy(desc1_shift, desc2_shift) 
        loss = loss.sum(0) / G / K
        loss_batch.append(loss)
    
    loss_batch = torch.stack(loss_batch)

    return loss_batch


