import torch

def shift_topk_candidate(kpts, desc, B, K, G, topk=1):
    shifts= torch.topk(desc[:,:,0,:], k= topk, dim=2)[1]
    desc = desc.reshape(B*K, -1, G)
    shifts = shifts.reshape(B*K, -1)

    kpts_update = kpts.repeat(1, topk, 1) ## [B, topk*K, 2] 
    desc_update = []
    for shift in shifts.t():
        for d, s in zip(desc, shift):
            desc_update.append(torch.roll(d, shifts=-int(s), dims=-1)) ## reverse shift

    desc_update = torch.stack(desc_update) ## [topk*B*K, C, G]
    desc_update = desc_update.reshape(topk,B,K,-1,G).transpose(0,1) ## [B*topk*K, C, G]   
    desc_update = desc_update.reshape(B, topk*K,-1)
    return kpts_update, desc_update

def shift_ratio_candidate(kpts, desc, B, K, CG, G, ratio=1.0):
    ## Warning: ratio sampling make different number of keypoints in an image.
    assert desc.min() >= 0
    value, _ = torch.max(desc[:, :, 0, :], dim=2)
    ratio_tensor = (desc[:, :, 0, :] / value.unsqueeze(-1))  ## obtain ratio
    ratio_mask = ratio_tensor >= ratio

    ### un-batchfying because the number of keypoints are different.
    kpts_update = []
    desc_update = []
    for _kpts, _desc, _ratio_mask in zip(kpts, desc, ratio_mask):
        kpts_update_iter = []
        desc_update_iter = []
        for k, d, r in zip(_kpts, _desc, _ratio_mask):
            shifts = r.nonzero().reshape(-1)
            for s in shifts:
                kpts_update_iter.append(k)
                desc_update_iter.append(torch.roll(d, shifts=-int(s), dims=-1))
            
        kpts_update.append(torch.stack(kpts_update_iter))
        desc_update.append(torch.stack(desc_update_iter).reshape(-1, CG))
    return kpts_update, desc_update


def shifting(desc, shift, B, K, G):
    desc = desc.reshape(B*K, -1, G)
    shift = shift.reshape(B*K)
    
    desc_update = []
    for d, s in zip(desc, shift):
        desc_update.append(torch.roll(d, shifts=-int(s), dims=-1)) ## reverse shift
    desc_update = torch.stack(desc_update).reshape(B,K,-1)
    return desc_update

