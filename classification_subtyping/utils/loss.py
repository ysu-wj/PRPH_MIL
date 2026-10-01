import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def define_loss(args):
    if args.loss == "ce_surv":
        return CrossEntropySurvLoss(alpha=0.0)
    elif args.loss == "nll_surv":
        return NLLSurvLoss(alpha=0.0)
    elif args.loss == "cox_surv":
        return CoxSurvLoss()
    elif args.loss == "nll_surv_kl":
        return [NLLSurvLoss(alpha=0.0), KLLoss()]
    elif args.loss == "nll_surv_mse":
        return [NLLSurvLoss(alpha=0.0), nn.MSELoss()]
    elif args.loss == "nll_surv_l1":
        return [NLLSurvLoss(alpha=0.0), nn.L1Loss()]
    elif args.loss == "nll_surv_cos":
        return [NLLSurvLoss(alpha=0.0), CosineLoss()]
    elif args.loss == "nll_surv_ol":
        return [NLLSurvLoss(alpha=0.0), OrthogonalLoss(gamma=0.5)]
    else:
        raise NotImplementedError

def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)
    c = c.view(batch_size, 1).float()
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    loss = (1 - alpha) * (censored_loss + uncensored_loss) + alpha * uncensored_loss
    return loss.mean()

def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)
    c = c.view(batch_size, 1).float()
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y) + eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = -c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps))
    loss = (1 - alpha) * ce_l + alpha * reg
    return loss.mean()

class CrossEntropySurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha
    def __call__(self, hazards, S, Y, c, alpha=None):
        return ce_loss(hazards, S, Y, c, alpha=self.alpha if alpha is None else alpha)

class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha
    def __call__(self, hazards, S, Y, c, alpha=None):
        return nll_loss(hazards, S, Y, c, alpha=self.alpha if alpha is None else alpha)

class CoxSurvLoss(object):
    def __call__(self, hazards, S, c, **kwargs):
        current_batch_len = len(S)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i, j] = S[j] >= S[i]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        R_mat = torch.FloatTensor(R_mat).to(device)
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * (1 - c))
        return loss_cox

class KLLoss(object):
    def __call__(self, y, y_hat):
        return F.kl_div(y_hat.softmax(dim=-1).log(), y.softmax(dim=-1), reduction="sum")

class CosineLoss(object):
    def __call__(self, y, y_hat):
        return 1 - F.cosine_similarity(y, y_hat, dim=1)

class OrthogonalLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(OrthogonalLoss, self).__init__()
        self.gamma = gamma
    def forward(self, P, P_hat, G, G_hat):
        pos = (1 - torch.abs(F.cosine_similarity(P.detach(), P_hat, dim=1))) + (1 - torch.abs(F.cosine_similarity(G.detach(), G_hat, dim=1)))
        neg = torch.abs(F.cosine_similarity(P, G, dim=1)) + torch.abs(F.cosine_similarity(P.detach(), G_hat, dim=1)) + torch.abs(F.cosine_similarity(G.detach(), P_hat, dim=1))
        return pos + self.gamma * neg