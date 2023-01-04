from torchlevy.approx_score import get_approx_score, get_approx_score2, get_approx_score3,generalized_gaussian_score
from torchlevy import LevyStable
import torch
from torchlevy import fitting_gen_gaussian_score

def score_loss(model, sde,
               x0: torch.Tensor,
               t: torch.LongTensor,
               e: torch.Tensor,
               label_mode
               ):
    x_coeff = sde.diffusion_coeff(t)
    sigma = sde.marginal_std(t)

    x_t = x0 * x_coeff[:, None] + e * sigma.view(-1, 1)

    if label_mode =="fDSM" or label_mode=='gaussian':
        score = - 1 / sde.alpha * e
    elif label_mode == "ReELS":
        score =generalized_gaussian_score(e,sde.alpha)

    output = model(x_t, t.float())

    weight = output - score
    loss =  (weight).square().sum(dim=(1)).mean(dim=0)
    return loss