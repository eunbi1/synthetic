import torch
from util import VPSDE, calculate_fid, dict2namespace
from model import Model, ScoreNetwork
from tqdm.notebook import tqdm, trange
from data import get_data

import tqdm
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from torchlevy import LevyStable
from loss import score_loss

from torch.utils.data import DataLoader, TensorDataset


device = "cuda" if torch.cuda.is_available() else "cpu"

def train(alpha, train_data, noise_mode, label_mode,  max_iter, dimension=2, condition=False):
    """
    noise mode: isotropic, independent
    label_model: fDSM, ReLES
    """
   


    sde = VPSDE(alpha, beta_min=0.1, beta_max=20,  schedule='cosine')
    levy = LevyStable()

    score_model = ScoreNetwork(alpha, dimension=dimension, condition=condition)
    score_model.to(device)
    optimizer = torch.optim.Adam(score_model.parameters(), amsgrad=True)
    score_model.train()

    L = []

    dataset = TensorDataset(train_data.float())
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True,
                            num_workers=0, generator=torch.Generator(device='cuda'))

    for epoch in tqdm.tqdm(range(max_iter)):
        avg_loss = 0.
        num_items = 0

        for i, data in enumerate(dataloader):
        
            x = data[0][:,:-1].to(device)
            y = data[0][:,-1].to(device)



            n = x.size(0)
            if noise_mode == "isotropic":
                e = levy.sample(alpha, 0, size=x.shape, is_isotropic=True, clamp=10).to(device)
            elif noise_mode == "independent":
                e = torch.clamp(levy.sample(alpha, 0, size=x.shape, is_isotropic=False),-10,10).to(device)
            elif noise_mode == 'gaussian':
                e = torch.randn(size = x.shape).to(device)
            t = torch.rand(x.shape[0]).to(device)*(sde.T-0.00001)+0.00001
            if condition == False:
                y= None

            loss = score_loss(score_model, sde, x, y, t, e, label_mode)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(score_model.parameters(), 1.0)
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        L.append(avg_loss / num_items)
        #print('Average Loss: {:5f}'.format(avg_loss / num_items))
        # torch.save(score_model.state_dict(), 'ckpt.pth')
    name = str(dimension)+'_'+label_mode +'__'+ noise_mode +str(alpha)+ 'ckpt.pth'
    name = os.path.join("/scratch/private/eunbiyoon/LIM-synthetic-data/dataset", name)
    torch.save(score_model.state_dict(), name)
    return score_model

