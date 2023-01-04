import torch
from util import VPSDE
import torch.nn.functional as F
from torch import nn
import math
class Model(torch.nn.Module):
    def __init__(self, alpha, dimension=2):
        super().__init__()

        self.alpha = alpha
        self.sde = VPSDE(alpha, schedule='cosine')
        self.dimension = dimension
        k = int(dimension/2)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(dimension + 1, 16*k),
            torch.nn.ReLU(),
            torch.nn.Linear(16*k, 32*k),
            torch.nn.ReLU(),
            torch.nn.Linear(32*k, 32*k),
            torch.nn.ReLU(),
            torch.nn.Linear(32*k, 16*k),
            torch.nn.ReLU(),
            torch.nn.Linear(16*k, dimension)
        )

    def forward(self, x, t):
        t = 10 * t - 5
        x = torch.concat([x, t[:, None]], dim=-1)
        return self.model(x)

def get_timestep_embedding(timesteps, embedding_dim=128):
    """
      From Fairseq.
      Build sinusoidal embeddings.
      This matches the implementation in tensor2tensor, but differs slightly
      from the description in Section 3.5 of "Attention Is All You Need".
      https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py
    """
    timesteps = 1000 * timesteps 
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * -emb)

    emb = timesteps.float().view(-1, 1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, [0,1])

    return emb

class MLP(torch.nn.Module):
    def __init__(self, input_dim, layer_widths, activate_final=False, activation_fn=F.relu):
        super(MLP, self).__init__()
        layers = []
        prev_width = input_dim
        for layer_width in layer_widths:
            layers.append(torch.nn.Linear(prev_width, layer_width))
            # # same init for everyone
            # torch.nn.init.constant_(layers[-1].weight, 0)
            prev_width = layer_width
        self.input_dim = input_dim
        self.layer_widths = layer_widths
        self.layers = torch.nn.ModuleList(layers)
        self.activate_final = activate_final
        self.activation_fn = activation_fn

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation_fn(layer(x))
        x = self.layers[-1](x)
        if self.activate_final:
            x = self.activation_fn(x)
        return x


class ScoreNetwork(torch.nn.Module):

    def __init__(self, alpha=1.8, dimension=2, encoder_layers=[16], pos_dim=16, decoder_layers=[128, 128], act_fn=nn.SiLU):
        super().__init__()
        self.alpha = alpha
        self.sde = VPSDE(alpha, schedule='cosine')
        self.dimension = dimension
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim * 2
        self.locals = [encoder_layers, pos_dim, decoder_layers, dimension]

        self.net = MLP(2 * t_enc_dim,
                       layer_widths=decoder_layers + [dimension],
                       activate_final=False,
                       activation_fn=act_fn())

        self.t_encoder = MLP(pos_dim,
                             layer_widths=encoder_layers + [t_enc_dim],
                             activate_final=False,
                             activation_fn=act_fn())

        self.x_encoder = MLP(dimension,
                             layer_widths=encoder_layers + [t_enc_dim],
                             activate_final=False,
                             activation_fn=act_fn())

    def forward(self, x, t):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.t_encoder(temb)
        xemb = self.x_encoder(x)
        h = torch.cat([xemb, temb], -1)
        out = self.net(h)
        return out