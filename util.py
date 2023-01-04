import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy
import torch
import argparse
import math

class VPSDE:
    def __init__(self, alpha, beta_min=0.1, beta_max=20, schedule='cosine', device='cuda'):
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.cosine_s = 0.008
        self.schedule = schedule
        self.cosine_beta_max = 999.
        self.cosine_t_max = math.atan(self.cosine_beta_max * (1. + self.cosine_s) / math.pi) * 2. * (
                    1. + self.cosine_s) / math.pi - self.cosine_s
        if schedule == 'cosine':
            # For the cosine schedule, T = 1 will have numerical issues. So we manually set the ending time T.
            # Note that T = 0.9946 may be not the optimal setting. However, we find it works well.
            self.T = 0.9946
        else:
            self.T = 1.
        self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.))
        self.alpha = alpha


    def beta(self, t):
        if self.schedule =='linear':
            beta= (self.beta_1 - self.beta_0) * t + self.beta_0
        elif self.schedule == 'cosine':
            beta = math.pi/2*self.alpha*(self.cosine_s+1)*torch.tan( (t+self.cosine_s)/(1+self.cosine_s)*math.pi/2 )
        return torch.clamp(beta,-20,20)


    def marginal_log_mean_coeff(self, t):
        if self.schedule =='linear':
          log_alpha_t = - 1 / (2 * self.alpha) * (t ** 2) * (self.beta_1 - self.beta_0) - 1 / self.alpha * t * self.beta_0
        elif self.schedule == 'cosine':
            log_alpha_fn = lambda s: torch.log(torch.clamp(torch.cos((s + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.),-1,1))
            log_alpha_t =  log_alpha_fn(t) - self.cosine_log_alpha_0

        return log_alpha_t

    def diffusion_coeff(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        sigma = torch.pow(1. - torch.exp(self.alpha * self.marginal_log_mean_coeff(t)), 1 / self.alpha)
        return sigma

    def marginal_lambda(self, t):
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_sigma = torch.log(torch.pow(1. - torch.exp(self.alpha * log_mean_coeff), 1 / self.alpha)+1e-5)
        return log_mean_coeff - log_sigma

    def inverse_lambda(self,l):
        return (-self.beta_0+torch.pow(self.beta_0**2+2*(self.beta_1-self.beta_0)*torch.log(1+torch.exp(-l*self.alpha)),1/2))/(self.beta_1-self.beta_0)


def jointplots(data, score_model, ts, steps, hue=None, height=9, ratio=5, space=0.03, xlim=(-7, 7), ylim=(-7, 7),
               pdf_ylim=np.array([0, 0.4])):
    sns.set_style("white")
    """
    -------------------
    Input Parameters
    -------------------
    xs      : (list or str) feature name(s) of data
    y       : (str) feature name of data
    data    : (pandas.DataFrame)
    hue     : (str) semantic variable that is mapped to determine the color of plot elements. Semantic variable that is mapped to determine the color of plot elements.

    height  : (float) size of the figure
    ratio   : (float) ratio of the joint axes height to marginal axes height.
    space   : (float) space between the joint and marginal axes

    xlabels : (list or str) xlabels
    ylabel  : (str) ylabel
    margin_norm : (boolean) if True, kdeplots at marginal axes have same scale.
    """
    ### 1. input check
    # input type
    # assert(len(xs) == len(ys))
    xs = data[:, :, 0]
    ys = data[:, :, 1]

    ### 2. figure
    h_margin = height / (ratio + 1)
    h_joint = height - h_margin

    n_x = len(xs)

    widths = [h_joint] * n_x + [h_margin]
    heights = [h_margin, h_joint]
    ncols = len(widths)
    nrows = len(heights)

    fig = plt.figure(figsize=(sum(widths), sum(heights)))

    ### 3. gridspec preparation
    spec = fig.add_gridspec(ncols=ncols, nrows=nrows,
                            width_ratios=widths, height_ratios=heights,
                            wspace=space, hspace=0
                            )

    ### 4. setting axes
    axs = {}
    for i in range(ncols * nrows):
        axs[i] = fig.add_subplot(spec[i // ncols, i % ncols])

    ### 5. jointplots (scatterplot + kdeplot)
    for i, (x, y, step, t) in enumerate(zip(xs, ys, steps, ts), ncols):

        # orange contour
#         a, Y = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-7, 7, 100))

#         XX = np.array([a.ravel(), Y.ravel()])
#         Z = -clf.score_samples(XX.T)
#         Z = Z.reshape(a.shape)

#         axs[i].contourf(
#             a, Y, Z, norm=LogNorm(vmin=Z.min(), vmax=Z.max()), levels=np.logspace(-3, 3, 100), alpha=0.8,
#             cmap="Oranges_r", zorder=0)

        # blue dots
        sns.scatterplot(x=x, y=y, data=data, hue=hue, alpha=0.8, s=100, ax=axs[i], zorder=1, legend=False)
        axs[i].set_xlim(xlim)
        axs[i].set_ylim(ylim)

        # score black arrow
        x = torch.arange(-7, 7.1, 1.0)
        y = torch.arange(-7, 7.1, 1.0)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        coord = torch.stack([grid_x, grid_y], dim=-1).cpu()
        coord = coord.reshape(-1, coord.size(-1))

        score = score_model(coord.cpu(), torch.ones(coord.shape[0]).cpu() * t).detach().numpy()
        score /= np.sqrt(np.sum(score ** 2))
        score *= 8

        for j in range(len(coord)):
            axs[i].arrow(coord[j, 0],  # x1
                         coord[j, 1],  # y1
                         score[j, 0],  # x2 - x1
                         score[j, 1],  # y2 - y1
                         head_width=0.15,
                         color='k'
                         )

        axs[i].set_yticklabels([])
        axs[i].set_xticklabels([])
        axs[i].set_xlabel(f"timestep={step+1}", fontsize=30)

    ### 6. kdeplots at marginal axes
    axs[ncols - 1].axis("off")

    axes_mx = list(range(ncols - 1))
    axes_my = 2 * ncols - 1

    for i, x in zip(axes_mx, xs):
        x = x[x > xlim[0]]
        x = x[x < xlim[1]]
        sns.kdeplot(x=x, data=data, hue=hue, fill=True, ax=axs[i], zorder=0, legend=False)

        axs[i].set_xlim(xlim)
        axs[i].set_ylim(pdf_ylim)
        axs[i].set_xlabel("")
        axs[i].set_xticklabels([])
        axs[i].spines["left"].set_visible(False)
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)
        axs[i].axis('off')

    y = ys[-1]
    y = y[y > ylim[0]]
    y = y[y < ylim[1]]
    sns.kdeplot(y=y, data=data, hue=hue, fill=True, ax=axs[axes_my], zorder=0, legend=False)
    plt.axis('off')
    # axs[axes_my].set_ylim(pdf_ylim)
    axs[axes_my].set_ylim(xlim)
    axs[axes_my].set_xlim(pdf_ylim)
    axs[axes_my].set_ylabel("")
    axs[axes_my].set_yticklabels([])
    axs[axes_my].spines["bottom"].set_visible(False)
    axs[axes_my].spines["top"].set_visible(False)
    axs[axes_my].spines["right"].set_visible(False)

    ### 7. unnecessary elements removal
    # 7.1. labels and ticklabels
    axes_j = list(range(ncols, 2 * ncols - 1))
    for i in axes_j:
        if i != ncols:
            axs[i].set_ylabel("")
            axs[i].set_yticklabels([])

    # 7.2. marginal axes
    for i in axes_mx:
        if i != 0:
            axs[i].set_ylabel("")

        axs[i].set_yticklabels([])

    plt.tight_layout()
    plt.show()

    return fig, axs


def dict2namespace(config_dic):
    """dictionary를 namespace로 바꿔줌."""
    namespace = argparse.Namespace()
    for key, value in config_dic.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def calculate_fid(act1, act2, dimension=1):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    if dimension==1:
        covmean = scipy.linalg.sqrtm(np.outer(sigma1, sigma2))
    else:
        covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid