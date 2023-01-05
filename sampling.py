import os
import os
from scipy.special import gamma
from train import train
import tqdm
import torch
import numpy as np
from util import jointplots
from data import get_data
from torchlevy import LevyStable
from util import VPSDE, calculate_fid
import math
from model import Model
import matplotlib.pyplot as plt
import ot
import torch
import random
import torch.backends.cudnn as cudnn
from sklearn.mixture import GaussianMixture
from likelihood import get_likelihood_fn

def gamma_func(x):
    return torch.tensor(gamma(x))

levy =LevyStable()
def ddim_score_update2(score_model, alpha, x_s, s, t, h=5, clamp=10, device='cuda', mode='approximation',
                       order=0, noise_mode = "isotropic", label_mode = 'fDSM', y=None):
    sde = VPSDE(alpha, schedule='cosine')
    if  y is not None:
        y = torch.ones((x_s.shape[0],))*y
    if label_mode =='gaussian':
        score_s = score_model(x_s,y, s) * torch.pow(sde.marginal_std(s) + 1e-5, -(1))[:, None]
    else:
     score_s = score_model(x_s,y, s) * torch.pow(sde.marginal_std(s)+1e-5, -(alpha - 1))[:, None]

    time_step = s - t
    beta_step = sde.beta(s) * time_step

    x_coeff = 1 + beta_step / alpha
    levy = LevyStable()

    if noise_mode == "isotropic":
        e_L = levy.sample(alpha, 0, size=x_s.shape, is_isotropic=True, clamp=20)
    elif noise_mode == "independent":
        e_L = torch.clamp(levy.sample(alpha, 0, size=x_s.shape, is_isotropic=False), -20, 20)
    elif noise_mode == 'gaussian':
        e_L = torch.randn(size = x_s.shape)*np.sqrt(2)
    # print('no', torch.max(torch.pow(beta_step, 1 / alpha)))
    # print('s',  torch.max(alpha * beta_step))
    # print('score', torch.max(score_s))
    # print('mar',sde.marginal_std(s)[0])
    if label_mode == "fDSM":
        score_coeff = alpha * beta_step
    elif label_mode == "gaussian":
        score_coeff = beta_step*2
    elif label_mode == "ReELS":
        score_coeff =  alpha*gamma_func(alpha - 1) / gamma_func(alpha / 2) ** 2 / h ** (alpha - 2)*beta_step
    noise_coeff = torch.pow(beta_step, 1 / alpha)

    x_t = x_coeff[:, None] * x_s + score_coeff[:, None] * score_s + noise_coeff[:, None] * e_L
    # print('x_t', torch.max(x_t), torch.min(x_t))

    return x_t


def pc_sampler2(score_model,
                score_check_steps=[0, 500, 800, 999],
                batch_size=50000,
                num_steps=1000,
                show_image=True,
                noise_mode = "isotropic", label_mode = 'fDSM',y=None):
    alpha = score_model.alpha
    dimension = score_model.dimension
    sde = VPSDE(alpha,  schedule='cosine')

    t = torch.ones(batch_size, )*sde.T

    levy = LevyStable()
    if noise_mode == "isotropic":
        e_L = levy.sample(alpha, 0, size=(batch_size, dimension), is_isotropic=True, clamp=20)
    elif noise_mode == "independent":
        e_L = torch.clamp(levy.sample(alpha, 0, size=(batch_size, dimension), is_isotropic=False), -20, 20)
    elif noise_mode == 'gaussian':
        e_L = torch.randn(size =(batch_size, dimension))*np.sqrt(2)

    x_s = e_L*sde.marginal_std(t)[:, None]
    # print('ini', torch.max(x_s))

    time_steps = torch.pow(torch.linspace(sde.T, 1e-5, num_steps), 1)

    batch_time_step_s = torch.ones(x_s.shape[0])*time_steps[0]

    data = []
    ts = []
    steps = []
    trajectory = [x_s]
    with torch.no_grad():
        for i, t in enumerate(time_steps[1:]):

            batch_time_step_t = torch.ones(x_s.shape[0]) * t

            x_s = ddim_score_update2(score_model, alpha, x_s, batch_time_step_s, batch_time_step_t,
                                     noise_mode = noise_mode, label_mode = label_mode, y=None)

            trajectory.append(x_s)
            batch_time_step_s = batch_time_step_t
    trajectory = torch.stack(trajectory,dim=1)
    return e_L, x_s, trajectory


def ode_score_update(score_model, sde, x_s, y, s, t, h=0.6, clamp=10):
    alpha = sde.alpha
    if  y is not None:
        y = torch.ones((x_s.shape[0],))*y


    score_s = score_model(x_s, y,s) * torch.pow(sde.marginal_std(s) + 1e-4, -(alpha-1))[:, None]

    lambda_s = sde.marginal_lambda(s)
    lambda_t = sde.marginal_lambda(t)
    h_t = lambda_t - lambda_s
    x_coeff = sde.diffusion_coeff(t) * torch.pow(sde.diffusion_coeff(s), -1)

    score_coeff = alpha * torch.pow(sde.marginal_std(s), alpha - 1) * sde.marginal_std(t) * (-1 + torch.exp(h_t))

    x_t = x_coeff[:, None] * x_s + score_coeff[:, None] * score_s

    return x_t

def ode_sampler(score_model,
                score_check_steps=[0, 500, 800, 999],
                batch_size=50000,
                num_steps=1000,
                show_image=True,
                noise_mode = "isotropic", label_mode = 'fDSM'):
    alpha = score_model.alpha
    sde = VPSDE(alpha,  schedule='cosine')

    t = torch.ones(batch_size, )*sde.T

    levy = LevyStable()
    if noise_mode == "isotropic":
        e_L = levy.sample(alpha, 0, size=(batch_size, dimension), is_isotropic=True, clamp=20)
    elif noise_mode == "independent":
        e_L = torch.clamp(levy.sample(alpha, 0, size=(batch_size, dimension), is_isotropic=False), -20, 20)
    elif noise_mode == 'gaussian':
        e_L = torch.randn(size =(batch_size, dimension))*np.sqrt(2)
    x_s = e_L
    trajectory = [x_s]

    time_steps = torch.linspace(sde.T, 1e-5, num_steps)

    batch_time_step_s = torch.ones(x_s.shape[0])*(sde.T)


    with torch.no_grad():
        for t in tqdm.tqdm(time_steps[1:]):
            batch_time_step_t = torch.ones(x_s.shape[0]) * t

            x_s = ode_score_update(score_model, sde, x_s, y, batch_time_step_s, batch_time_step_t)
            trajectory.append(x_s)
            batch_time_step_s = batch_time_step_t
    trajectory = torch.stack(trajectory, dim=1)

    return  e_L, x_s, trajectory
from sklearn.mixture import GaussianMixture
import seaborn as sns

def plt_marginal_density(score_model, ax, npts=1000, memory=5000, title="q0(x)", dim=1, LOW=-7, HIGH=7, device='cuda', name='img.png', noise_mode= 'isotropic'):
    if dim == 1:
        yy = torch.from_numpy(np.linspace(-1.5,1.5, 1000)).reshape((-1, 1)).float()
        yy =yy.to(device)
        print(yy.shape)

        sde = VPSDE(alpha, beta_min=0.1, beta_max=20, schedule='cosine')
        likelihood_fn = get_likelihood_fn(sde, device, noise_mode= noise_mode, alpha=score_model.alpha, hutchinson_type='Rademacher',
                          rtol=1e-5, atol=1e-5, method='RK45', eps=1e-5)

        logpx = likelihood_fn(score_model, yy, inverse_scalar=0, t_0=1e-5, T=1)

        ax.plot(yy.flatten().detach().cpu().numpy(),np.exp(logpx).reshape((-1,)))
        plt.grid()


    elif dim == 2:
        side = np.linspace(LOW, HIGH, npts)
        xx, yy = np.meshgrid(side, side)
        sde = VPSDE(alpha, beta_min=0.1, beta_max=20, schedule='cosine')
        x = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
        xt = torch.from_numpy(x).type(torch.float32).to(device)
        inds = torch.arange(0, xt.shape[0]).to(torch.int64)
        logpx = []
        likelihood_fn = get_likelihood_fn( sde,device, hutchinson_type='Rademacher',
                                          rtol=1e-5, atol=1e-5, method='RK45', eps=1e-5)
        for ii in torch.split(inds, memory):
            logpx.append(likelihood_fn(score_model, xt[ii]).reshape((-1,)))
        logpx = np.concatenate(logpx, axis=0)
        logpx = logpx[: npts*npts]
        px = np.exp(logpx).reshape(npts, npts)
        ax.imshow(px, cmap='inferno')
from prdc import compute_prdc

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation





from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.transforms as mtransforms
import matplotlib.text as mtext


class MyLine(lines.Line2D):
    def __init__(self, *args, **kwargs):
        # we'll update the position when the line data is set
        self.text = mtext.Text(0, 0, '')
        super().__init__(*args, **kwargs)

        # we can't access the label attr until *after* the line is
        # initiated
        self.text.set_text(self.get_label())

    def set_figure(self, figure):
        self.text.set_figure(figure)
        super().set_figure(figure)

    def set_axes(self, axes):
        self.text.set_axes(axes)
        super().set_axes(axes)

    def set_transform(self, transform):
        # 2 pixel offset
        texttrans = transform + mtransforms.Affine2D().translate(2, 2)
        self.text.set_transform(texttrans)
        super().set_transform(transform)

    def set_data(self, x, y):
        if len(x):
            self.text.set_position((x[-1], y[-1]))

        super().set_data(x, y)

    def draw(self, renderer):
        # draw my label at the end of the line with 2 pixel offset
        super().draw(renderer)
        self.text.draw(renderer)




def animation_gif(trajectory):
    fig, ax = plt.subplots()
    i=0

    line = MyLine([], [], mfc='red', ms=12, label='line label')
    def update(i):
        print(i)


        line.set_data(trajectory[0, :i+1, 0], trajectory[0, :i+1, 1])

    graph_ani = FuncAnimation(fig=fig, func=update, frames=10)
    graph_ani.save('graph_ani.gif', writer='imagemagick',  dpi=100)

    plt.show()
def runner(alpha,  dataset= "swiss roll",train_max_iter=100,batch_size=1000,
           max_iter=100, noise_mode="isotropic", label_mode = "fDSM",dimension=2 ,ratio=4,
           sampler = 'ode_sampler', mode=2,
           condition = False, y= None):
    dir_path = dataset
    dir_path = os.path.join('/scratch/private/eunbiyoon/LIM-synthetic-data', dir_path)
    datasets = get_data(dataset, dimension=dimension,f=ratio)

    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    train_dataset = datasets[0]
    testset = datasets[2]
    x_min = torch.min(testset).item()
    x_max =torch.max(testset).item()
    testset = testset.detach().cpu().numpy()


    sde = VPSDE(alpha, beta_min=0.1, beta_max=20,  schedule='cosine')
    levy = LevyStable()
    F = []
    P=[]
    R=[]
    D=[]
    C=[]
    Ra=[]
    total_distance = 0

    score_model = train(alpha, train_dataset,
                            noise_mode=noise_mode, label_mode = label_mode,
                        max_iter = train_max_iter, dimension =dimension, condition=condition)
    for e in range(max_iter):
        if sampler == "pc_sampler2":
            e_L, sample, trajectory= pc_sampler2(score_model,
                             score_check_steps=[0, 500, 800, 999],
                             batch_size=batch_size,
                             num_steps=1000,
                             show_image=True, noise_mode = noise_mode, label_mode = label_mode,
                             y=y)
        elif sampler == "ode_sampler":
            e_L, sample, trajectory= ode_sampler(score_model,
                             score_check_steps=[0, 500, 800, 999],
                             batch_size=batch_size,
                             num_steps=50,
                             show_image=True, noise_mode = noise_mode, label_mode = label_mode)
        e_L = e_L.cpu().numpy()
        trajectory = trajectory.cpu().numpy()
        sample = sample.cpu().numpy()
        sample = np.nan_to_num(sample)


        distance = ot.sliced_wasserstein_distance(testset[:,:-1], sample, n_projections=10, seed=0)
        total_distance += distance
        fid = calculate_fid(sample, testset[:,:-1],dimension=dimension)
        # metrics = compute_prdc(real_features=testset,
        #                        fake_features=sample,
        #                        nearest_k=5)



        F.append(fid)
        f = np.array(F)

        ra = np.array(Ra)

        average_fid = np.mean(f)
        variance = np.var(f)


        if dimension == 2:
            gm = GaussianMixture(n_components=mode, random_state=0).fit(sample)
            labels = gm.predict(sample)
            a = len(np.where(labels == 0)[0])
            b = batch_size - a
            if a > b:
                predict_f = a / b
            else:
                predict_f = b / a
            Ra.append(predict_f)
            ra = np.array(Ra)
            average_ratio = np.mean(ra)
        print(f'Alpha{alpha}, Average FID:{average_fid}, Variance: {variance}, distance: {total_distance}')

    F = np.array(F)
    average_fid = np.mean(F)
    variance = np.var(F)
    print(f'Alpha{alpha}, FID:{average_fid}, Variance: {variance} ratio{ratio} predict:{average_ratio:.1f}')
    if dimension >1:
        X = sample[:10,0]
        Y = sample[:10,1]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.scatter(X, Y, s=1, label = f' ratio{ratio},Alpha{alpha} predict:{average_ratio:.1f}var:{variance:.4f}FID:{average_fid:.4f}distance: {total_distance/3/(e+1):.4f}')
        trajectory = trajectory[:10,:,:]
        plt.scatter(trajectory[:,0,0], trajectory[:,0,1], s=1, c='red',alpha=0.5)
        plt.scatter(trajectory[:,-1, 0], trajectory[:,-1, 1], s=1, c='blue', alpha=0.5)
        cmap = get_cmap(len(trajectory))
        print('shape', np.shape(trajectory))
        for i in tqdm.tqdm(range(len(trajectory))):
            #c = cmap(i))
            ax.add_line(Line2D(trajectory[i, :, 0], trajectory[i, :, 1], alpha=0.1, c = cmap(i)))
        #     plt.scatter(trajectory[i, :, 0], trajectory[i, :, 1], s=0.02, c='black', alpha=0.5)
        # #
        # plt.scatter(e_L[:, 0], e_L[:, 1], s=1, alpha=0.1, label=f'pre{average_precision:.4f}, rec{average_recall:.4f}, den{average_density:.4f}, cov{average_coverage:.4f}', c='black')
        plt.scatter(testset[:,0], testset[:,1], s=1, label=f'testset', alpha=0.03,c='black')
        plt.xlim(x_min, x_max)
        plt.ylim(x_min, x_max)
        plt.axis('equal')


        # plt.legend(loc='lower right')

    elif dimension==1:
        gm = GaussianMixture(n_components=3, random_state=0).fit(testset)
        x = np.linspace(-1.5, 1.5, 1000).reshape(-1, 1)
        y = np.exp((gm.score_samples(x)).reshape(-1, 1))
        fig, ax = plt.subplots()
        ax.plot(x, y, label=f'pre{precision:.4f}, rec{recall:.4f}, den{density:.4f}, cov{coverage:.4f}')
        ax.legend(loc='upper right')

    name = sampler+ str(ratio)+'frac'+str(dimension)+ label_mode+''+noise_mode+dataset+str(alpha)+'.png'
    if condition==True:
        name = 'condition'+name
    name = os.path.join(dir_path, name)
    #plt_marginal_density(score_model, ax, name=name, dim=dimension, noise_mode=noise_mode)

    plt.savefig(name)

    # name ='density_'+ str(dimension)+ label_mode+''+noise_mode+dataset+str(alpha)+'.png'
    # name = os.path.join(dir_path, name)
    # plt_marginal_density(score_model, ax, name=name, dim =dimension,noise_mode=noise_mode)
    return average_fid

def sample(alpha, path="/scratch/private/eunbiyoon/LIM-synthetic-data/10_fDSM__isotropic1.8ckpt.pth",
            dataset= "mixture of gaussian",noise_mode="isotropic", label_mode = "fDSM", device='cuda', dimension=10):
    testset = get_data(dataset, dimension)[2]
    testset = testset.cpu().numpy()
    sde = VPSDE(alpha, beta_min=0.1, beta_max=20,  schedule='cosine')
    ckpt = torch.load(path, map_location=device)
    score_model = Model(alpha,dimension)
    score_model.to(device)
    score_model.load_state_dict(ckpt, strict=False)
    score_model.eval()


    total_distance = 0
    e_L, sample = pc_sampler2(score_model,
                             score_check_steps=[0, 500, 800, 999],
                             batch_size=500,
                             num_steps=1000,
                             show_image=True, noise_mode = noise_mode, label_mode = label_mode)
    e_L = e_L.cpu().numpy()
    sample = sample.cpu().numpy()
    sample = np.nan_to_num(sample)

    distance = ot.sliced_wasserstein_distance(testset, sample, n_projections=10, seed=0)
    total_distance += distance
    fid = calculate_fid(sample, testset)

    print(f'Alpha{alpha}, Average FID:{fid},  distance: {total_distance/3}')
    X = sample[:, 0]
    Y = sample[:, 1]
    plt.scatter(X, Y, s=1, label=f'Alpha{alpha}, Average FID:{fid:.4f}, distance: {total_distance/3:.4f}')
    plt.scatter(e_L[:,0], e_L[:,1], s=1, label=f'Alpha{alpha}, Average FID:{fid:.4f}, distance: {total_distance/3:.4f}',  c='black',alpha=0.1)
    plt.axis('equal')
    plt.legend(loc='upper right')
    name = str(dimension)+label_mode +noise_mode + dataset + str(alpha) + '.png'
    name = os.path.join("/scratch/private/eunbiyoon/LIM-synthetic-data/dataset", name)
    plt.savefig(name, dpi=500)
    plt.cla()
    plt.clf()




# train_max_iter=500
# max_iter=10
# for noise_mode in tqdm.tqdm(['independent', 'isotropic' ]):
#  for label_mode in ['fDSM', 'ReELS']:
#   for alpha in [1.5, 1.2, 1.8]:
#     for dataset in ["mixture of gaussian", "two moon", "swiss roll", "circles"]:
#         runner(alpha, dataset=dataset, train_max_iter= train_max_iter,
#                max_iter=max_iter, noise_mode=noise_mode, label_mode=label_mode)

# train_max_iter=500
# max_iter=1
# for dimension in [2,10,100,1000,10000]:
#  for alpha, label_mode, noise_mode in [(2.0, 'gaussian', 'gaussian')]:
#     for dataset in ["mixture of gaussian"]:
#         runner(alpha, dataset=dataset, train_max_iter= train_max_iter,dimension = dimension,
#                max_iter=max_iter, noise_mode=noise_mode, label_mode=label_mode)
# train_max_iter=1
# max_iter=1
# condition = False
# y=None
# fraction = [5]
# for dimension in [2]:
# #,(1.8, 'fDSM', 'independent'), (1.8, 'fDSM', 'isotropic'),
#  for alpha, label_mode, noise_mode in [(2.0, 'fDSM', 'isotropic'), (1.5, 'fDSM', 'isotropic')]:
#   for dataset in ["mixture of gaussian"]:
#     F = []
#     for ratio in fraction:
#      if dataset== 'mog':
#          dimension=1
#      elif dataset == 'checkerboard':
#          dimension=2
#      else:
#          pass
#      fid = runner(alpha, dataset=dataset, train_max_iter=train_max_iter, dimension=dimension,
#                       max_iter=max_iter, noise_mode=noise_mode, label_mode=label_mode, ratio=ratio,
#                   sampler = "ode_sampler")
#
#      name = 'fid' + str(dimension) + label_mode + '' + noise_mode + dataset + str(alpha) + '.png'
#      if condition == True:
#          name = 'condition'+name
#      name = os.path.join("/scratch/private/eunbiyoon/LIM-synthetic-data/dataset", name)
#

train_max_iter=100
max_iter=1
condition = False
y=None
fraction = [10]
for dimension in [2]:
#,(1.8, 'fDSM', 'independent'), (1.8, 'fDSM', 'isotropic'),
 for alpha, label_mode, noise_mode in [(2.0, 'fDSM', 'isotropic'), (1.5, 'fDSM', 'isotropic')]:
  for dataset in ["mixture of gaussian"]:
    F = []
    for ratio in fraction:
     if dataset== 'mog':
         dimension=1
     elif dataset == 'checkerboard':
         dimension=2
     else:
         pass
     fid = runner(alpha, dataset=dataset, train_max_iter=train_max_iter, dimension=dimension,
                      max_iter=max_iter, noise_mode=noise_mode, label_mode=label_mode, ratio=ratio,
                  sampler = "pc_sampler2", condition=condition, y= y)

     name = 'fid' + str(dimension) + label_mode + '' + noise_mode + dataset + str(alpha) + '.png'
     if condition == True:
         name = 'condition'+name
     name = os.path.join("/scratch/private/eunbiyoon/LIM-synthetic-data/dataset", name)
     plt.savefig(name, dpi=500)


# for dimension in [2,10]:
#  for alpha, label_mode, noise_mode in [(2.0, 'gaussian', 'gaussian')]:
#   F=[]
#   for f in fraction:
#     for dataset in ["mixture of gaussian"]:
#         fid = runner(alpha, dataset=dataset, train_max_iter= train_max_iter,dimension = dimension,
#                max_iter=max_iter, noise_mode=noise_mode, label_mode=label_mode, f=f)
#         F.append(fid)
#
#         name = 'fid'+str(dimension) + label_mode + '' + noise_mode + dataset + str(alpha) + '.png'
#         name = os.path.join("/scratch/private/eunbiyoon/LIM-synthetic-data/dataset", name)
#   plt.plot(fraction, F)
#   plt.legend(loc='upper right')
#   plt.savefig(name, dpi=500)
# sample(2, path="/scratch/private/eunbiyoon/LIM-synthetic-data/10_gaussian__gaussian1.8ckpt.pth",noise_mode="gaussian", label_mode = "gaussian")
# sample(1.8, path="/scratch/private/eunbiyoon/LIM-synthetic-data/10_fDSM__isotropic1.8ckpt.pth",noise_mode="isotropic", label_mode = "fDSM")
