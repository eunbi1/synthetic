import torch
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_moons, make_swiss_roll, make_s_curve, make_circles
import sklearn

device ='cuda'
mus = torch.from_numpy(np.array([-2., -6., 4.])).to(device).reshape((-1, 1)).float() / 7.
stds = torch.from_numpy(np.array([0.3, 0.3, 1.])).to(device).reshape((-1, 1)).float() / 7. ** 2
probs = torch.from_numpy(np.array([0.4, 0.4, 0.2])).to(device).reshape((-1, 1)).float()

def get_data(dataset, dimension=2, f = 4):
    if dataset == "mixture of gaussian":
        n=int(15000*f/(f+1))
        m = 15000-n
        mu1 = np.array([10]*dimension)
        sigma1 = np.array(np.diag([0.5]*dimension))


        X1 = np.random.multivariate_normal(mean=mu1, cov=sigma1, size=n)

        mu2 = np.array([-10]*dimension)

        sigma2 = np.array(np.diag([0.5]*dimension))
    



        X2 = np.random.multivariate_normal(mean=mu2, cov=sigma2, size=m)


        X = np.concatenate([X1, X2])

        # clf = GaussianMixture(n_components=2, covariance_type="full")
        # clf.fit(X)


    elif dataset == "gaussian":
        mu1 = np.array([5, 5])
        sigma1 = np.array([[1, 0], [0, 1]])
        X = np.random.multivariate_normal(mean=mu1, cov=sigma1, size=150000)


    elif dataset == "two moon":
        X, _ = make_moons(n_samples=150000, noise=0.05)
        X = X * 3 - 2


    elif dataset == "swiss roll":
        X, _ = make_swiss_roll(n_samples=150000, noise=0.05)
        X = X[:, [0, 2]] * 0.5

    elif dataset == "s curve":
        X, _ = make_s_curve(n_samples=150000, noise=0.02)
        X = X[:, [0, 2]] * 7

    elif dataset == "circles":
        X, _ = make_circles(n_samples=150000, noise=0.01, factor=0.7)
        X = X * 6

    elif dataset == 'mog':
        assert 15000 % 10 == 0
        dim = int(dimension)
        x = torch.cat([torch.randn(int(15000 * probs[i]), dim).to(device) * torch.sqrt(stds[i]) + mus[i] for i in range(len(mus))], axis=0).reshape((-1, dim))
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        X= x[indices].float()
    elif dataset == 'point':
        assert dimension == 1
        p = [0.2, 0.2, 0.2, 0.2, 0.2]
        points = [-6., -3, 0., 3, 6.]
        length = 0.001
        def uniform(shape, middle, length):
            return torch.rand(shape).to(device) * length + middle - length / 2.
        x = torch.cat([uniform((int(15000 * p[i]), dim), points[i], length) for i in range(len(points))], axis=0).reshape((-1, dim))
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        X= x[indices].float()
    elif dataset == 'checkerboard':
        assert dimension == 2
        x1 = np.random.rand(15000) * 4 - 2
        x2_ = np.random.rand(15000) - np.random.randint(0, 2, 15000) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        X = torch.from_numpy(np.concatenate([x1[:, None], x2[:, None]], 1) * 2).float()

    elif dataset == "swissroll":
        assert dimension == 2

        data1 = sklearn.datasets.make_swiss_roll(n_samples=15000, noise=0.1)[0]
        data1 = data1.astype("float32")[:, [0, 2]]
        data1 /= 10.
        data1 = torch.from_numpy(data1).float()
        data2 = sklearn.datasets.make_swiss_roll(n_samples=15000, noise=0.3)[0]
        data2 = data2.astype("float32")[:, [0, 2]]
        data2 /= 10.
        data2 = torch.from_numpy(data2).float()

        data3 = sklearn.datasets.make_swiss_roll(n_samples=15000, noise=0.5)[0]
        data3 = data3.astype("float32")[:, [0, 2]]
        data3 /= 10.
        data3 = torch.from_numpy(data3).float()

        data4 = sklearn.datasets.make_swiss_roll(n_samples=15000, noise=0.05)[0]
        data4 = data4.astype("float32")[:, [0, 2]]
        data4 /= 10.
        data4 = torch.from_numpy(data4).float()
        r = 4.5
        data1 = data1.to('cuda')+ torch.tensor([-r, -r]).to('cuda')
        data2 = data2.to('cuda')+ torch.tensor([-r, r]).to('cuda')
        data3 = data3.to('cuda') + torch.tensor([r, -r]).to('cuda')
        data4 =data4.to('cuda') + torch.tensor([r, r]).to('cuda')
        data = torch.cat([ data1, data2, data3, data4], axis=0)
        X=data

    else:

            n = int(15000 * f / (f + 1))
            m = 15000 - n
            mu1 = np.array([10] * dimension)
            sigma1 = np.array(np.diag([0.5] * dimension))

            X1 = np.random.multivariate_normal(mean=mu1, cov=sigma1, size=n)

            mu2 = np.array([-10] * dimension)

            sigma2 = np.array(np.diag([0.5] * dimension))

            X2 = np.random.multivariate_normal(mean=mu2, cov=sigma2, size=m)

            X = np.concatenate([X1, X2])

    np.random.shuffle(X)
    train_data = torch.tensor(X[:5000]).clone().detach().requires_grad_(True)
    valid_data = torch.tensor(X[5000:10000]).clone().detach().requires_grad_(True)
    test_data = torch.tensor(X[10000:]).clone().detach()


    return train_data, valid_data, test_data
