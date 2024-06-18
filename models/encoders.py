"""
TODO:
    * Base class for encoders
    * Normalizing flow encoder
"""

import torch
from torch import nn
from torch import distributions as dist

class AmortizedGaussianEncoder(nn.Module):
    """
    Multivariate Gaussian encoder

    TODO:
        * Replace the KL divergence Monte Carlo gradient estimate with analytical calculation
    """
    def __init__(self, n_latent, n_dim, mean_layers, sigma_layers, cov_layers, nonlinear_last=False):
        super().__init__()

        self.n_latent = n_latent
        self.n_dim = n_dim
        self.mean_layers = mean_layers
        self.sigma_layers = sigma_layers
        self.cov_layers = cov_layers
        self.nonlinear_last = nonlinear_last

        self.mean_layers.insert(0, self.n_dim)
        self.mean_layers.append(self.n_latent)

        self.sigma_layers.insert(0, self.n_dim)
        self.sigma_layers.append(self.n_latent)

        self.cov_layers.insert(0, self.n_dim)
        self.cov_layers.append(self.n_latent * self.n_latent)

        # Mean network
        self.mean = nn.Sequential()
        for i in range(len(self.mean_layers) - 1):
            self.mean.add_module(
                f"mean_linear_{i}",
                nn.Linear(self.mean_layers[i], self.mean_layers[i + 1]),
            )
            self.mean.add_module(f"mean_nonlinear_{i}", nn.Sigmoid())

        # log-sigma network
        self.log_sigma = nn.Sequential()
        for i in range(len(self.sigma_layers) - 1):
            self.log_sigma.add_module(
                f"sigma_linear_{i}",
                nn.Linear(self.sigma_layers[i], self.sigma_layers[i + 1]),
            )
            self.log_sigma.add_module(f"sigma_nonlinear_{i}", nn.Sigmoid())

        self.cov = nn.Sequential()
        for i in range(len(self.cov_layers) - 1):
            self.cov.add_module(
                f"cov_linear_{i}",
                nn.Linear(self.cov_layers[i], self.cov_layers[i + 1]),
            )
            self.cov.add_module(f"cov_nonlinear_{i}", nn.Sigmoid())

        if self.nonlinear_last == False:
            self.mean.pop(-1)
            self.log_sigma.pop(-1)
            self.cov.pop(-1)

    def forward(self, x, cond=None, n=1, jitter=1e-8):
        """
        Returns log q_phi(z|x^i)
        """

        if cond is not None:
            x_c = torch.hstack((x, cond))
        else:
            x_c = x

        mu = self.mean(x_c)
        sigma = self.log_sigma(x_c).exp()
        L = torch.tril(self.cov(x_c).reshape(-1, self.n_latent, self.n_latent), diagonal=-1)
        L += torch.diag_embed(sigma + jitter)
        z = self.sample(mu, L, n=n)
        dens_z = self.log_prob(z, mu, L)
        return z, dens_z

    def sample(self, mu, L, n):
        eps = torch.randn((n, *list(mu.shape)))
        return mu + torch.matmul(L, eps.unsqueeze(-1)).squeeze(-1)

    def log_prob(self, z, mu, L):
        return dist.MultivariateNormal(mu, scale_tril=L).log_prob(z)

    def cond_log_prob(self, z, x, cond=None, jitter=1e-8):
        if cond is not None:
            x_c = torch.hstack((x, cond))
        else:
            x_c = x

        mu = self.mean(x_c)
        sigma = self.log_sigma(x_c).exp()
        L = torch.tril(self.cov(x_c).reshape(-1, self.n_latent, self.n_latent), diagonal=-1)
        L += torch.diag_embed(sigma + jitter)
        return self.log_prob(z, mu, L)

    def kl_div(self, prior, x, cond=None, n=1):
        z, dens_z = self.forward(x, cond=cond, n=n)
        dens_p = prior.log_prob(z).sum(dim=-1)
        return torch.mean(dens_z - dens_p, dim=0)



class GaussianEncoder(nn.Module):
    def __init__(self, n_latent, n_dim):
        super().__init__()

        self.n_latent = n_latent
        self.n_dim = n_dim

        # Initialize guide parameters
        self.mu = nn.Parameter(torch.zeros(self.n_latent))
        self.log_sigma = nn.Parameter(torch.zeros(self.n_latent))
        self.cov = nn.Parameter(torch.zeros(self.n_latent * self.n_latent))

    def forward(self, x, n=1, jitter=1e-6):
        """
        Returns log q_phi(z|x^i)
        """
        L = torch.tril(self.cov.reshape(-1, self.n_latent, self.n_latent), diagonal=-1)
        L += torch.diag_embed(self.log_sigma.exp() + jitter)
        z = self.sample(self.mu, L, n=n).unsqueeze(1)
        dens_z = self.log_prob(z, self.mu, L)
        return z, dens_z

    def sample(self, mu, L, n):
        eps = torch.randn((n, *list(mu.shape)))
        return mu + torch.matmul(L, eps.unsqueeze(-1)).squeeze(-1)

    def log_prob(self, z, mu, L):
        return dist.MultivariateNormal(mu, scale_tril=L).log_prob(z)

    def kl_div(self, prior, x, n=1):
        z, dens_z = self.forward(x, n=n)
        dens_p = prior.log_prob(z).sum(dim=-1)
        return torch.mean(dens_z - dens_p, dim=0)