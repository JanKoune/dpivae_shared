"""
TODO:
    * Likelihood estimation should be part of the decoder but currently it is done in the VAE class. FIX.
"""

import torch
from torch import nn
from torch import distributions as dist
from torch.autograd import Function
from utils.transforms import IdentityTransform

class GaussianDecoder(nn.Module):
    def __init__(self, model, sigma=None, input_transform=IdentityTransform()):
        super().__init__()
        self.input_transform = input_transform
        self.model = model

        if sigma is not None:
            self.log_sigma = torch.log(sigma)
        else:
            self.log_sigma = nn.Parameter(torch.tensor(0.0))

    def forward(self, z):
        x_hat = self.model(self.input_transform(z))
        noise = dist.Normal(
            torch.zeros_like(x_hat), self.log_sigma.exp() * torch.ones_like(x_hat)
        ).rsample()
        return x_hat + noise

    def log_prob(self, x, z):
        x_hat = self.model(self.input_transform(z))
        return (
            dist.Normal(x_hat, self.log_sigma.exp())
            .log_prob(x)
            .sum(dim=-1)
            .mean(dim=0)
        )
