import torch
from torch import nn
from torch import distributions as dist

class VariationalInference(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        prior,
        jitter=1e-6,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.jitter = jitter

    def forward(self, x, n=1):
        z, dens_z = self.encode(x, n=n)
        x_hat = self.decode(z)
        return x_hat, z, dens_z

    def loss(self, x, n):
        x_hat, z, dens_z = self.forward(x, n=n)
        kl = self.kl_div(x, n=n)
        r = self.decoder.log_prob(x, z)
        return kl - r

    def encode(self, x, n=1):
        z, dens_z = self.encoder(x, n=n)
        return z, dens_z

    def kl_div(self, x, n=1):
        return self.encoder.kl_div(self.prior, x, n=n)

    def decode(self, z):
        return self.decoder(z)