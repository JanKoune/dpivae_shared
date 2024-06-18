"""
TODO:
    * Implement grad reverse as a transform
    * Fix the discriminator
"""
import torch
from torch import nn
from utils.transforms import IdentityTransform


class MLP(nn.Module):
    def __init__(self, n_latent, n_dim, layers, input_transform=IdentityTransform(), output_transform=IdentityTransform(), nonlinear_last=None, reverse_grad=False, nonlinearity=nn.ReLU):
        super().__init__()

        self.n_latent = n_latent
        self.n_dim = n_dim
        self.layers = layers
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.nonlinear_last = nonlinear_last
        self.reverse_grad = reverse_grad
        self.nonlinearity = nonlinearity

        self.layers.insert(0, self.n_latent)
        self.layers.append(self.n_dim)

        self.net = nn.Sequential()
        for i in range(len(self.layers) - 1):
            self.net.add_module(
                f"linear_{i}",
                nn.Linear(self.layers[i], self.layers[i + 1]),
            )
            self.net.add_module(f"nonlinear_{i}", self.nonlinearity())
        self.net.pop(-1)

        if self.nonlinear_last is not None:
            self.net.add_module(f"output", self.nonlinear_last())

    def forward(self, z):
        return self.output_transform(self.net(self.input_transform(z)))


class Discriminator(nn.Module):

    def __init__(self, model, eps=1e-8):
        super().__init__()
        self.model = model
        self.eps = eps
        self.fn_loss = nn.BCELoss(reduction='none')


    def forward(self, z):
        return self.model(z)


    def loss(self, z, z_prime):
        z_perm = torch.cat(self.permute_dims(z_prime), dim=-1)
        D_z = self.forward(torch.cat(z, dim=-1))
        D_p = self.forward(z_perm)
        return self.fn_loss(D_z.squeeze(dim=-1), D_p.squeeze(dim=-1))


    def permute_dims(self, z_in):
        z_out = []
        for j, z_j in enumerate(z_in):
            idx_p = torch.stack([torch.randperm(z_j.shape[1]) for _ in range(z_j.shape[0])])
            idx_p = idx_p.unsqueeze(-1).repeat(1, 1, z_j.shape[-1])
            z_out.append(torch.gather(z_j, 1, idx_p))
        return z_out