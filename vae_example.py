import numpy as np
import pandas as pd
import scipy
import torch
import torch.distributions as dist
from torch import nn, optim
from tqdm import trange
from tqdm import tqdm

import matplotlib as mpl
mpl.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns

from models.nn import MLP
from models.encoders import AmortizedGaussianEncoder
from models.decoders import GaussianDecoder
from models.vae import VariationalInference
from utils.transforms import StandardScaler

# ==================================================================
# Problem setup
# ==================================================================
# torch.manual_seed(123)
n_train = 1024
n_batch = 32
nd_physics = 32
n_iter = 20_000
n_mc = 64
n_plot = 5_000

# Parameters
nz_full = 4
nz_part = 2
nz_data = 4

# NN settings
layers_nn_full = [256, 64]
layers_nn_part = [256, 64]
layers_nn_data = [256]

# Surrogate models
model_path = "./trained_models/"

# Optimization
lr_vae = 1e-4

# ==================================================================
# Beam example
# ==================================================================
# Domain
v_min, v_max = 0.00001, 1.0
v = torch.linspace(v_min, v_max, nd_physics)

# Noise
sigma_p = torch.tensor(0.2)

# Indices of entries of X corresponding to each effect
z_idx_p = [0, 1]
z_idx_d = [2, 3]
idx_obs = [False, True]
latent_groups = [z_idx_p, z_idx_d]
param_labels = [r"$E \ [Pa]$", r"$x_F [m]$", r"$\mathrm{log}k_v [N/m]$", r"$T [C^o]$"]

mu_prior = torch.tensor([4.0, 0.5, 8.0, -4.0])
sigma_prior = torch.tensor([0.5, 0.1, 1.0, 3.0])

mu_gt = mu_prior
sigma_gt = 0.7 * sigma_prior

# Priors of physics-based latent variables
mu_hybrid_prior = torch.cat((mu_prior[z_idx_p], torch.zeros(nz_data)))
sigma_hybrid_prior = torch.cat((sigma_prior[z_idx_p], torch.ones(nz_data)))
dist_hybrid_prior = dist.Normal(mu_hybrid_prior, sigma_hybrid_prior)

# ==================================================================
# Load data and models
# ==================================================================
# Load data, interpolate y
X_full_load = torch.load("./data/X_beam.pt")
y_full_load = torch.load("./data/y_beam.pt")
X_part_load = torch.load("./data/X_beam_partial.pt")
y_part_load = torch.load("./data/y_beam_partial.pt")

# Load X
X_full = X_full_load[1:]
X_part = X_part_load[1:]

y_full_raw = y_full_load[1:]
y_part_raw = y_part_load[1:]
coords = y_full_load[0]

# Interpolate y
f_full = scipy.interpolate.interp1d(coords.numpy(), y_full_raw.numpy())
f_part = scipy.interpolate.interp1d(coords.numpy(), y_part_raw.numpy())
y_full = torch.tensor(f_full(v.numpy()))
y_part = torch.tensor(f_part(v.numpy()))

# Scaling
X_scaler_full = StandardScaler()
X_scaler_full = X_scaler_full.fit(X_full)

X_scaler_part = StandardScaler()
X_scaler_part = X_scaler_part.fit(X_part)

# loss
loss_fn_train = nn.MSELoss()
loss_fn_test = nn.MSELoss()

# Models
full_model = MLP(
    nz_full,
    nd_physics,
    layers_nn_full,
    input_transform=X_scaler_full,
    nonlinear_last=None,
)

part_model = MLP(
    nz_part,
    nd_physics,
    layers_nn_part,
    input_transform=X_scaler_part,
    nonlinear_last=None,
)

data_model = MLP(
    nz_data,
    nd_physics,
    layers_nn_data,
    nonlinear_last=None,
)

# Load pre-trained surrogates
full_model.load_state_dict(torch.load(model_path + "full_model"))
full_model.eval()

# Load pre-trained surrogates
part_model.load_state_dict(torch.load(model_path + "part_model"))
part_model.eval()

# Freeze models
for param in full_model.parameters():
    param.requires_grad = False

# Freeze models
for param in part_model.parameters():
    param.requires_grad = False

# # ==================================================================
# # Synthetic data generation
# # ==================================================================
# Sample inputs and outputs from the ground truth
def generate_synthetic_data(n):
    z_gt_sample = dist.Normal(mu_gt, sigma_gt).sample((n,))
    x_sample = full_model(z_gt_sample)
    x_sample += dist.Normal(0.0, sigma_p).sample(x_sample.shape)
    return z_gt_sample, x_sample

z_gt, x_meas = generate_synthetic_data(n_train)
z_gt_plot, x_meas_plot = generate_synthetic_data(n_plot)
x_meas_plot_mean = x_meas_plot.mean(dim=0).detach()
x_meas_plot_std = x_meas_plot.std(dim=0).detach()

plt.figure()
plt.plot(v, x_meas.T.detach().numpy(), color="blue", linewidth=1.0, alpha=0.2)
plt.plot(v, x_meas_plot_mean, color="red")
plt.fill_between(v, x_meas_plot_mean - 2 * x_meas_plot_std, x_meas_plot_mean + 2 * x_meas_plot_std, alpha=0.2, color="red")
plt.grid()
plt.title("Measurements")
plt.show()

# ==================================================================
# Encoders
# ==================================================================
encoder = AmortizedGaussianEncoder(
    n_latent=nz_part + nz_data,
    n_dim=nd_physics,
    mean_layers=[128],
    sigma_layers=[128],
    cov_layers=[128],
)

# ==================================================================
# Decoders
# ==================================================================
class HybridModel(nn.Module):
    def __init__(self, phys_model, nn_model):
        super().__init__()
        self.phys_model = phys_model
        self.nn_model = nn_model
    def forward(self, z):
        z_phys = z[..., 0:nz_part]
        z_data = z[..., nz_part:(nz_part + nz_data)]
        return self.phys_model(z_phys) + self.nn_model(z_data)

hybrid_model = HybridModel(part_model, data_model)

# Response physics-driven decoder
decoder = GaussianDecoder(hybrid_model, sigma=None)

# ==================================================================
# VAE
# ==================================================================
vae = VariationalInference(
    encoder,
    decoder,
    dist_hybrid_prior,
    jitter=1e-6,
)

# ==================================================================
# Optimization
# ==================================================================
loss_vec = []
optimizer = optim.Adam(vae.parameters(), lr=lr_vae)

pbar = trange(n_iter)
for idx in pbar:
    optimizer.zero_grad()
    sample_idx = torch.multinomial(torch.ones(n_train), n_batch, replacement=False)
    x_i = x_meas[sample_idx, :]
    loss_full = vae.loss(x_i, n=n_mc).sum() / (n_batch * nd_physics)
    loss_full.backward()
    optimizer.step()
    if idx % 100 == 0:
        pbar.set_postfix(ELBO=loss_full.detach(), refresh=True)
    loss_vec.append(loss_full.detach())


plt.figure()
plt.plot(range(0, n_iter), loss_vec, label="loss")
plt.grid()
plt.legend()
plt.show()

# ==================================================================
# Vizualization
# ==================================================================
# Vizualization parameters
n_interp = 5
alpha_plot = torch.tensor(0.99)
cmap_name = "plasma"

# Plot interpolation of the response across the latent space
c_interp = mpl.colormaps[cmap_name](np.linspace(0.0, 1.0, n_interp))
fig_pred, ax_pred = plt.subplots(3, 4, figsize=(12, 9), sharex="col", layout="compressed")

# Linear interpolation across a dimension of the latent space while keeping
# the remaining latent variables equal to the ground truth mean
for idx_z_interp in range(4):

    # Interpolate z
    z_interp = torch.linspace(
        mu_gt[idx_z_interp] - 3.0 * sigma_gt[idx_z_interp],
        mu_gt[idx_z_interp] + 3.0 * sigma_gt[idx_z_interp],
        n_interp,
    )
    z_gt_interp = mu_gt.repeat(n_interp, 1)
    z_gt_interp[:, idx_z_interp] = z_interp
    z_idx_obs = np.array(z_idx_d)[np.array(idx_obs)]
    z_idx_nobs = np.array(z_idx_d)[np.logical_not(np.array(idx_obs))]

    # Generate synthetic data
    x_interp = full_model(z_gt_interp)
    x_interp += dist.Normal(0.0, sigma_p).sample(x_interp.shape)

    # VAE prediction
    x_pred, z_post, dens_z_post = vae.forward(x_interp, n=n_plot)
    x_pred_phys = part_model(z_post[..., 0:nz_part])
    x_pred_data = data_model(z_post[..., nz_part:(nz_part + nz_data)])

    # Get means and std. devs.
    mean_pred = torch.mean(x_pred, dim=0).detach()
    sigma_pred = torch.std(x_pred, dim=0).detach()

    mean_pred_phys = torch.mean(x_pred_phys, dim=0).detach()
    sigma_pred_phys = torch.std(x_pred_phys, dim=0).detach()

    mean_pred_data = torch.mean(x_pred_data, dim=0).detach()
    sigma_pred_data = torch.std(x_pred_data, dim=0).detach()

    # Create colorbar for the current latent variable
    norm_bar = Normalize(vmin=z_interp[0], vmax=z_interp[-1])
    cmap_bar = LinearSegmentedColormap.from_list(cmap_name, c_interp, N=n_interp)
    smap_bar = ScalarMappable(norm_bar, cmap=cmap_bar)

    for i in range(n_interp):

        # Physics-based model prediction
        ax_pred[0, idx_z_interp].plot(
            v,
            mean_pred_phys[i],
            alpha=0.5,
            color=c_interp[i],
            label=param_labels[idx_z_interp] + r"$={:.3f}$".format(z_interp[i]),
        )
        ax_pred[0, idx_z_interp].fill_between(
            v,
            mean_pred_phys[i] - 2.0 * sigma_pred_phys[i],
            mean_pred_phys[i] + 2.0 * sigma_pred_phys[i],
            alpha=0.5,
            color=c_interp[i],
        )

        # Data driven model prediction
        ax_pred[1, idx_z_interp].plot(
            v,
            mean_pred_data[i],
            alpha=0.5,
            color=c_interp[i],
            label=param_labels[idx_z_interp] + r"$={:.3f}$".format(z_interp[i]),
        )
        ax_pred[1, idx_z_interp].fill_between(
            v,
            mean_pred_data[i] - 2.0 * sigma_pred_data[i],
            mean_pred_data[i] + 2.0 * sigma_pred_data[i],
            alpha=0.5,
            color=c_interp[i],
        )

        # Hybrid model prediction
        ax_pred[2, idx_z_interp].plot(
            v,
            mean_pred[i],
            alpha=0.5,
            color=c_interp[i],
            label=param_labels[idx_z_interp] + r"$={:.3f}$".format(z_interp[i]),
        )
        ax_pred[2, idx_z_interp].fill_between(
            v,
            mean_pred[i] - 2.0 * sigma_pred[i],
            mean_pred[i] + 2.0 * sigma_pred[i],
            alpha=0.5,
            color=c_interp[i],
        )

        # Plot measurements
        ax_pred[2, idx_z_interp].scatter(v, x_interp[i].detach(), color=c_interp[i])

    ax_pred[0, idx_z_interp].grid()
    ax_pred[1, idx_z_interp].grid()
    ax_pred[2, idx_z_interp].grid()

    ax_pred[2, idx_z_interp].set_xlabel("X-position [m]", fontsize=16)
    cbar_pred = fig_pred.colorbar(
        smap_bar, ax=ax_pred[0, idx_z_interp], orientation="horizontal", location="top"
    )
    cbar_pred.set_label(label=param_labels[idx_z_interp], size=18)
    cbar_pred.ax.tick_params(labelsize=12)

ax_pred[0, 0].set_ylabel("Physics-based pred. [m]", fontsize=18)
ax_pred[1, 0].set_ylabel("Data-driven pred. [m]", fontsize=18)
ax_pred[2, 0].set_ylabel("Combined pred. [m]", fontsize=18)
