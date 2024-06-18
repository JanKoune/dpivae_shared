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
import seaborn as sns

from models.nn import MLP
from models.encoders import GaussianEncoder
from models.decoders import GaussianDecoder
from models.vae import VariationalInference
from utils.transforms import StandardScaler

# ==================================================================
# Problem setup
# ==================================================================
# torch.manual_seed(123)
n_data = 2
nd_physics = 32
n_iter = 25_000
n_mc = 128
n_plot = 5_000

# Parameters
nz_full = 4
nz_part = 2
nz_data = 4

# NN settings
layers_nn_full = [256, 64]
layers_nn_part = [256, 64]

# Surrogate models
model_path = "./trained_models/"

# Optimization
lr_vi = 1e-3

# ==================================================================
# Beam example setup
# ==================================================================
# Domain
v_min, v_max = 0.00001, 1.0
v = torch.linspace(v_min, v_max, nd_physics)

# Noise
sigma_meas = torch.tensor(0.2)

# Indices of entries of X corresponding to each effect
z_idx_p = [0, 1]
z_idx_d = [2, 3]
idx_obs = [False, True]
latent_groups = [z_idx_p, z_idx_d]
param_labels = [r"$E \ [Pa]$", r"$x_F [m]$", r"$\mathrm{log}k_v [N/m]$", r"$T [C^o]$"]

mu_prior = torch.tensor([4.0, 0.5, 8.0, -4.0])
sigma_prior = torch.tensor([0.5, 0.1, 1.0, 3.0])

mu_gt = mu_prior
sigma_gt = 0.1 * sigma_prior

# Priors of physics-based latent variables
dist_part_prior = dist.Normal(mu_prior[z_idx_p], sigma_prior[z_idx_p])
dist_full_prior = dist.Normal(mu_prior, sigma_prior)

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

# Load pre-trained surrogates
full_model.load_state_dict(torch.load(model_path + "full_model"))
full_model.eval()

part_model.load_state_dict(torch.load(model_path + "part_model"))
part_model.eval()

# Freeze models
for param in full_model.parameters():
    param.requires_grad = False

for param in part_model.parameters():
    param.requires_grad = False
# # ==================================================================
# # Synthetic data generation
# # ==================================================================
# Sample inputs and outputs from the ground truth
def generate_synthetic_data(n):
    z_gt_sample = dist.Normal(mu_gt, sigma_gt).sample((n,))
    x_sample = full_model(z_gt_sample)
    x_sample += dist.Normal(0.0, sigma_meas).sample(x_sample.shape)
    return z_gt_sample, x_sample

z_gt, x_meas = generate_synthetic_data(n_data)
z_gt_plot, x_meas_plot = generate_synthetic_data(n_plot)
x_meas_plot_mean = x_meas_plot.mean(dim=0).detach()
x_meas_plot_std = x_meas_plot.std(dim=0).detach()

plt.figure()
plt.plot(v, x_meas.T.detach().numpy(), color="blue", linewidth=1.0)
plt.plot(v, x_meas_plot_mean, color="red")
plt.fill_between(v, x_meas_plot_mean - 2 * x_meas_plot_std, x_meas_plot_mean + 2 * x_meas_plot_std, alpha=0.2, color="red")
plt.grid()
plt.title("Measurements")
plt.show()

# ==================================================================
# Encoders
# ==================================================================
full_encoder = GaussianEncoder(
    n_latent=nz_full,
    n_dim=nd_physics,
)

part_encoder = GaussianEncoder(
    n_latent=nz_part,
    n_dim=nd_physics,
)

# ==================================================================
# Decoders
# ==================================================================
full_decoder = GaussianDecoder(full_model, sigma=None)
part_decoder = GaussianDecoder(part_model, sigma=None)

# ==================================================================
# VAE
# ==================================================================
full_vae = VariationalInference(
    full_encoder,
    full_decoder,
    dist_full_prior,
    jitter=1e-6,
)

part_vae = VariationalInference(
    part_encoder,
    part_decoder,
    dist_part_prior,
    jitter=1e-6,
)

# ==================================================================
# Optimization
# ==================================================================
loss_full_vec = []
loss_part_vec = []
optimizer_full = optim.Adam(full_vae.parameters(), lr=lr_vi)
optimizer_part = optim.Adam(part_vae.parameters(), lr=lr_vi)

pbar = trange(n_iter)
for idx in pbar:
    optimizer_full.zero_grad()
    loss_full = full_vae.loss(x_meas, n=n_mc).sum() / (n_data * nd_physics)
    loss_full.backward()
    optimizer_full.step()
    if idx % 100 == 0:
        pbar.set_postfix(ELBO=loss_full.detach(), refresh=True)
    loss_full_vec.append(loss_full.detach())

pbar = trange(n_iter)
for idx in pbar:
    optimizer_part.zero_grad()
    loss_part = part_vae.loss(x_meas, n=n_mc).sum() / (n_data * nd_physics)
    loss_part.backward()
    optimizer_part.step()
    if idx % 100 == 0:
        pbar.set_postfix(ELBO=loss_part.detach(), refresh=True)
    loss_part_vec.append(loss_part.detach())

plt.figure()
plt.plot(range(0, n_iter), loss_full_vec, label="Full model loss")
plt.plot(range(0, n_iter), loss_part_vec, label="Partial model loss")
plt.grid()
plt.legend()
plt.show()

# ==================================================================
# Vizualization
# ==================================================================
# Sample posterior predictive dist.
x_pred_full, z_full_post, dens_z_full_post = full_vae.forward(None, n=n_plot)
x_pred_part, z_part_post, dens_z_part_post = part_vae.forward(None, n=n_plot)
z_full_post, dens_z_full_post = z_full_post.squeeze(1), dens_z_full_post.squeeze(1)
z_part_post, dens_z_part_post = z_part_post.squeeze(1), dens_z_part_post.squeeze(1)

# Get means and std. devs.
mean_pred_full = torch.mean(x_pred_full.squeeze(1), dim=0).detach()
mean_pred_part = torch.mean(x_pred_part.squeeze(1), dim=0).detach()

sigma_pred_full = torch.std(x_pred_full.squeeze(1), dim=0).detach()
sigma_pred_part = torch.std(x_pred_part.squeeze(1), dim=0).detach()

# Ground truth
z_gt_plot, x_gt_plot = generate_synthetic_data(n_plot)
mean_gt_plot = torch.mean(x_gt_plot, dim=0).detach()
sigma_gt_plot = torch.std(x_gt_plot, dim=0).detach()

fig, ax = plt.subplots(1, 2, figsize=(16, 9))
ax[0].plot(v, mean_pred_full, linestyle="dashed", c="blue")
ax[0].fill_between(v, mean_pred_full - 2 * sigma_pred_full, mean_pred_full + 2 * sigma_pred_full, alpha=0.5, color="blue")
ax[0].set_title("Full physics model")

ax[1].plot(v, mean_pred_part, linestyle="dashed", c="blue", label="Mean pred.")
ax[1].fill_between(v, mean_pred_part - 2 * sigma_pred_part, mean_pred_part + 2 * sigma_pred_part, alpha=0.5, color="blue", label=r"$\pm 2 \sigma$" + " pred.")
ax[1].set_title("Partial physics model")

for i in range(2):
    ax[i].plot(v, mean_gt_plot, c="red", label="Mean g.t.")
    ax[i].fill_between(v, mean_gt_plot - 2 * sigma_gt_plot, mean_gt_plot + 2 * sigma_gt_plot, alpha=0.2, color="red", label=r"$\pm 2 \sigma$" + " g.t.")
    ax[i].grid()

ax[1].legend()
plt.show()


# ==================================================================
# Visualize latent space
# ==================================================================
# Sample prior
z_part_prior = dist_part_prior.sample((n_plot,))
z_full_prior = dist_full_prior.sample((n_plot,))

# Plot prior and posterior for partial physics model
df_part_prior = pd.DataFrame(z_part_prior.detach().numpy())
df_part_prior.columns = [param_labels[z_idx_p[i]] for i in range(2)]
df_part_prior.insert(0, "type", ["Prior"] * n_plot)
df_part_post = pd.DataFrame(z_part_post.detach().numpy())
df_part_post.columns = [param_labels[z_idx_p[i]] for i in range(2)]
df_part_post.insert(0, "type", ["Posterior"] * n_plot)
df_part_plot = pd.concat([df_part_prior, df_part_post])
plot_part_latent = sns.pairplot(df_part_plot, hue="type", kind="hist")
plot_part_latent.fig.suptitle("Partial physics model")
for i in range(nz_part):
        plot_part_latent.axes[i, i].axvline(mu_gt[z_idx_p[i]], color="red", linestyle="dashed")
plt.show()

# Plot prior and posterior for full physics model
df_full_prior = pd.DataFrame(z_full_prior.detach().numpy())
df_full_prior.columns = param_labels
df_full_prior.insert(0, "type", ["Prior"] * n_plot)
df_full_post = pd.DataFrame(z_full_post.detach().numpy())
df_full_post.columns = param_labels
df_full_post.insert(0, "type", ["Posterior"] * n_plot)
df_full_plot = pd.concat([df_full_prior, df_full_post])
plot_full_latent = sns.pairplot(df_full_plot, hue="type", kind="hist")
plot_full_latent.fig.suptitle("Full physics model")
for i in range(nz_full):
        plot_full_latent.axes[i, i].axvline(mu_gt[i], color="red", linestyle="dashed")
plt.show()
