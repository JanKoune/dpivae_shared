"""
TODO:
    * Replace with the new version from `css`
"""

import torch
from utils import device, neg_inf, pos_inf


class StandardScaler:
    """Standardize data by removing the mean and scaling to
    unit variance.  This object can be used as a transform
    in PyTorch data loaders.

    Args:
        mean (FloatTensor): The mean value for each feature in the data.
        scale (FloatTensor): Per-feature relative scaling.

    From: https://discuss.pytorch.org/t/advice-on-implementing-input-and-output-data-scaling/64369
    """

    def __init__(self, mean=None, scale=None):
        if mean is not None:
            mean = torch.FloatTensor(mean).to(device)
        if scale is not None:
            scale = torch.FloatTensor(scale).to(device)
        self.mean_ = mean
        self.scale_ = scale

    def fit(self, sample):
        """Set the mean and scale values based on the sample data.
        """
        self.mean_ = sample.mean(0, keepdim=True)
        self.scale_ = sample.std(0, unbiased=False, keepdim=True)
        return self

    def __call__(self, sample):
        return (sample - self.mean_)/self.scale_

    def inverse_transform(self, sample):
        """Scale the data back to the original representation
        """
        return sample * self.scale_ + self.mean_


class PlausibleBoxScaler:
    """Standardize coordinates with respect to a specified box.

    NOTE: If this transform is preceded by a logit transform, the lower and upper bound
    must also be transformed to the unconstrained space

    Args:
        lb (FloatTensor): 1D tensor of lower bounds
        ub (FloatTensor): 1D tensor of upper bounds

    See: L. Acerbi - Variational Bayes Monte Carlo
    """

    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub

    def fit(self, sample):
        raise NotImplementedError(f"`fit()` is not implemented for this transform")

    def __call__(self, sample):
        return (sample - (self.lb + self.ub)/2) / (self.ub - self.lb)

    def inverse_transform(self, sample):
        """Scale the data back to the original representation
        """
        return sample * (self.ub - self.lb) + (self.lb + self.ub)/2


class IdentityTransform:
    """Dummy transform to be used as default transform in surrogate models
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        return sample

    def inverse_transform(self, sample):
        return sample


class ChainTransform:
    """
    Class used to chain together a series of transforms

    TODO:
        * Should this make a copy of `X` first or change it inplace?
    """

    def __init__(self, *args):
        self.lst_transforms = list(args)

    def __call__(self, sample):
        for transf in self.lst_transforms:
            sample = transf.forward(sample)
        return sample

    def reverse(self, sample):
        for transf in self.lst_transforms.__reversed__():
            sample = transf.reverse(sample)
        return sample
