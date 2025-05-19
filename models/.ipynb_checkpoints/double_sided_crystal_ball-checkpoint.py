import torch
import torch.nn as nn

import numpy as np

from torch import Tensor, Size
from scipy.interpolate import interp1d
import torch.distributions as D

# from dscb import *


def interpolate_gpu(x_values, y_values, y_targets):
    """
    Interpolates cdf_values at y_targets using x_values on the GPU.
    """

    x_values = x_values.to(y_targets.device)  # Ensure on same device
    y_values = y_values.to(y_targets.device)

    # Find indices of nearest x_values
    indices = torch.searchsorted(x_values, y_targets)  # Find where to insert
    indices = torch.clamp(indices, 0, len(x_values) - 1) # Clamp to valid indices

    # Handle edge cases to prevent out of bounds errors
    indices_lower = torch.clamp(indices - 1, 0, len(x_values) - 1)
    indices_upper = indices

    x_lower = x_values[indices_lower]
    x_upper = x_values[indices_upper]
    y_lower = y_values[indices_lower]
    y_upper = y_values[indices_upper]

    # Linear interpolation
    weights = (y_targets - x_lower) / (x_upper - x_lower)
    interpolated_values = y_lower + weights * (y_upper - y_lower)
    return interpolated_values

def interpolate_gpu_batched(x_values, y_values, y_targets):
    """
    Interpolates y_values at y_targets using x_values on the GPU, with batching.

    Args:
        x_values: Tensor of shape (batch_size, n_x) representing x-coordinates.
        y_values: Tensor of shape (batch_size, n_x) representing y-coordinates.
        y_targets: Tensor of shape (batch_size, n_targets) representing target x-coordinates.

    Returns:
        Tensor of shape (batch_size, n_targets) representing interpolated y-values.
    """

    batch_size, n_x = x_values.shape
    _, n_targets = y_targets.shape

    x_values = x_values.to(y_targets.device)  # Ensure on same device
    y_values = y_values.to(y_targets.device)

    # Find indices of nearest x_values for each batch and target
    indices = torch.searchsorted(x_values, y_targets)  # Key change: Specify dimension
    indices = torch.clamp(indices, 0, n_x - 1)

    # Handle edge cases
    indices_lower = torch.clamp(indices - 1, 0, n_x - 1)
    indices_upper = indices

    # Gather values using advanced indexing (critical for batching)
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, n_targets).to(y_targets.device)
    x_lower = x_values[batch_indices, indices_lower]
    x_upper = x_values[batch_indices, indices_upper]
    y_lower = y_values[batch_indices, indices_lower]
    y_upper = y_values[batch_indices, indices_upper]


    # Linear interpolation
    weights = (y_targets - x_lower) / (x_upper - x_lower + 1e-8)  # Add small epsilon for stability
    interpolated_values = y_lower + weights * (y_upper - y_lower)

    return interpolated_values


class DoubleSidedCrystalBall(torch.distributions.Distribution):
    """
    Double Crystal Ball distribution.

    Args:
        mu (torch.Tensor): Mean of the distribution.
        sigma (torch.Tensor): Standard deviation of the distribution.
        logit_a1 (torch.Tensor): Left tail parameter.
        logit_a1 (torch.Tensor): Right tail parameter.
        https://root-forum.cern.ch/t/how-to-fit-a-double-sided-crystal-ball-in-roofit/45293

    """

    def __init__(self, mu, width, a1, a2, p1, p2, xmin=-3.5, xmax=3.5):
        super().__init__()
        self.mu = torch.as_tensor(mu)
        self.width = torch.as_tensor(width)
        self.a1 = torch.as_tensor(a1)
        self.a2 = torch.as_tensor(a2)
        self.p1 = torch.as_tensor(p1)
        self.p2 = torch.as_tensor(p2)
        self.xmax = torch.as_tensor(xmax)
        self.xmin = torch.as_tensor(xmin)

        self.A1 = torch.pow(self.p1 / self.a1, self.p1) * torch.exp(-self.a1**2/2)
        self.A2 = torch.pow(self.p2 / self.a2, self.p2) * torch.exp(-self.a2**2/2)
        self.B1 = self.p1/self.a1 - self.a1
        self.B2 = self.p2/self.a2 - self.a2
        # Pre-calculate constants for efficiency
        #self._norm_const_left = self._calculate_normalization_constant(self.alpha_left, self.n)
        #self._norm_const_right = self._calculate_normalization_constant(self.alpha_right, self.n)
        self.normalization = self._calculate_normalization_constant()
        self.log_normalization = - torch.log(self.normalization)  # 1/N --> -log(n)

    def _calculate_normalization_constant(self):
        """Calculates the normalization constant for one side of the Crystal Ball."""
        u_left = -self.a1
        u_right = self.a2
        u_min_left =  (self.xmin - self.mu)/self.width
        u_max_right = (self.xmax - self.mu)/self.width
        # remember to add the width as the jacobian
        norm_left = self.width*(self.A1/(self.p1-1))*( (self.B1 - u_left)**(1-self.p1) -  (self.B1 - u_min_left )**(1-self.p1)) 
        norm_right = self.width*(self.A2/(1-self.p2))*( (self.B2 + u_max_right)**(1-self.p2) - (self.B2 + u_right)**(1-self.p2)) 
        norm_gauss = self.width*(torch.pi/2)**0.5 *(torch.erf(u_right/(2.**0.5)) -torch.erf(u_left/(2.**0.5)))
        self.norm_left = norm_left
        self.norm_right = norm_right
        self.norm_gauss = norm_gauss
        return norm_left + norm_gauss + norm_right

    def cdf(self, x):
        x = torch.as_tensor(x)
        if x.ndim == 0:
            x = torch.tensor([x])
        if x.ndim < 2:
            x = torch.as_tensor(x).unsqueeze(-1)
        else:
            x = torch.as_tensor(x)
        out = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
        u = (x - self.mu)/self.width
        u_min_left = (self.xmin - self.mu)/self.width
        u_max_right = (self.xmax - self.mu)/self.width
        u_left = -self.a1
        u_right = self.a2
        
        left_mask = (u>=u_min_left) & (u<u_left)
        center_mask = (u>=u_left)&(u<u_right)
        right_mask = (u>=u_right)&(u<u_max_right)
        out_left_mask =  (u<u_min_left)
        out_right_mask =  (u>=u_max_right)

        out = torch.where(
            left_mask,
            self.width*(self.A1/(self.p1-1))*( (self.B1 - u)**(1-self.p1)-(self.B1 - u_min_left )**(1-self.p1)),
            
            torch.where(
                right_mask,
                self.norm_left + self.norm_gauss + self.width*(self.A2/(1-self.p2))*( (self.B2 + u)**(1-self.p2) - (self.B2 + u_right)**(1-self.p2)),
                self.norm_left + self.width*(torch.pi/2)**0.5 *(torch.erf(u/(2.**0.5)) -torch.erf(u_left/(2.**0.5)))
            )
        )
        out /= self.normalization
        out[out_left_mask] *= 0.
        out[out_right_mask] = 1.
                
        
        return out.squeeze()

    def log_prob(self, x):
        """
        Computes the log probability of the given value.

        Args:
            x (torch.Tensor): Value to compute the log probability for.

        Returns:
            torch.Tensor: Log probability of x.
        """
        x = torch.as_tensor(x)
        if x.ndim == 0:
            x = torch.tensor([x])
        if x.ndim < 2:
            u = (x.unsqueeze(-1) - self.mu) / self.width
        else:
            u = (x - self.mu) / self.width
        
        # Use vectorized operations for efficiency
        left_mask = u < -self.a1
        right_mask = u > self.a2
        gaus_mask = ~(left_mask | right_mask)  # Not left or right tail

        log_prob = torch.full_like(u, -torch.inf)
        if self.A1.ndim == 0:
            A1_e = self.A1.expand(u.shape[0],1)
            A2_e = self.A2.expand(u.shape[0],1)
            B1_e = self.B1.expand(u.shape[0],1)
            B2_e = self.B2.expand(u.shape[0],1)
            p1_e = self.p1.expand(u.shape[0],1)
            p2_e = self.p2.expand(u.shape[0],1)
            n_e = self.log_normalization.expand(u.shape[0],1)
        else:
            A1_e = self.A1.expand(u.shape[0],-1)
            A2_e = self.A2.expand(u.shape[0],-1)
            B1_e = self.B1.expand(u.shape[0],-1)
            B2_e = self.B2.expand(u.shape[0],-1)
            p1_e = self.p1.expand(u.shape[0],-1)
            p2_e = self.p2.expand(u.shape[0],-1)
            n_e = self.log_normalization.expand(u.shape[0],-1)

        log_prob[left_mask] =  n_e[left_mask] + torch.log(A1_e[left_mask]+1e-8) + torch.log(torch.pow(B1_e[left_mask] - u[left_mask], -p1_e[left_mask]) +1e-8)
        log_prob[right_mask] = n_e[right_mask] + torch.log(A2_e[right_mask]+1e-8) + torch.log(torch.pow(B2_e[right_mask] + u[right_mask], -p2_e[right_mask])+1e-8)
        log_prob[gaus_mask] =  n_e[gaus_mask] - (u[gaus_mask]**2 / 2)

        return log_prob

    
    def rsample(self, sample_shape, n_samples_cdf_inversion=40):
        """
        Generates random samples from the distribution.  This uses inverse transform sampling.

        Args:
            sample_shape (torch.Size): Shape of the samples to generate.

        Returns:
            torch.Tensor: Random samples from the distribution.
        """
        batch_size = sample_shape
        u = torch.rand(( batch_size, self.batch_shape), device=self.mu.device)  # Use device of parameters
        #samples = torch.empty_like(u, dtype=self.mu.dtype)
        # interpolating CDF
        x_values = torch.linspace(self.xmin, self.xmax, n_samples_cdf_inversion).to(self.mu.device) # On the GPU
        cdf_values = self.cdf(x_values) # Values from CDF, also on GPU
        x_values = x_values[:,None].expand(-1, self.batch_shape)
        if self.batch_shape == 1:
            cdf_values = cdf_values.unsqueeze(1)
        # searchsorted needs the last dimension to match, 
        samples = interpolate_gpu_batched(cdf_values.T, x_values.T, u.T)
        return samples.T

     
    @property
    def batch_shape(self):
        if self.mu.ndim == 0:
            return 1
        else:
            return self.mu.shape[0]

    @property
    def event_shape(self):
        return torch.Size([])