import torch
from torch import Tensor
from numpy import ndarray

import matplotlib.pyplot as plt


def plot_harmonic_oscillator(
        t_obs: Tensor | ndarray, x_obs: Tensor | ndarray, 
        t: Tensor | ndarray, x: Tensor | ndarray) -> None:
    """
    params: 
        t_obs: observed noisy data points (time)
        x_obs: observed noisy data points (coordinate)
        t: time for predicted/test coordinates
        x: predicted/test coordinate
    """
    plt.figure(figsize=(5, 2.5))
    plt.scatter(t_obs, x_obs,  c='b', s=30, alpha=0.7, label='Observations')
    plt.plot(t, x, label="Solution", color="grey", alpha=0.6)
    plt.grid(True, alpha=0.3)
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.legend(fontsize=9, loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_kepler(
        phi_exact: Tensor, r_exact: Tensor, 
        phi_obs: Tensor, r_obs: Tensor,
        phi_predict = None, r_predict = None,
    ) -> None:
    """
    params:
        phi_exact: [Array] true anomaly (angle) for exact solution, polar c.s.
        r_exact: [Array] rho (distance in polar c.s.) for exact solution
        phi_obs: [Array] observed data points
        r_obs: [Array] observed data points
        phi_predict: [Optional[Array]] angle for predicted trajectory
        r_predict:  [Optional[Array]] predicted trajectory
    """
    fig, axes = plt.subplots(1,2, figsize=(15, 5))
    
    axes[0].plot(phi_exact.numpy(), 1/r_exact.numpy(), 'k-', linewidth=2, label='True u=1/r')
    axes[0].scatter(phi_obs.numpy(), 1/r_obs.numpy(), c='blue', s=30, alpha=0.7, label='Data')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    x_true = r_exact * torch.cos(phi_exact)
    y_true = r_exact * torch.sin(phi_exact) 
    x_obs = r_obs * torch.cos(phi_obs)
    y_obs = r_obs * torch.sin(phi_obs) 
    
    axes[1].plot(x_true.numpy(), y_true.numpy(), 'k-', linewidth=2, label='True')
    axes[1].scatter(0, 0, c='orange', s=200, marker='*', label='Focus')
    axes[1].scatter(x_obs.numpy(), y_obs.numpy(), c='blue', s=30, alpha=0.7, label='Data')
    
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title('Orbit')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')

    axes[0].plot(phi_exact.numpy(), r_exact.numpy(), 'gray', linewidth=2, label='True r')
    axes[0].scatter(phi_obs.numpy(), r_obs.numpy(), c='blue', s=30, alpha=0.3, label='Data r[φ]')
    axes[0].set_xlabel('φ [rad]')
    axes[0].set_title('r(φ) - Prediction/Test vs Observed')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if phi_predict is not None and r_predict is not None:
        axes[0].plot(phi_predict.numpy(), 1/r_predict.numpy(), 'r--', linewidth=2, label='Predicted u=1/r')
         
        x_pred = r_predict * torch.cos(phi_predict)
        y_pred = r_predict * torch.sin(phi_predict) 
        axes[1].plot(x_pred.numpy(), y_pred.numpy(), 'r--', linewidth=2, label='Predicted')
        axes[1].set_title('Prediction vs True')

