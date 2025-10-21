import torch
import numpy as np


def gauss_legendre_points_weights(n, device=None):
    """
    Returns Gauss-Legendre quadrature points and weights on [a,b].
    """
    xi, wi = np.polynomial.legendre.leggauss(n)
    xi = torch.tensor(xi, dtype=torch.float32, device=device)
    wi = torch.tensor(wi, dtype=torch.float32, device=device)
    return xi, wi