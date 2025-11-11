import torch
import numpy as np
import pandas as pd

def interval_gauss_points(order=1, device=None, dtype=torch.float32):
    """
    Returns Gauss-Legendre quadrature points and weights on [0,1].
    """
    xi, wi = np.polynomial.legendre.leggauss(order)
    xi = torch.tensor(xi, dtype=dtype, device=device)
    wi = torch.tensor(wi, dtype=dtype, device=device)
    return xi, wi


def triangle_gauss_points(order, device=None, dtype=torch.float32):
    """
    Returns quadrature points (r,s) and weights on the standard triangle (0,0),(1,0),(0,1)
    using a precomputed Dunavant table loaded from CSV.
    
    Parameters:
        order : int
            Number of points in the Dunavant quadrature.
        device : torch.device
        dtype : torch.dtype
    Returns:
        rs : torch.Tensor [n_points, 2] -> quadrature points
        w  : torch.Tensor [n_points]    -> weights
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the table 
    DUNAVANT_TABLE = pd.read_csv("./duvant_rule/dunavant_triangle_quadrature.csv")
    
    # Filter table by the desired order
    table = DUNAVANT_TABLE[DUNAVANT_TABLE['order'] == order]
    if table.empty:
        raise NotImplementedError(f"Dunavant quadrature of order {order} not in CSV table.")
    
    # Extract points and weights as tensors
    rs = torch.tensor(table[['xi','eta']].values, dtype=dtype, device=device)
    w  = torch.tensor(table['weight'].values, dtype=dtype, device=device)
    
    return rs, w


def test_gradients(model, loss_fn):
    
    # Test u_free gradients
    loss = loss_fn(model)
    loss.backward()
    assert model.u_free.grad is not None
    assert not torch.isnan(model.u_free.grad).any()
    
    # Test node_coords_free gradients
    assert model.node_coords_free.grad is not None
    assert not torch.isnan(model.node_coords_free.grad).any()
    
    print("Gradient magnitudes:")
    print(f"u_free: {model.u_free.grad.norm()}")
    print(f"node_coords: {model.node_coords_free.grad.norm()}")