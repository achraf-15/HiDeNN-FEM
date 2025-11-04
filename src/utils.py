import torch
import numpy as np

def interval_gauss_points(order=1, device=None, dtype=torch.float32):
    """
    Returns Gauss-Legendre quadrature points and weights on [0,1].
    """
    xi, wi = np.polynomial.legendre.leggauss(order)
    xi = torch.tensor(xi, dtype=dtype, device=device)
    wi = torch.tensor(wi, dtype=dtype, device=device)
    return xi, wi

def triangle_gauss_points(order=1, device=None, dtype=torch.float32):
    """
    Returns quadrature points (r,s) and weights on standard triangle (0,0),(1,0),(0,1)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if order == 1:
        # 1-point rule (centroid)
        rs = torch.tensor([[1/3, 1/3]], dtype=dtype, device=device)
        w = torch.tensor([0.5], dtype=dtype, device=device)

    elif order == 3:
        # 3-point rule (mid-edge)
        a = 1/6
        rs = torch.tensor([[a, a], [4*a, a], [a, 4*a]], dtype=dtype, device=device)
        w = torch.tensor([1/6, 1/6, 1/6], dtype=dtype, device=device)

    elif order == 4:
        # 4-point rule
        rs = torch.tensor([
            [1/3, 1/3],
            [0.6, 0.2],
            [0.2, 0.6],
            [0.2, 0.2],
        ], dtype=dtype, device=device)
        w = 0.5 * torch.tensor([-27/96, 25/96, 25/96, 25/96], dtype=dtype, device=device)

    elif order == 6:
        # 6-point rule (Dunavant 6-point)
        a = 0.445948490915965
        b = 0.091576213509771
        w1 = 0.111690794839005
        w2 = 0.054975871827661
        rs = torch.tensor([
            [a, a],
            [1 - 2*a, a],
            [a, 1 - 2*a],
            [b, b],
            [1 - 2*b, b],
            [b, 1 - 2*b],
        ], dtype=dtype, device=device)
        w = 0.5 * torch.tensor([w1, w1, w1, w2, w2, w2], dtype=dtype, device=device)

    elif order == 7:
        # 7-point rule (Dunavant 7-point)
        rs = torch.tensor([
            [1/3, 1/3],
            [0.0597158717, 0.4701420641],
            [0.4701420641, 0.0597158717],
            [0.4701420641, 0.4701420641],
            [0.7974269853, 0.1012865073],
            [0.1012865073, 0.7974269853],
            [0.1012865073, 0.1012865073],
        ], dtype=dtype, device=device)
        w = 0.5 * torch.tensor([
            0.225,
            0.1323941527,
            0.1323941527,
            0.1323941527,
            0.1259391805,
            0.1259391805,
            0.1259391805,
        ], dtype=dtype, device=device)

    else:
        raise NotImplementedError("Supported orders: 1, 3, 4, 6, 7")

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