from typing import Callable, Optional
import torch

from utils import triangle_gauss_points, interval_gauss_points

class EnergyLoss2D:
    def __init__(
        self,
        E: float = 10e9,
        nu: float = 0.3,
        length: float = 1.0,
        height: float = 1.0,
        gauss_order: int = 4,
        gauss_order_1d: int = 2,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ):
        self.E = E
        self.nu = nu
        self.length = length
        self.height = height
        self.gauss_order = gauss_order
        self.gauss_order_1d = gauss_order_1d

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # Plane-stress constitutive matrix
        factor = E / (1 - nu**2)
        self.C = torch.tensor([[1.0, nu, 0.0],
                               [nu, 1.0, 0.0],
                               [0.0, 0.0, (1.0-nu)/2.0]], dtype=dtype, device=self.device) * factor

        # Precompute domain Gauss points
        self.xg, self.wg = triangle_gauss_points(order=self.gauss_order, device=self.device, dtype=self.dtype)
        self.ng = self.xg.shape[0]

        # Precompute 1D edge Gauss points
        self.xg_1d, self.wg_1d = interval_gauss_points(order=self.gauss_order_1d, device=self.device, dtype=self.dtype)
        self.ng1 = self.xg_1d.shape[0]

    # Default uniform forces
    def uniform_body_force(self, x: torch.Tensor) -> torch.Tensor:
        # Zero body force by default
        return torch.zeros_like(x)

    def uniform_edge_force(self, x: torch.Tensor, L: float = 1.0, F_total: float = 100e3) -> torch.Tensor:
        # Uniform traction in +x direction
        t_x = torch.full((x.shape[0],), F_total/L, device=x.device, dtype=x.dtype)
        t_y = torch.zeros_like(t_x)
        return torch.stack([t_x, t_y], dim=1)


    # Domain contribution
    def domain_energy(self, model, b_force: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> torch.Tensor:
        n_elem = model.Nelems
        #M = n_elem * self.ng

        # Expand Gauss points to all elements
        x_eval = self.xg.unsqueeze(0).expand(n_elem, self.ng, 2).reshape(-1, 2).to(self.device, dtype=self.dtype)
        elem_id = torch.arange(n_elem, device=self.device).unsqueeze(1).repeat(1, self.ng).reshape(-1)
        wg_flat = self.wg.unsqueeze(0).repeat(n_elem, 1).reshape(-1).to(self.device, dtype=self.dtype)

        # Evaluate displacement and gradients
        u_eval, detJ, grad_u = model(x_eval, elem_id)  # [M,2], [M], [M,2,2]
        grad_u_x = grad_u[:, 0, :]
        grad_u_y = grad_u[:, 1, :]

        # Strain components (infinitesimal)
        eps_xx = grad_u_x[:, 0]
        eps_yy = grad_u_y[:, 1]
        eps_xy = 0.5 * (grad_u_x[:, 1] + grad_u_y[:, 0])
        eps_voigt = torch.stack([eps_xx, eps_yy, 2*eps_xy], dim=1)

        # Stress
        sigma_voigt = eps_voigt @ self.C.T
        elastic_density = 0.5 * torch.sum(eps_voigt * sigma_voigt, dim=1)

        # Body force contribution
        b_vec = b_force(x_eval) if b_force is not None else self.uniform_body_force(x_eval)
        body_density = torch.sum(b_vec * u_eval, dim=1)

        # Domain integration
        quad_weights = wg_flat * detJ.abs()
        domain_energy = torch.sum(quad_weights * elastic_density)
        body_work = torch.sum(quad_weights * body_density)

        return domain_energy - body_work

    # Neumann edge contribution
    def edge_energy(self, model, t_force: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> torch.Tensor:
        x_i, x_ip1 = model.nm_edges[:]  # [N_edges,2]
        N_edges = model.N_edges

        # Map 1D Gauss -> physical points
        xq = (1.0 - self.xg_1d[None,:,None]) * x_i[:,None,:] + self.xg_1d[None,:,None] * x_ip1[:,None,:]
        xq_flat = xq.reshape(-1, 2)
        wq_flat = self.wg_1d[None,:].expand(N_edges, self.ng1).reshape(-1)

        # Edge evaluation
        x_eval = self.xg_1d[None,:].expand(N_edges, self.ng1).reshape(-1,1)
        edge_id = torch.repeat_interleave(torch.arange(N_edges, device=self.device), repeats=self.ng1)
        u_edge, ds = model(x_eval, edge_id, edge=True)

        # Traction
        t_edge = t_force(xq_flat) if t_force is not None else self.uniform_edge_force(xq_flat)

        # Weighted contribution
        w_edge = wq_flat * ds
        return torch.sum((u_edge * t_edge).sum(dim=1) * w_edge)

    # Full total potential
    def __call__(self, model, b_force=None, t_force=None) -> torch.Tensor:
        domain = self.domain_energy(model, b_force)
        edge = self.edge_energy(model, t_force)
        return domain - edge