from typing import Callable, Optional
import torch

from .utils import triangle_gauss_points, interval_gauss_points

class EnergyLoss2D:
    def __init__(
        self,
        E: float = 10e9,
        nu: float = 0.3,
        length: float = 1.0,
        height: float = 1.0,
        F_total: float = 100e3,
        gauss_order: int = 12,
        gauss_order_1d: int = 3,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ):
        self.E = E
        self.nu = nu
        self.length = length
        self.height = height
        self.F_total = F_total
        self.gauss_order = gauss_order
        self.gauss_order_1d = gauss_order_1d

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # characteristic scales
        self.L0 = self.length                            # length scale
        self.T0 = (self.F_total / self.L0)               # N/m (approx)
        self.U0 = (self.T0 * self.L0) / self.E           # U0 = T0*L0/E
        # Model coordinates/values should be converted to dimeionless as well -> u = u/U0; x = x/L0

        # store scale factors to convert dimensional -> dimensionless
        # b' = b * (L0^2) / (E * U0)
        self._b_scale = (self.L0 ** 2) / (self.E * self.U0)
        # t' = t * L0 / (E * U0)
        self._t_scale = (self.L0) / (self.E * self.U0)

        # Plane-stress constitutive matrix
        factor = E / (1 - nu**2)
        self.C = torch.tensor([[1.0, nu, 0.0],
                               [nu, 1.0, 0.0],
                               [0.0, 0.0, (1.0-nu)/2.0]], dtype=dtype, device=self.device) * factor
        
        # Dimensionless constitutive matrix: C' = C / E
        self.C_dimless = (self.C / self.E)

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

    def uniform_edge_force(self, x: torch.Tensor) -> torch.Tensor:
        # Uniform traction in +x direction
        L = self.height
        F_total = self.F_total
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

        # # Test model gradients
        # self.debug_grad(model, x_eval, elem_id)

        # Evaluate displacement and gradients
        u_eval, detJ, grad_u = model(x_eval, elem_id)  # [M,2], [M], [M,2,2]
        grad_u_x = grad_u[:, 0, :] # ∂u_x/∂(x,y)
        grad_u_y = grad_u[:, 1, :] # ∂u_y/∂(x,y)

        # Strain components (infinitesimal)
        eps_xx = grad_u_x[:, 0]  # ∂u_x/∂x
        eps_yy = grad_u_y[:, 1]  # ∂u_y/∂y
        eps_xy = 0.5 * (grad_u_x[:, 1] + grad_u_y[:, 0])
        eps_voigt = torch.stack([eps_xx, eps_yy, 2*eps_xy], dim=1)

        # Stress
        sigma_voigt = eps_voigt @ self.C_dimless.T
        elastic_density = 0.5 * torch.sum(eps_voigt * sigma_voigt, dim=1)

        # Body force contribution
        b_vec = b_force(x_eval) if b_force is not None else self.uniform_body_force(x_eval)
        # convert to dimensionless
        b_vec_dimless= b_vec * (self._b_scale)    # [M,2] dimensionless body force
        body_density = torch.sum(b_vec_dimless * u_eval, dim=1) # u_eval is nondimensional by model construction

        # Domain integration
        quad_weights = wg_flat * detJ.abs()
        domain_energy = torch.sum(quad_weights * elastic_density)
        body_work = torch.sum(quad_weights * body_density)

        return domain_energy - body_work

    # Neumann edge contribution
    def edge_energy(self, model, t_force: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> torch.Tensor: 
        coords_edge = model.edge_nodes[:]  # [N_edges,2]
        x_i   = coords_edge[:, 0, :]   # shape [N_edges, 2]
        x_ip1 = coords_edge[:, 1, :]   # shape [N_edges, 2]
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
        # convert to dimensionless
        t_edge_dimless = t_edge * (self._t_scale) # [M,2] dimensionless traction force

        # Weighted contribution
        w_edge = wq_flat * ds
        return torch.sum((t_edge_dimless * u_edge).sum(dim=1) * w_edge)

    # Full total potential
    def __call__(self, model, b_force=None, t_force=None) -> torch.Tensor:
        domain = self.domain_energy(model, b_force)
        edge = self.edge_energy(model, t_force)
        return domain - edge
    
    def debug_grad(self, model, x_eval: torch.Tensor, elem_id: torch.Tensor, m_debug: int = 50):
        """
        Debug gradient computation of the c-HiDeNN model by comparing 
        the model's analytic gradients with finite-difference and autograd estimates.
        """

        # Small subset for clarity
        M_debug = min(m_debug, x_eval.shape[0])
        rand_idx = torch.randperm(x_eval.shape[0], device=x_eval.device)[:M_debug]
        x_eval_debug = x_eval[rand_idx].clone()  # do not require grad
        elem_debug = elem_id[rand_idx]

        # Forward on the debug subset
        _, _, grad_model = model(x_eval_debug, elem_debug)  # grad_model: [M,dim_u,2]

        # ========================= DEBUG GRADIENT CHECK (FINITE DIFF) =========================
        print("================ FINITE-DIFF GRADIENT DEBUG ================")
        # Finite differences parameters
        eps = 1e-8 if self.dtype == torch.float64 else 1e-4

        # Placeholder for finite-diff gradient: [M, dim_u, 2]
        grad_fd_xieta  = torch.zeros_like(grad_model)

        for d in range(2):  # ξ, η
            x_perturb_plus = x_eval_debug.clone()
            x_perturb_minus = x_eval_debug.clone()
            x_perturb_plus[:, d] += eps
            x_perturb_minus[:, d] -= eps

            u_plus, _, _ = model(x_perturb_plus, elem_debug)
            u_minus, _, _ = model(x_perturb_minus, elem_debug)

            grad_fd_xieta [:, :, d] = (u_plus - u_minus) / (2 * eps)

        # Map FD reference gradient to physical coordinates
        # ∂u/∂x = ∂u/∂ξ * ∂ξ/∂x + ∂u/∂η * ∂η/∂x
        Jinv_debug = model.Jinv[elem_debug]  # [M,2,2]
        # explicit safe batching: (M,dim_u,2) @ (M,2,2) -> (M,dim_u,2)
        grad_fd_phys = torch.einsum('mdk,mkj->mdj', grad_fd_xieta, Jinv_debug)

        # Compare with model backward
        diff = grad_fd_phys - grad_model

        print("Max |grad_fd - grad_model| =", diff.abs().max().item())
        print("Mean |grad_fd - grad_model| =", diff.abs().mean().item())
        print("Component-wise:")
        print("∂u_x/∂x diff =", diff[:, 0, 0].abs().max().item())
        print("∂u_x/∂y diff =", diff[:, 0, 1].abs().max().item())
        print("∂u_y/∂x diff =", diff[:, 1, 0].abs().max().item())
        print("∂u_y/∂y diff =", diff[:, 1, 1].abs().max().item())

        # ========================= DEBUG GRADIENT CHECK (AUTOGRAD) =========================
        print("\n================ AUTOGRAD GRADIENT DEBUG ================")
        x = x_eval_debug.clone().detach().double().requires_grad_(True)  # [M,2]
        elems = elem_debug

        # autograd jacobian: ∂u/∂(xi,eta)
        from torch.autograd.functional import jacobian
        autograd_list = []
        for i in range(x.shape[0]):
            def single_forward(xi):
                out = model(xi.unsqueeze(0), elems[i:i+1])[0].squeeze(0)   # [dim_u]
                return out
            Ji = jacobian(single_forward, x[i], create_graph=False)      # [dim_u, 2]
            autograd_list.append(Ji)
        autograd_jac = torch.stack(autograd_list, dim=0)  # [M, dim_u, 2]

        # map autograd (ξ,η) -> physical (x,y)
        Jinv_debug = model.Jinv[elems].double()    # [M,2,2]
        autograd_phys = torch.einsum('mdk,mkj->mdj', autograd_jac, Jinv_debug)  # [M,dim_u,2]

        diff = autograd_phys - grad_model
        print("Overall: max |autograd_phys - grad_model| =", (diff).abs().max().item())
        print("Overall: mean |autograd_phys - grad_model| =", (diff).abs().mean().item())
        print("Component-wise:")
        print("∂u_x/∂x diff =", diff[:, 0, 0].abs().max().item())
        print("∂u_x/∂y diff =", diff[:, 0, 1].abs().max().item())
        print("∂u_y/∂x diff =", diff[:, 1, 0].abs().max().item())
        print("∂u_y/∂y diff =", diff[:, 1, 1].abs().max().item())
        # ======================= END AUTOGRAD GRADIENT DEBUG =======================

        print("\n================ GRADIENT DEBUG ================")
        diff = autograd_phys - grad_fd_phys
        print("Overall: max |autograd_phys - grad_model| =", (diff).abs().max().item())
        print("Overall: mean |autograd_phys - grad_model| =", (diff).abs().mean().item())
        print("Component-wise:")
        print("∂u_x/∂x diff =", diff[:, 0, 0].abs().max().item())
        print("∂u_x/∂y diff =", diff[:, 0, 1].abs().max().item())
        print("∂u_y/∂x diff =", diff[:, 1, 0].abs().max().item())
        print("∂u_y/∂y diff =", diff[:, 1, 1].abs().max().item())
        print("===================================================================\n")
        # ======================= END GRADIENT DEBUG =======================