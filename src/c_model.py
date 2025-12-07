import torch
import torch.nn as nn

from cuda_kernel.cuda_functions import SolveMasked
    
    
class ConnectivityWrapper:
    def __init__(self, coords, connectivity):
        self.coords = coords  # [Nnodes, 2]
        self.connectivity = connectivity # [N_elem, 3]

    def __getitem__(self, idx):
        # Support slicing and single index
        return self.coords[self.connectivity[idx]]

    def __len__(self):
        return self.connectivity.shape[0]
    
class PatchWrapper:
    def __init__(self, coords: torch.Tensor, connectivity: torch.Tensor,
                 patch_safe: torch.Tensor, patch_mask: torch.Tensor):
        self.coords = coords                # [Nnodes, 2]
        self.connectivity = connectivity    # [Ne, 3]
        self.patch_safe = patch_safe        # [Nnodes, n_patch] (safe indices)
        self.patch_mask = patch_mask        # [Nnodes, n_patch] (bool)

    def __getitem__(self, idx: int):
        elem_nodes = self.connectivity[idx]             # [3]
        patch_idx = self.patch_safe[elem_nodes]         # [3, n_patch]
        mask = self.patch_mask[elem_nodes]              # [3, n_patch] (bool)
        coords_patch = self.coords[patch_idx]           # [3, n_patch, 2]
        # Do NOT attempt to fill zeros here; consumer will use mask to ignore padded entries.
        return coords_patch, mask, patch_idx

    def __len__(self):
        return self.connectivity.shape[0]



class c_HiDeNN(nn.Module):
    def __init__(self, node_coords, connectivity, patch, boundary_mask=None, dirichlet_mask=None, u_fixed=None, neumann_edges=None):
        super().__init__()

        self.dim_u = 2

        self.alpha = 0.2

        self.register_buffer("initial_node_coords", node_coords.clone())   # [N,2]
        self.Nnodes = node_coords.shape[0] #N

        # connectivity
        self.register_buffer("connectivity", connectivity.long().clone())  # [Ne,3]
        self.Nelems = connectivity.shape[0] #Ne

        # patches
        patch_raw = patch.long().clone()
        self.register_buffer("patch_raw", patch_raw)

        # precompute safe indices and mask once
        patch_mask = patch_raw >= 0                    # bool [Nnodes, n_patch]
        patch_safe = patch_raw.clone()
        patch_safe[~patch_mask] = 0                    # safe index for torch indexing

        self.register_buffer("patch_safe", patch_safe) # long
        self.register_buffer("patch_mask", patch_mask) # bool

        self.m_patch = 6
        self.n_patch = self.patch_mask.shape[1]

        # boundary mask
        if boundary_mask is None:
            boundary_mask = torch.zeros(self.Nnodes, dtype=torch.bool)
        self.register_buffer("boundary_mask", boundary_mask.clone())

        free_mask = ~boundary_mask  
        self.node_coords_free = nn.Parameter(node_coords[free_mask])
        self.register_buffer("node_coords_fixed", node_coords[boundary_mask])
        self.register_buffer("free_mask", free_mask)

        # Dirichlet mask
        if dirichlet_mask is None:
            dirichlet_mask = torch.zeros(self.Nnodes, dtype=torch.bool)
        self.register_buffer("dirichlet_mask", dirichlet_mask.clone())

        u_free_mask = ~dirichlet_mask  
        self.register_buffer("u_free_mask", u_free_mask)

        # nodal DOFs
        self.u_free = nn.Parameter(torch.randn(u_free_mask.sum().item(), self.dim_u, dtype=self.dtype))
        if u_fixed is not None:
            u_fixed = torch.tensor(u_fixed, dtype=self.dtype)
            self.register_buffer("u_fixed", u_fixed)

        # Neumann mask
        if neumann_edges is not None:
            self.register_buffer("neumann_edges", neumann_edges) # [N_edges, 2]
            self.N_edges = neumann_edges.shape[0]

        # Custom CUDA inv function
        self.solve_masked = SolveMasked()

    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def coords(self):
        coords = torch.zeros_like(self.initial_node_coords, device=self.device, dtype=self.dtype)
        coords[self.free_mask] = self.node_coords_free
        coords[self.boundary_mask] = self.node_coords_fixed
        return coords
        
    @property
    def values(self):
        u = torch.zeros(self.Nnodes, self.dim_u, device=self.device, dtype=self.dtype)
        u[self.u_free_mask] = self.u_free
        if self.u_fixed is not None:
            u[self.dirichlet_mask] = self.u_fixed
        return u
    
    def freeze_coords(self, freeze: bool = True):
        """Freeze or unfreeze node coordinates."""
        self.node_coords_free.requires_grad_(not freeze)
        return self  # allows chaining 

    def freeze_u(self, freeze: bool = True):
        """Freeze or unfreeze u parameters."""
        self.u_free.requires_grad_(not freeze)
        return self  # allows chaining 
    # We can add a context manager to allow for the "with model.eval(): or something similar"  

    @property
    def element_nodes(self):
        return ConnectivityWrapper(self.coords, self.connectivity)
    
    @property
    def edge_nodes(self):
        return ConnectivityWrapper(self.coords, self.neumann_edges)
    
    @property
    def element_patch(self):
        return PatchWrapper(self.coords, self.connectivity, self.patch_safe, self.patch_mask)
    
    @property
    def edge_patch(self):
        return PatchWrapper(self.coords, self.neumann_edges, self.patch_safe, self.patch_mask)
        
            
    def forward(self, x_eval, elem_id, edge=False):
        if not edge:
            # --- 2D triangle / domain ---
            # Gather the 3 node coordinates per element
            coords_elem = self.element_nodes[elem_id] 
            
            # x_eval is assumed in reference triangle coordinates (xi, eta)
            xi = x_eval[:, 0:1]  # [M,1]
            eta = x_eval[:, 1:2]  # [M,1]
            zeta = 1.0 - xi - eta
            # Shape function weights = barycentric coordinates
            N = torch.cat([xi, eta, zeta], dim=1)  # [M,3]

            # physical coordinates
            x_physical = torch.sum(N.unsqueeze(-1) * coords_elem, dim=1)  # [M,2]

            Jinv = self.Jinv[elem_id]
            detJ = self.detJ[elem_id]

            # Shape function derivatives w.r.t local coords (ξ, η) # [2 local derivatives, 3 nodes]
            dN_dxi = torch.tensor([[1., 0., -1.],
                                   [0., 1., -1.]], device=self.device, dtype=self.dtype)  # [2,3]

            # Derivatives in physical coords: dN_dx = J^-1 * dN_dxi
            dN_dx = torch.einsum("mij,jk->mik", Jinv, dN_dxi)  # [M,2,3]


            # Patch coordinates
            coords_patch, patch_mask_elem, patch_idx_elem = self.element_patch[elem_id]  # coords_patch: [M,3,n_patch,2], patch_mask_elem [M,3,n_patch], patch_idx_elem [M,3,n_patch]

            # # ---- Patch derivatives Debug (c-HiDeNN) ----
            # self._check_patch_derivatives(elem_id, x_physical, coords_patch, patch_mask_elem)

            # Compute radial basis
            R_vector, dR_dx = self.compute_patch_radials(x_physical, coords_patch, patch_mask_elem)
            # Compute polynomial basis
            P_vector, dP_dx = self.compute_patch_polynomials(x_physical, coords_patch, patch_mask_elem)

            # Gather the patch nodal u values per element:
            u_patch = self.values[patch_idx_elem]  # [M,3,n_patch, dim_u]
            u_patch = u_patch * patch_mask_elem[..., None]   # zero out padded nodes


            # Solve patch weights: W [M,3,n_patch], dW_dx [M,3,n_patch,2]
            W, dW_dx = self.solve_patch_weights(elem_id, R_vector, P_vector, dR_dx, dP_dx, patch_mask_elem)

            # # ---- Partition of Unity Debug (c-HiDeNN) ----
            # self._check_partition_of_unit(W, N)

            # W: [M,3,n_patch], u_patch: [M,3,n_patch,dim_u] -> sum_j W_ij * u_j 
            Wu = torch.einsum('mij,mijd->mid', W, u_patch) #[M,3,dim_u] 
            dWu_dx = torch.einsum('mijk,mijd->midk', dW_dx, u_patch)  # [M,3,dim_u,2]


            # N: [M,3], Wu: [M,3,dim_u] -> -> sum_i N_i * Wu_i 
            u_h = torch.einsum('mi,mid->md', N, Wu) # [M,dim_u]

            # First derivative term
            grad_term1 = torch.einsum('mij,mjd->mid', dN_dx, Wu).transpose(1, 2)  # [M,dim_u,2]
            # First derivative term
            grad_term2 = torch.einsum('mi,midk->mdk', N, dWu_dx)  # [M,dim_u,2]
            #grad_term2 = torch.einsum('mijk,mijd->mdk', dW_dx, u_patch)  # [M,3,dim_u,2]
            # grad_u: [M,2,2]  (rows=u components, cols=∂/∂x,∂/∂y)
            grad_u = grad_term1 + grad_term2  # [M,dim_u,2]

            return u_h, detJ, grad_u
        
        else:
            # --- 1D edge / Neumann ---
            # Get the two physical nodes of each edge
            coords_edge = self.edge_nodes[elem_id] 
            x_i   = coords_edge[:, 0, :]   # shape [M, 2]
            x_ip1 = coords_edge[:, 1, :]   # shape [M, 2]

            # x_eval: [M,1] in reference edge coordinates ξ ∈ [0,1]
            xi = x_eval[:, 0:1]  # [M,1]
            N = torch.cat([1.0 - xi, xi], dim=1)  # linear shape functions for 2 nodes

            # physical coordinates
            x_physical = torch.sum(N.unsqueeze(-1) * coords_edge, dim=1)  # [M,2]

            # Compute 1D Jacobian = edge length
            ds = torch.norm(x_ip1 - x_i, dim=1)  # [M]


            # Patch coordinates
            coords_patch, patch_mask_elem, patch_idx_elem = self.edge_patch[elem_id]  # coords_patch: [M,2,n_patch,2], patch_mask_elem [M,2,n_patch], patch_idx_elem [M,2,n_patch]
            # Compute radial basis
            R_vector, dR_dx = self.compute_patch_radials(x_physical, coords_patch, patch_mask_elem)
            # Compute polynomial basis
            P_vector, dP_dx = self.compute_patch_polynomials(x_physical, coords_patch, patch_mask_elem, edge=edge)

            # Gather the patch nodal u values per element:
            u_patch = self.values[patch_idx_elem]  # [M,2,n_patch, dim_u]
            u_patch = u_patch * patch_mask_elem[..., None]   # zero out padded nodes


            # Solve patch weights: W [M,2,n_patch]
            W, _ = self.solve_patch_weights(elem_id, R_vector, P_vector, dR_dx, dP_dx, patch_mask_elem, edge=edge)

            # W: [M,3,n_patch], u_patch: [M,2,n_patch,dim_u] -> sum_j W_ij * u_j 
            Wu = torch.einsum('mij,mijd->mid', W, u_patch) #[M,2,dim_u,2]


            #N: [M,2], Wu: [M,2,dim_u] -> -> sum_i N_i * Wu_i 
            u_h = torch.einsum('mi,mid->md', N, Wu) # [M,dim_u]

            return u_h, ds
        

    def precompute_Jaccobians(self):
        elem_id = torch.arange(self.Nelems, device=self.device) 
        # Gather the 3 node coordinates per element
        coords_elem = self.element_nodes[elem_id]

        # Compute 2x2 Jacobian for area / quadrature mapping using all 3 nodes
        v0 = coords_elem[:, 0, :]  # [Nelems,2]
        v1 = coords_elem[:, 1, :]
        v2 = coords_elem[:, 2, :]
        Jmat = torch.stack([v0 - v2, v1 - v2], dim=2)  # [Nelems,2,2]
        detJ = torch.linalg.det(Jmat)  # Determinent Jacobian: [Nelems]
        Jinv = torch.linalg.inv(Jmat)  # Inverse Jacobian: [Nelems, 2, 2]

        # Save Jaccobian inverse et determinent
        self.register_buffer("Jinv", Jinv)
        self.register_buffer("detJ", detJ)        

    def precompute_G_patch(self):
        # --- 2D triangle / domain ---
        node_per_elem = 3
        elem_id = torch.arange(self.Nelems, device=self.device)
        coords_patch, patch_mask_elem, _ = self.element_patch[elem_id]  # coords_patch: [Nelems,3,n_patch,2], patch_mask_elem [Nelems,3,n_patch], patch_idx_elem [Nelems,3,n_patch]
        
        # Compute and save G_patch 
        G = self._compute_G(self.Nelems, node_per_elem, coords_patch, patch_mask_elem)
        self.register_buffer("G_patch"+str(node_per_elem), G)
        self.register_buffer("patch_mask_elem"+str(node_per_elem), patch_mask_elem)

        # --- 1D edge / Neumann ---
        node_per_elem = 2
        elem_id = torch.arange(self.N_edges, device=self.device)
        coords_patch, patch_mask_elem, _ = self.edge_patch[elem_id]  # coords_patch: [N_edges,2,n_patch,2], patch_mask_elem [N_edges,2,n_patch], patch_idx_elem [N_edges,2,n_patch]
        
        # Compute and save G_patch 
        G_edge = self._compute_G(self.N_edges, node_per_elem, coords_patch, patch_mask_elem)
        self.register_buffer("G_patch"+str(node_per_elem), G_edge)
        self.register_buffer("patch_mask_elem"+str(node_per_elem), patch_mask_elem)

    def _compute_G(self, Nelems, node_per_elem, coords_patch, patch_mask_elem):

        # 2D mask for pairwise: valid if both patch entries are valid
        mask_mom_valid = patch_mask_elem.unsqueeze(-1) & patch_mask_elem.unsqueeze(-2)  # [Nelems,node_per_elem,n_patch,n_patch]
        
        # --- r_moments: pairwise distances between patch nodes ---
        diff = coords_patch.unsqueeze(3) - coords_patch.unsqueeze(2)  # [Nelems,node_per_elem,n_patch,n_patch,2]
        r_moments = torch.norm(diff, dim=-1)                           # [Nelems,node_per_elem,n_patch,n_patch]

        # --- R_moments: cubic spline ---
        s_mom = r_moments / self.alpha
        mask_mom = (s_mom <= 1) & mask_mom_valid # Active support: s <= 1 and patch mask

        # Wendland C2 phi(s) = (1-s)^4 (4s+1)
        R_moments = torch.zeros_like(r_moments)
        #R_moments[mask_mom] = (1 - s_mom[mask_mom])**2 * (1 + 2 * s_mom[mask_mom])
        R_moments[mask_mom] = (1 - s_mom[mask_mom])**4 * (4 * s_mom[mask_mom] + 1)


        # --- P_moments on patch nodes ---
        x_patch = coords_patch[..., 0]  # [Nelems,node_per_elem,n_patch]
        y_patch = coords_patch[..., 1]  # [Nelems,node_per_elem,n_patch]

        P_moments = self._polynomial_basis(x_patch, y_patch)

        # Mask out padded nodes (broadcast mask to last dim)
        P_moments = P_moments * patch_mask_elem.unsqueeze(-1).float()  # [Nelems,node_per_elem3,n_patch,m_patch]

        # --- Build block matrix G: [Nelems,node_per_elem,n_patch+m_patch, n_patch+m_patch] ---
        # Upper-left: R_moments
        G_UL = R_moments
        # Upper-right: P_moments
        G_UR = P_moments
        # Lower-left: P_moments^T (transpose last two dims)
        G_LL = P_moments.transpose(-2, -1)
        # Lower-right: zeros [M,3,m_patch,m_patch]
        G_LR = torch.zeros(Nelems, node_per_elem, self.m_patch, self.m_patch, device=self.device, dtype=self.dtype)

        # Concatenate along last dimension
        G_top = torch.cat([G_UL, G_UR], dim=-1)    # [Nelems,node_per_elem,n_patch,n_patch+m_patch]
        G_bottom = torch.cat([G_LL, G_LR], dim=-1) # [Nelems,node_per_elem,m_patch,n_patch+m_patch]
        G = torch.cat([G_top, G_bottom], dim=-2)   # [Nelems,node_per_elem,n_patch+m_patch,n_patch+m_patch]

        return G

    def compute_patch_radials(self, x_physical: torch.Tensor, coords_patch: torch.Tensor, patch_mask_elem: torch.Tensor):

        # --- r_vector: distances from physical points to patch nodes ---
        x_phys_exp = x_physical.unsqueeze(1).unsqueeze(2)  # [M,1,1,2]
        diff_vec = x_phys_exp - coords_patch               # [M,node_per_elem,n_patch,2]
        r_vector = torch.norm(diff_vec, dim=-1)           # [M,node_per_elem,n_patch]

        # --- R_vector: cubic spline ---
        s_vec = r_vector / self.alpha
        mask_vec = (s_vec <= 1) & patch_mask_elem  # Active support: s <= 1 and patch mask

        # Wendland C2 phi(s) = (1-s)^4 (4s+1)
        R_vector = torch.zeros_like(r_vector)
        R_vector[mask_vec] = (1 - s_vec[mask_vec])**4 * (4 * s_vec[mask_vec] + 1)

        # --- derivatives dR_vector/dx_physical ---
        dR_dr = torch.zeros_like(r_vector)
        dR_dr[mask_vec] = -20 * s_vec[mask_vec] * (1 - s_vec[mask_vec])**3 / self.alpha

        eps = 1e-12
        dr_dx = diff_vec / (r_vector.unsqueeze(-1) + eps)  # [M,node_per_elem,n_patch,2]
        dR_vector_dx = dR_dr.unsqueeze(-1) * dr_dx         # [M,node_per_elem,n_patch,2]

        return R_vector, dR_vector_dx
    
    def compute_patch_polynomials(self, x_physical: torch.Tensor, coords_patch: torch.Tensor, patch_mask_elem: torch.Tensor, edge=False):

        # Evaluation points 
        x_eval = x_physical[:, 0:1]  # [M,1]
        y_eval = x_physical[:, 1:2]  # [M,1]

        if edge:
            # Broadcast to element nodes
            x_eval_nodes = x_eval.expand(-1, 2)  # [M,2]
            y_eval_nodes = y_eval.expand(-1, 2)  # [M,2]
        else:
            # Broadcast to element nodes
            x_eval_nodes = x_eval.expand(-1, 3)  # [M,3]
            y_eval_nodes = y_eval.expand(-1, 3)  # [M,3]

        P_vector = self._polynomial_basis(x_eval_nodes, y_eval_nodes)
        dP_dx = self._polynomial_derivatives(x_eval_nodes, y_eval_nodes)

        return P_vector, dP_dx

    def solve_patch_weights(self, elem_id: torch.Tensor,
                        R_vector: torch.Tensor, P_vector: torch.Tensor,
                        dR_dx: torch.Tensor, dP_dx: torch.Tensor,
                        patch_mask_elem: torch.Tensor, edge=False):

        # Build vector b: [M,node_per_elem,n_patch+m_patch]
        b = torch.cat([R_vector, P_vector], dim=-1)  # [M,node_per_elem,n_patch+m_patch]

        # Build derivative vector b: [M,node_per_elem,n_patch+m_patch,2] 
        db_dx = torch.cat([dR_dx, dP_dx], dim=-2)   # [M,node_per_elem,n_patch+m_patch,2]

        # Get precomputed inverse 
        if edge:
            G_mat = self.G_patch2[elem_id]  # [M,2,n_patch+m_patch,n_patch+m_patch]
        else:
            G_mat = self.G_patch3[elem_id]  # [M,3,n_patch+m_patch,n_patch+m_patch]
        
        # Mask first n_patch from input, last m_patch always active
        mask_tail = torch.ones_like(P_vector, dtype=torch.bool)
        mask = torch.cat([patch_mask_elem.bool(), mask_tail], dim=-1)

        # Solve for Weights
        W_tilde = self.solve_masked(G_mat, b, mask)                                         # [M,node_per_elem,n_patch+m_patch]
        dW_tilde = torch.stack([self.solve_masked(G_mat, db_dx[..., 0], mask),              # [M,node_per_elem,n_patch+m_patch,2]
                                self.solve_masked(G_mat, db_dx[..., 1], mask)], dim=-1)

        # Extract patch weights
        W = W_tilde[..., :self.n_patch]          #  [M,node_per_elem,n_patch]
        dW_dx = dW_tilde[..., :self.n_patch, :]  #  [M,node_per_elem,n_patch,2]

        return W, dW_dx
    
    def _polynomial_basis(self, x, y):

        if self.m_patch == 6 :
            return torch.stack([
                        torch.ones_like(x),       # 1
                        x,                        # x
                        y,                        # y
                        x**2,                     # x^2
                        x * y,                    # x*y
                        y**2                      # y^2
                    ], dim=-1) 

        if self.m_patch == 3:
            return torch.stack([
                        torch.ones_like(x),       # 1
                        x,                        # x
                        y,                        # y
                    ], dim=-1) 
        
    def _polynomial_derivatives(self, x, y):

        if self.m_patch == 6 :
            return torch.stack([
                        torch.stack([torch.zeros_like(x), torch.zeros_like(y)], dim=-1),          # derivative of 1
                        torch.stack([torch.ones_like(x),  torch.zeros_like(y)], dim=-1),          # derivative of x
                        torch.stack([torch.zeros_like(x), torch.ones_like(y)], dim=-1),           # derivative of y
                        torch.stack([2*x,                 torch.zeros_like(y)], dim=-1),          # derivative of x^2
                        torch.stack([y,                   x], dim=-1),                            # derivative of x*y
                        torch.stack([torch.zeros_like(x), 2*y], dim=-1)                           # derivative of y^2
                    ], dim=2)

        if self.m_patch == 3:
            return torch.stack([
                        torch.stack([torch.zeros_like(x), torch.zeros_like(y)], dim=-1),          # derivative of 1
                        torch.stack([torch.ones_like(x),  torch.zeros_like(y)], dim=-1),          # derivative of x
                        torch.stack([torch.zeros_like(x), torch.ones_like(y)], dim=-1),           # derivative of y
                    ], dim=2)

    @staticmethod
    def _check_partition_of_unit(W, N):
        # ---- Partition of Unity Debug (c-HiDeNN) ----
        with torch.no_grad():
            # W: [M, 3, n_patch]
            # N: [M, 3]
            Weff = torch.einsum("m l p, m l -> m p", W, N)  # shape [M, n_patch]

            sum_p = Weff.sum(dim=-1)  # sum across patch functions

            max_dev = (sum_p - 1).abs().max()
            bad = (sum_p - 1).abs() > 1e-4

        print(f"[DEBUG][PoU] max deviation: {max_dev.item():.6e} | "
            f"bad nodes: {bad.sum().item()}/{sum_p.numel()}")
        # ----------------------------------------------

    def _check_patch_derivatives(self, elem_id, x_physical, coords_patch, patch_mask_elem, edge=False):

        eps = 1e-8 if self.dtype == torch.float64 else 1e-4
        # ========================= RADIAL GRADIENT CHECK (FINITE DIFF) =========================
        print("================ FINITE-DIFF RADIAL GRADIENT DEBUG ================")
        ## Test radial derivatives dR_dx with finite differences
        R0, dR_dx = self.compute_patch_radials(x_physical, coords_patch, patch_mask_elem)  
        # R0: [M, node_per_elem, n_patch]
        # dR_dx: [M, node_per_elem, n_patch, 2]

        # numerical check
        num_dR_dx = torch.zeros_like(dR_dx)
        for d in range(2):
            xp = x_physical.clone()
            xm = x_physical.clone()
            xp[:, d] += eps
            xm[:, d] -= eps
            Rp, _ = self.compute_patch_radials(xp, coords_patch, patch_mask_elem)
            Rm, _ = self.compute_patch_radials(xm, coords_patch, patch_mask_elem)
            num_dR_dx[..., d] = (Rp - Rm) / (2*eps)

        err = (num_dR_dx - dR_dx).abs()
        print("Radial derivative max error:", err.max().item())
        print("Radial derivative mean error:", err.mean().item())

        # ======================= POLYNOMIAL GRADIENT CHECK (FINITE DIFF) =======================
        print("\n============== FINITE-DIFF POLYNOMIAL GRADIENT DEBUG ==============")
        ## Test polynomial derivatives dp_dx with finite differences
        P0, dP_dx = self.compute_patch_polynomials(x_physical, coords_patch, patch_mask_elem, edge)  
        # P0: [M,node_per_elem,m_patch], 
        # dP_dx: [M,node_per_elem,m_patch,2]]

        # numerical check
        num_dP_dx = torch.zeros_like(dP_dx)
        for d in range(2):
            xp = x_physical.clone()
            xm = x_physical.clone()
            xp[:, d] += eps
            xm[:, d] -= eps
            Pp, _ = self.compute_patch_polynomials(xp, coords_patch, patch_mask_elem, edge)
            Pm, _ = self.compute_patch_polynomials(xm, coords_patch, patch_mask_elem, edge)
            num_dP_dx[..., d] = (Pp - Pm) / (2*eps)

        errP = (num_dP_dx - dP_dx).abs()
        print("Polynomial derivative max error:", errP.max().item())
        print("Polynomial derivative mean error:", errP.mean().item())

        # ======================= PATCH GRADIENT CHECK (FINITE DIFF) =======================
        print("\n================ FINITE-DIFF PATCH GRADIENT DEBUG =================")
        ## Test patch derivatives dW_dx with finite differences
        W, dW_dx = self.solve_patch_weights(elem_id, R0, P0, dR_dx, dP_dx, patch_mask_elem, edge)
        # PW: [M,node_per_elem,n_patch], 
        # dW_dx: [M,node_per_elem,n_patch,2]

        num_dW_dx = torch.zeros_like(dW_dx)
        for d in range(2):
            x_perturb = x_physical.clone()
            x_perturb[:, d] += eps
            R_p, dR_p = self.compute_patch_radials(x_perturb, coords_patch, patch_mask_elem)
            P_p, dP_p = self.compute_patch_polynomials(x_perturb, coords_patch, patch_mask_elem)
            W_p, _ = self.solve_patch_weights(elem_id, R_p, P_p, dR_p, dP_p, patch_mask_elem)
            
            x_perturb[:, d] -= 2*eps
            R_m, dR_m = self.compute_patch_radials(x_perturb, coords_patch, patch_mask_elem)
            P_m, dP_m = self.compute_patch_polynomials(x_perturb, coords_patch, patch_mask_elem)
            W_m, _ = self.solve_patch_weights(elem_id, R_m, P_m, dR_m, dP_m, patch_mask_elem)
            
            num_dW_dx[..., d] = (W_p - W_m)/(2*eps)

        errW = (num_dW_dx - dW_dx).abs()
        print("Patch derivative max error:", errW.max().item())
        print("Patch derivative mean error:", errW.mean().item())

        print("===================================================================\n")

    
    def check_stability(self):
        # Debug element patch matrix
        mat = self.G_patch3
        mask = self.patch_mask_elem3
        print("Element patch matrices quality:")
        self._compute_matrix_stability(mat, mask)

        # Debug edge patch matrix
        mat = self.G_patch2
        mask = self.patch_mask_elem2
        print("Edge patch matrices quality:")
        self._compute_matrix_stability(mat, mask)

    def _compute_matrix_stability(self, mat, mask):
        Nelems, node_per_elem, D, _ = mat.shape

        cond_list = []
        eig_min_list = []
        eig_max_list = []

        # --- Compute conditioning and eigenvalues for valid submatrices
        for e in range(Nelems):
            for n in range(node_per_elem):
                # Determine indices of valid entries
                radial_idx = torch.nonzero(mask[e, n], as_tuple=True)[0]
                poly_idx = torch.arange(self.n_patch, self.n_patch + self.m_patch, device=self.device)
                valid_idx = torch.cat([radial_idx, poly_idx])
                k = valid_idx.numel()
                if k == 0:
                    continue

                sub_mat = mat[e, n][valid_idx][:, valid_idx]
                sub_cond = torch.linalg.cond(sub_mat)
                cond_list.append(sub_cond)

                sub_eig = torch.linalg.eigvalsh(sub_mat)
                eig_min_list.append(sub_eig[0])
                eig_max_list.append(sub_eig[-1])

        cond_tensor = torch.tensor(cond_list, device=self.device)
        eig_min_tensor = torch.tensor(eig_min_list, device=self.device)
        eig_max_tensor = torch.tensor(eig_max_list, device=self.device)

        # Print statistics
        print("Condition number stats (valid submatrices):")
        print(f" min : {cond_tensor.min().item():.2e}")
        print(f" median: {cond_tensor.median().item():.2e}")
        print(f" max : {cond_tensor.max().item():.2e}")

        # Identify degenerate patches
        threshold = 1e10 if self.dtype == torch.float64 else 1e4
        num_bad = (cond_tensor > threshold).sum().item()
        print(f"Ill-conditioned patches: {num_bad} / {cond_tensor.numel()}")
        print('-----'*5)

        # --- Symmetry check on full matrices
        sym_err = (mat - mat.transpose(-1, -2)).abs().max()
        print("Symmetry stats:")
        print(f" Max asymmetry:{sym_err.item():.2e}")
        print(f" eig_max : [{eig_max_tensor.min().item():.2e}; {eig_max_tensor.max().item():.2e}]")
        print(f" eig_min : [{eig_min_tensor.min().item():.2e}; {eig_min_tensor.max().item():.2e}]")
        
        eps = 1e-8
        num_pos_eig = (eig_min_tensor > eps).sum().item()
        print(f"Positive definite patches: {num_pos_eig} / {eig_min_tensor.numel()}")
        print('-----'*5)

        
    def gradient_check_solve(self, atol=1e-5, rtol=1e-3):
        # ---- Setup ----
        node_per_elem = 3
        n_patch = self.n_patch
        m_patch = self.m_patch
        G = self.G_patch3          # [Nelem, node_per_elem, n_total, n_total]
        mask_patch = self.patch_mask_elem3  # [Nelem, node_per_elem, n_patch] boolean
        Nelems, node_per_elem, D, _ = G.shape

        # Create full mask including polynomial tail
        ones_tail = torch.ones(m_patch, dtype=torch.bool, device=self.device)
        mask_full = torch.cat([mask_patch, ones_tail[None, None, :].expand(Nelems, node_per_elem, -1)], dim=-1)
        print("mask_full", mask_full.shape)

        # Random b vector
        b = torch.randn(G.shape[0], node_per_elem, G.shape[-1], device=self.device, dtype=self.dtype, requires_grad=True)
        b = b * mask_full
        b.retain_grad()

        # ---- CUDA masked solver ----
        W_custom = self.solve_masked(G, b, mask_full)
        loss_custom = (W_custom ** 2).sum()
        loss_custom.backward()
        grad_custom = b.grad.detach().clone()

        print("Max |W|:", W_custom.abs().max().item())
        print("Masked entries:", W_custom[~mask_full].sum().item())

        # ---- PyTorch autograd reference ----
        grad_ref = torch.zeros_like(b)
        for e in range(G.shape[0]):
            for n in range(node_per_elem):
                valid_idx = torch.nonzero(mask_full[e, n], as_tuple=True)[0]
                k = valid_idx.numel()
                if k == 0:
                    continue
                # Extract masked block
                A = G[e, n][valid_idx][:, valid_idx]  # [k,k]
                b_block = b[e, n][valid_idx]         # [k]

                # Solve
                W_block = torch.linalg.solve(A, b_block)
                loss_block = (W_block ** 2).sum()

                # Gradients w.r.t. b_block
                grad_block = torch.autograd.grad(loss_block, b_block, retain_graph=True)[0]

                # Scatter back
                grad_ref[e, n].index_put_((valid_idx,), grad_block)

        # ---- Compare ----
        abs_diff = (grad_custom - grad_ref).abs()
        rel_diff = abs_diff / (grad_ref.abs() + 1e-12)

        max_abs = abs_diff.max().item()
        max_rel = rel_diff.max().item()
        mean_abs = abs_diff.mean().item()
        mean_rel = rel_diff.mean().item()

        print("\n--- CUDA Solver Gradient Check ---")
        print(f"Max abs diff : {max_abs:e}")
        print(f"Mean abs diff: {mean_abs:e}")
        print(f"Max rel diff : {max_rel:e}")
        print(f"Mean rel diff: {mean_rel:e}")

        passed = torch.allclose(grad_custom, grad_ref, atol=atol, rtol=rtol)
        print(f"Gradient match: {'✅ PASSED' if passed else '❌ FAILED'}")


    



