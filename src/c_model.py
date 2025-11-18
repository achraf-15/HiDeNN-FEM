import torch
import torch.nn as nn
import torch.nn.functional as F
    
    
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



class PiecewiseLinearShapeNN2D(nn.Module):
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

        self.m_patch = 3
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
        self.u_free = nn.Parameter(torch.randn(u_free_mask.sum().item(), self.dim_u))
        if u_fixed is not None:
            u_fixed = torch.tensor(u_fixed)
            self.register_buffer("u_fixed", u_fixed)

        # Neumann mask
        if neumann_edges is not None:
            self.register_buffer("neumann_edges", neumann_edges) # [N_edges, 2]
            self.N_edges = neumann_edges.shape[0]

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

            Jinv = self.Jinv[elem_id]
            detJ = self.detJ[elem_id]

            # Shape function derivatives w.r.t local coords (ξ, η) # [2 local derivatives, 3 nodes]
            dN_dxi = torch.tensor([[1., 0., -1.],
                                   [0., 1., -1.]], device=self.device, dtype=self.dtype)  # [2,3]

            # Derivatives in physical coords: dN_dx = J^-1 * dN_dxi
            dN_dx = torch.einsum("mij,jk->mik", Jinv, dN_dxi)  # [M,2,3]

            # physical coordinates
            x_physical = torch.sum(N.unsqueeze(-1) * coords_elem, dim=1)  # [M,2]

            # Patch coordinates
            coords_patch, patch_mask_elem, patch_idx_elem = self.element_patch[elem_id]  # coords_patch: [M,3,n_patch,2], patch_mask_elem [M,3,n_patch], patch_idx_elem [M,3,n_patch]
            # Compute radial basis
            R_vector, dR_dx = self.compute_patch_radials(x_physical, coords_patch, patch_mask_elem)
            # Compute polynomial basis
            P_vector, dP_dx = self.compute_patch_polynomials(x_physical, coords_patch, patch_mask_elem)

            # Gather the patch nodal u values per element:
            u_patch = self.values[patch_idx_elem]  # [M,3,n_patch, dim_u]
            u_patch = u_patch * patch_mask_elem[..., None]   # zero out padded nodes


            # # Solve patch weights
            # W, dW_dx = self.solve_patch_weights(elem_id, R_vector, P_vector, dR_dx, dP_dx)
            # # Compute patch weights
            # W, dW_dx = self.compute_patch_weights(elem_id, R_vector, P_vector, dR_dx, dP_dx) # G_inv must be precomputed

            # # W: [M,3,n_patch], u_patch: [M,3,n_patch,dim_u] -> sum_j W_ij * u_j 
            # Wu = torch.einsum('mij,mijd->mid', W, u_patch) #[M,3,dim_u] 

            # # N: [M,3], Wu: [M,3,dim_u] -> -> sum_i N_i * Wu_i 
            # u_h = torch.einsum('mi,mid->md', N, Wu) # [M,dim_u]

            # # First derivative term
            # grad_term1 = torch.einsum('mij,mjd->mid', dN_dx, Wu).transpose(1, 2)  # [M,dim_u,2]
            # # First derivative term
            # grad_term2 = torch.einsum('mijc,mijk->mck', u_patch, dW_dx)  # [M,dim_u,2]
            # # grad_u: [M,2,2]  (rows=u components, cols=∂/∂x,∂/∂y)
            # grad_u = grad_term1 + grad_term2  # [M,dim_u,2]

            # Compute RPI coefficients 
            Wu, dWu_dx = self.compute_coefficients(elem_id, u_patch, R_vector, P_vector, dR_dx, dP_dx, edge=False) # [M,3,n_patch+m_patch]

            # N: [M,3], Wu: [M,3,dim_u] -> -> sum_i N_i * Wu_i 
            u_h = torch.einsum('mi,mid->md', N, Wu) # [M,dim_u]
            # First derivative term
            grad_term1 = torch.einsum('mkj,mjd->mkd', dN_dx, Wu).transpose(1, 2)  # [M,dim_u,2] (mdk)
            # Second derivative term
            grad_term2 = torch.einsum('mi,midk->mdk', N, dWu_dx)  # [M,dim_u,2]
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

            # Compute 1D Jacobian = edge length
            ds = torch.norm(x_ip1 - x_i, dim=1)  # [M]

            # physical coordinates
            x_physical = torch.sum(N.unsqueeze(-1) * coords_edge, dim=1)  # [M,2]

            # Patch coordinates
            coords_patch, patch_mask_elem, patch_idx_elem = self.edge_patch[elem_id]  # coords_patch: [M,2,n_patch,2], patch_mask_elem [M,2,n_patch], patch_idx_elem [M,2,n_patch]
            # Compute radial basis
            R_vector, dR_dx = self.compute_patch_radials(x_physical, coords_patch, patch_mask_elem)
            # Compute polynomial basis
            P_vector, dP_dx = self.compute_patch_polynomials(x_physical, coords_patch, patch_mask_elem, edge=edge)

            # Gather the patch nodal u values per element:
            u_patch = self.values[patch_idx_elem]  # [M,2,n_patch, dim_u]
            u_patch = u_patch * patch_mask_elem[..., None]   # zero out padded nodes


            # # Compute patch weights
            # W, _ = self.solve_patch_weights(elem_id, R_vector, P_vector, dR_dx, dP_dx, edge=edge)
            # # Compute patch weights
            # #W, _ = self.compute_patch_weights(elem_id, R_vector, P_vector, dR_dx, dP_dx, edge=edge) # G_inv must be precomputed

            # # W: [M,3,n_patch], u_patch: [M,2,n_patch,dim_u] -> sum_j W_ij * u_j 
            # Wu = torch.einsum('mij,mijd->mid', W, u_patch) #[M,2,dim_u] 

            # #N: [M,2], Wu: [M,2,dim_u] -> -> sum_i N_i * Wu_i 
            # u_h = torch.einsum('mi,mid->md', N, Wu) # [M,dim_u]


            # Compute RPI coefficients 
            Wu, _ = self.compute_coefficients(elem_id, u_patch, R_vector, P_vector, dR_dx, dP_dx, edge=edge) # [M,3,n_patch+m_patch]

            # N: [M,3], Wu: [M,3,dim_u] -> -> sum_i N_i * Wu_i 
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

        # # Compute and save G_inv_patch
        # Ginv = torch.linalg.inv(G)  # Inverse G: [Nelems,node_per_elem,n_patch+m_patch,n_patch+m_patch]
        # self.register_buffer("G_inv_patch", Ginv)

        # --- 1D edge / Neumann ---
        node_per_elem = 2
        elem_id = torch.arange(self.N_edges, device=self.device)
        coords_patch, patch_mask_elem, _ = self.edge_patch[elem_id]  # coords_patch: [N_edges,2,n_patch,2], patch_mask_elem [N_edges,2,n_patch], patch_idx_elem [N_edges,2,n_patch]
        
        # Compute and save G_patch 
        G_edge = self._compute_G(self.N_edges, node_per_elem, coords_patch, patch_mask_elem)
        self.register_buffer("G_patch"+str(node_per_elem), G_edge)

        # # Compute and save G_inv_patch
        # Ginv_edge = torch.linalg.inv(G_edge)  # Inverse G: [N_edges,2,n_patch+m_patch,n_patch+m_patch]
        # self.register_buffer("G_inv_patch_edge", Ginv_edge)

    def _compute_G(self, Nelems, node_per_elem, coords_patch, patch_mask_elem):

        # 2D mask for pairwise: valid if both patch entries are valid
        mask_mom_valid = patch_mask_elem.unsqueeze(-1) & patch_mask_elem.unsqueeze(-2)  # [Nelems,node_per_elem,n_patch,n_patch]
        
        # --- r_moments: pairwise distances between patch nodes ---
        diff = coords_patch.unsqueeze(3) - coords_patch.unsqueeze(2)  # [Nelems,node_per_elem,n_patch,n_patch,2]
        r_moments = torch.norm(diff, dim=-1)                           # [Nelems,node_per_elem,n_patch,n_patch]

        # --- R_moments: cubic spline ---
        s_mom = r_moments / self.alpha
        mask_mom = (s_mom <= 1) & mask_mom_valid # first mask is for the cubic spline; second mask to 0 out the padding nodes
        R_moments = torch.zeros_like(r_moments)
        R_moments[mask_mom] = (1 - s_mom[mask_mom])**2 * (1 + 2 * s_mom[mask_mom])


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

        # --- Add small epsilon on diagonal for padded nodes ---
        eps = 1e-8
        # Create a mask for the upper-left diagonal: True where padded
        padded_diag_mask = ~patch_mask_elem  # [Nelems,node_per_elem,n_patch]
        diag_indices = torch.arange(self.n_patch)
        # Broadcast to [Nelems,3,n_patch]
        G[..., diag_indices, diag_indices] += eps * padded_diag_mask

        return G


    def compute_patch_radials(self, x_physical: torch.Tensor, coords_patch: torch.Tensor, patch_mask_elem: torch.Tensor):

        # --- r_vector: distances from physical points to patch nodes ---
        x_phys_exp = x_physical.unsqueeze(1).unsqueeze(2)  # [M,1,1,2]
        diff_vec = x_phys_exp - coords_patch               # [M,node_per_elem,n_patch,2]
        r_vector = torch.norm(diff_vec, dim=-1)           # [M,node_per_elem,n_patch]

        # --- R_vector: cubic spline ---
        s_vec = r_vector / self.alpha
        mask_vec = (s_vec <= 1) & patch_mask_elem # first mask is for the cubic spline; second mask to 0 out the padding nodes
        R_vector = torch.zeros_like(r_vector)
        R_vector[mask_vec] = (1 - s_vec[mask_vec])**2 * (1 + 2 * s_vec[mask_vec])

        # --- derivatives dR_vector/dx_physical ---
        dR_dr = torch.zeros_like(r_vector)
        dR_dr[mask_vec] = -6 * (1 - s_vec[mask_vec]) / self.alpha

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
    
    def compute_coefficients(self, elem_id: torch.Tensor, u_patch: torch.Tensor,
                            R_vector: torch.Tensor, P_vector: torch.Tensor,
                            dR_dx: torch.Tensor, dP_dx: torch.Tensor,
                            edge=False):

        # Build zero extension tensor for polynomial block
        zeros_ext = torch.zeros(
            *u_patch.shape[:2],   # Nelems, node_per_elem
            self.m_patch,         # append m_patch rows
            u_patch.shape[-1],    # dim_u
            device=u_patch.device,
            dtype=u_patch.dtype
        )

        # Concatenate along the -2 dimension (patch dimension)
        u_patch_extended = torch.cat([u_patch, zeros_ext], dim=-2)

        # Get precomputed inverse 
        if edge:
            G_mat = self.G_patch2[elem_id]  # [M,2,n_patch+m_patch,n_patch+m_patch]
        else:
            G_mat = self.G_patch3[elem_id]  # [M,3,n_patch+m_patch,n_patch+m_patch]

        coeffs = torch.linalg.solve(G_mat, u_patch_extended) # [M,node_per_elem,n_patch+m_patch]

        # Build vector b 
        b = torch.cat([R_vector, P_vector], dim=-1)  # [M,3,n_patch+m_patch]
        # Build derivative vector b
        db_dx = torch.cat([dR_dx, dP_dx], dim=-2)   # [M,3,n_patch+m_patch,2]

        # b: [M,3,n_patch+m_patch], ceoffs: [M,3,n_patch+m_patch,dim_u] -> sum_j b_j * ceoff_j 
        Wu = torch.einsum('mij,mijd->mid', b, coeffs) #[M,3,dim_u]
        # db_dx: [M,3,n_patch+m_patch, 2], ceoffs: [M,3,n_patch+m_patch,dim_u] -> sum_j b_j * ceoff_j 
        dWu_dx = torch.einsum('mijk,mijd->midk', db_dx, coeffs) #[M,3,dim_u,2]

        return Wu, dWu_dx
    
    def compute_patch_weights(self, elem_id: torch.Tensor,
                        R_vector: torch.Tensor, P_vector: torch.Tensor,
                        dR_dx: torch.Tensor, dP_dx: torch.Tensor,
                        edge=False):

        # Build vector b: [M,node_per_elem,n_patch+m_patch]
        b = torch.cat([R_vector, P_vector], dim=-1)  # [M,node_per_elem,n_patch+m_patch]

        # Build derivative vector b: [M,node_per_elem,n_patch+m_patch,2] 
        db_dx = torch.cat([dR_dx, dP_dx], dim=-2)   # [M,node_per_elem,n_patch+m_patch,2]

        # Get precomputed inverse 
        if edge:
            G_inv = self.G_inv_patch_edge[elem_id]  # [M,2,n_patch+m_patch,n_patch+m_patch]
        else:
            G_inv = self.G_inv_patch[elem_id]  # [M,3,n_patch+m_patch,n_patch+m_patch]

        # Multiply matrices
        W_tilde = torch.einsum('mijk,mij->mik', G_inv, b)           # [M,node_per_elem,n_patch+m_patch]
        dW_tilde = torch.einsum('mijk,mijd->mikd', G_inv, db_dx)    # [M,node_per_elem,n_patch+m_patch,2]

        # Extract patch weights
        W = W_tilde[..., :self.n_patch]          # [M,node_per_elem,n_patch]
        dW_dx = dW_tilde[..., :self.n_patch, :]  # [M,node_per_elem,n_patch,2]

        return W, dW_dx

    def solve_patch_weights(self, elem_id: torch.Tensor,
                        R_vector: torch.Tensor, P_vector: torch.Tensor,
                        dR_dx: torch.Tensor, dP_dx: torch.Tensor,
                        edge=False):

        # Build vector b: [M,node_per_elem,n_patch+m_patch]
        b = torch.cat([R_vector, P_vector], dim=-1)  # [M,node_per_elem,n_patch+m_patch]

        # Build derivative vector b: [M,node_per_elem,n_patch+m_patch,2] 
        db_dx = torch.cat([dR_dx, dP_dx], dim=-2)   # [M,node_per_elem,n_patch+m_patch,2]

        # Get precomputed inverse 
        if edge:
            G_mat = self.G_patch2[elem_id]  # [M,2,n_patch+m_patch,n_patch+m_patch]
        else:
            G_mat = self.G_patch3[elem_id]  # [M,3,n_patch+m_patch,n_patch+m_patch]

        # Solve for Weights
        W_tilde = torch.linalg.solve(G_mat, b)          # [M,node_per_elem,n_patch+m_patch]
        dW_tilde = torch.linalg.solve(G_mat, db_dx)    # [M,node_per_elem,n_patch+m_patch,2]

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
                    ], dim=-2)

        if self.m_patch == 3:
            return torch.stack([
                        torch.stack([torch.zeros_like(x), torch.zeros_like(y)], dim=-1),          # derivative of 1
                        torch.stack([torch.ones_like(x),  torch.zeros_like(y)], dim=-1),          # derivative of x
                        torch.stack([torch.zeros_like(x), torch.ones_like(y)], dim=-1),           # derivative of y
                    ], dim=-2)


    
    def test_conditioning(self):

        # Compute conditioning BEFORE inversion
        mat = self.G_patch3
        # Use SVD: cond = sigma_max / sigma_min
        cond = torch.linalg.cond(mat)  # S: [Nelems,3,n_patch+m_patch]

        # Print statistics
        cond_flat = cond.reshape(-1)
        print("Condition number stats:")
        print(f" min : {cond_flat.min().item():.2e}")
        print(f" median: {cond_flat.median().item():.2e}")
        print(f" max : {cond_flat.max().item():.2e}")

        # Identify degenerate patches (threshold depends on dtype)
        threshold = 1e6 if self.dtype == torch.float64 else 1e3
        bad = cond > threshold
        num_bad = bad.sum().item()
        print(f"Ill-conditioned patches: {num_bad} / {bad.numel()}")

        # Check for symmetry 
        sym_err = (mat - mat.transpose(-1, -2)).abs().max()

        # Compute eigenvalues (since G is symmetric)
        eig  = torch.linalg.eigvalsh(mat)  # S: [Nelems,3,n_patch+m_patch]
        eig_min = eig[..., 0]   # smallest eigen
        eig_max = eig[..., -1]  # largest eigen

        # Print statistics
        print("Symmetry stats:")
        print(f" Max asymmetry:{sym_err.item():.2e}")
        print(f" eig_max : [{eig_max.max().item():.2e}; {eig_max.min().item():.2e}]")
        print(f" eig_min : [{eig_min.max().item():.2e}; {eig_min.min().item():.2e}]")

        # Check for inversion 
        mat_inv = torch.linalg.inv(mat)  # Inverse G: [Nelems,3,n_patch+m_patch,n_patch+m_patch]
        inv_err = (mat_inv @ mat - torch.eye(self.n_patch+self.m_patch, device=self.device, dtype=self.dtype)).abs().max()
        print(f"Inversion error:{inv_err.item():.2e}")

        print('-----'*5)




