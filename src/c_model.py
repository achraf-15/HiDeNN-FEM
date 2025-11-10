import torch
import torch.nn as nn
import torch.nn.functional as F
    
class NeumannEdgesWrapper:
    def __init__(self, coords, edges):
        self.coords = coords        # [Nnodes, 2]
        self.edges = edges          # [N_edges, 2]

    def __getitem__(self, idx):
        # Support slicing and single index
        x_i = self.coords[self.edges[idx, 0]]
        x_ip1 = self.coords[self.edges[idx, 1]]
        return x_i, x_ip1

    def __len__(self):
        return self.edges.shape[0]
    
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

        self.scale = 1e-5
        self.dim_u = 2

        self.register_buffer("initial_node_coords", node_coords.clone())   # [N,2]
        self.Nnodes = node_coords.shape[0] #N

        # connectivity
        self.register_buffer("connectivity", connectivity.long().clone())  # [Ne,3]
        self.Nelems = connectivity.shape[0] #Ne

        # # connectivity
        # self.register_buffer("patch", patch.long().clone())  # [n_patch,3]
        # self.Nelems = patch.shape[0] #n_patch

        # patches
        patch_raw = patch.long().clone()
        self.register_buffer("patch_raw", patch_raw)

        # precompute safe indices and mask once
        patch_mask = patch_raw >= 0                    # bool [Nnodes, n_patch]
        patch_safe = patch_raw.clone()
        patch_safe[~patch_mask] = 0                    # safe index for torch indexing

        self.register_buffer("patch_safe", patch_safe) # long
        self.register_buffer("patch_mask", patch_mask) # bool

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
        self.u_free = nn.Parameter(self.scale*torch.randn(u_free_mask.sum().item(), self.dim_u))
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
    def u_full(self):
        u = torch.zeros(self.Nnodes, self.dim_u, device=self.device, dtype=self.dtype)
        u[self.u_free_mask] = self.u_free
        if self.u_fixed is not None:
            u[self.dirichlet_mask] = self.u_fixed
        return u

    @property
    def domain_elements(self):
        return ConnectivityWrapper(self.coords, self.connectivity)
    
    @property
    def nm_edges(self):
        return NeumannEdgesWrapper(self.coords, self.neumann_edges)
    
    @property
    def domain_patch(self):
        return PatchWrapper(self.coords, self.connectivity, self.patch_safe, self.patch_mask)
        
            
    def forward(self, x_eval, elem_id, edge=False):
        if not edge:
            # --- 2D triangle / domain ---
            # Gather the 3 node coordinates per element
            coords_elem = self.domain_elements[elem_id] 
            
            # x_eval is assumed in reference triangle coordinates (xi, eta)
            xi = x_eval[:, 0:1]  # [M,1]
            eta = x_eval[:, 1:2]  # [M,1]
            zeta = 1.0 - xi - eta
            # Shape function weights = barycentric coordinates
            N = torch.cat([xi, eta, zeta], dim=1)  # [M,3]

            # physical coordinates
            x_physical = torch.sum(N.unsqueeze(-1) * coords_elem, dim=1)  # [M,2]

            coords_patch, patch_mask_elem, patch_idx_elem = self.domain_patch[elem_id]  # coords_patch: [M,3,n_patch,2], patch_mask_elem [M,3,n_patch], patch_idx_elem [M,3,n_patch]
            #print("coords_patch", coords_patch.shape)
            #print("patch_mask_elem", patch_mask_elem.shape)

            R_moments, R_vector, dR_dx = self.compute_patch_splines(x_physical, coords_patch, patch_mask_elem, alpha=0.2)
            #print("r_moments", r_moments.shape)
            #print("R_moments", R_moments.shape)
            #print("r_vector", r_vector.shape)
            #print("R_vector", R_vector.shape)
            #print("dR_dx", dR_dx.shape)

            P_moments, P_vector, dP_dx = self.compute_patch_polynomials(x_physical, coords_patch, patch_mask_elem)
            #print("P_moments", P_moments.shape)
            #print("P_vector", P_vector.shape)
            #print("dP_dx", dP_dx.shape)

            W, dW_dx = self.solve_patch_weights(R_moments, P_moments, R_vector, P_vector, dR_dx, dP_dx, patch_mask_elem)
            #print("W matrix", W.shape)
            #print("dW_dx matrix", dW_dx.shape)

            # Gather the patch nodal u values per element:
            u_patch = self.u_full[patch_idx_elem]  # [M,3,n_patch, dim_u]
            u_patch = u_patch * patch_mask_elem[..., None]   # zero out padded nodes
            #print("u_patch", u_patch.shape)

            # Step 1: sum over patch nodes
            # W: [M,3,n_patch], u_patch: [M,3,n_patch,dim_u] -> sum_j W_ij * u_j 
            Wu = torch.einsum('mij,mijd->mid', W, u_patch) #[M,3,dim_u] 
            #print("Wu", Wu.shape)

            # Step 2: sum over element nodes with shape functions
            # N: [M,3], Wu: [M,3,dim_u] -> -> sum_i N_i * Wu_i 
            u_h = torch.einsum('mi,mid->md', N, Wu) # [M,dim_u]
            #print("u_h", u_h.shape)

            # Compute 2x2 Jacobian for area / quadrature mapping using all 3 nodes
            v0 = coords_elem[:, 0, :]  # [M,2]
            v1 = coords_elem[:, 1, :]
            v2 = coords_elem[:, 2, :]
            Jmat = torch.stack([v0 - v2, v1 - v2], dim=2)  # [M,2,2]
            detJ = torch.linalg.det(Jmat)  # Determinent Jacobian: [M]
            Jinv = torch.linalg.inv(Jmat)  # Inverse Jacobian: [M, 2, 2]

            # Shape function derivatives w.r.t local coords (ξ, η) # [2 local derivatives, 3 nodes]
            dN_dxi = torch.tensor([[1., 0., -1.],
                                   [0., 1., -1.]], device=self.device, dtype=self.dtype)  # [2,3]

            # Derivatives in physical coords: dN_dx = J^-1 * dN_dxi
            dN_dx = torch.einsum("mij,jk->mik", Jinv, dN_dxi)  # [M,2,3]

            grad_term1 = torch.einsum('mij,mjd->mid', dN_dx, Wu).transpose(1, 2)  # [M,dim_u,2]

            grad_term2 = torch.einsum('mijc,mijk->mck', u_patch, dW_dx)  # [M,dim_u,2]

            # grad_u: [M,2,2]  (rows=u components, cols=∂/∂x,∂/∂y)
            grad_u = grad_term1 + grad_term2  # [M,dim_u,2]

            return u_h, detJ, grad_u
        
        else:
            # --- 1D edge / Neumann ---
            # Get the two physical nodes of each edge
            x_i, x_ip1 = self.nm_edges[elem_id] 

            # x_eval: [M,1] in reference edge coordinates ξ ∈ [0,1]
            xi = x_eval[:, 0:1]  # [M,1]
            N = torch.cat([1.0 - xi, xi], dim=1)  # linear shape functions for 2 nodes

            # Gather nodal u values for edges
            u_nodes = self.u_full[self.neumann_edges[elem_id]]         # [M,2,dim(u)]

            # Interpolate displacement along edge
            u_h = torch.sum(N.unsqueeze(2) * u_nodes, dim=1)  # [M, dim(u)]

            # Compute 1D Jacobian = edge length
            ds = torch.norm(x_ip1 - x_i, dim=1)  # [M]
            return u_h, ds
        

    def compute_patch_splines(self, x_physical: torch.Tensor, coords_patch: torch.Tensor, patch_mask_elem: torch.Tensor, alpha: float):

        # 2D mask for pairwise: valid if both patch entries are valid
        mask_mom_valid = patch_mask_elem.unsqueeze(-1) & patch_mask_elem.unsqueeze(-2)  # [M,3,n_patch,n_patch]
        
        # --- r_moments: pairwise distances between patch nodes ---
        diff = coords_patch.unsqueeze(3) - coords_patch.unsqueeze(2)  # [M,3,n_patch,n_patch,2]
        r_moments = torch.norm(diff, dim=-1)                           # [M,3,n_patch,n_patch]

        # --- R_moments: cubic spline ---
        s_mom = r_moments / alpha
        mask_mom = (s_mom <= 1) & mask_mom_valid # first mask is for the cubic spline: might move it later; second mask to 0 out the padding nodes
        R_moments = torch.zeros_like(r_moments)
        R_moments[mask_mom] = (1 - s_mom[mask_mom])**2 * (1 + 2 * s_mom[mask_mom])

        # --- r_vector: distances from physical points to patch nodes ---
        x_phys_exp = x_physical.unsqueeze(1).unsqueeze(2)  # [M,1,1,2]
        diff_vec = x_phys_exp - coords_patch               # [M,3,n_patch,2]
        r_vector = torch.norm(diff_vec, dim=-1)           # [M,3,n_patch]

        # --- R_vector: cubic spline ---
        s_vec = r_vector / alpha
        mask_vec = (s_vec <= 1) & patch_mask_elem # first mask is for the cubic spline: might move it later; second mask to 0 out the padding nodes
        R_vector = torch.zeros_like(r_vector)
        R_vector[mask_vec] = (1 - s_vec[mask_vec])**2 * (1 + 2 * s_vec[mask_vec])

        # --- derivatives dR_vector/dx_physical ---
        dR_dr = torch.zeros_like(r_vector)
        dR_dr[mask_vec] = -6 * (1 - s_vec[mask_vec]) / alpha

        eps = 1e-12
        dr_dx = diff_vec / (r_vector.unsqueeze(-1) + eps)  # [M,3,n_patch,2]
        dR_vector_dx = dR_dr.unsqueeze(-1) * dr_dx         # [M,3,n_patch,2]

        return R_moments, R_vector, dR_vector_dx
    
    def compute_patch_polynomials(self, x_physical: torch.Tensor, coords_patch: torch.Tensor, patch_mask_elem: torch.Tensor):

        m_patch = 6  # [1, x, y, x^2, x*y, y^2]
        three = 3

        # --- P_moments on patch nodes ---
        x_patch = coords_patch[..., 0]  # [M,3,n_patch]
        y_patch = coords_patch[..., 1]  # [M,3,n_patch]
        P_moments = torch.stack([
            torch.ones_like(x_patch),       # 1
            x_patch,                        # x
            y_patch,                        # y
            x_patch**2,                     # x^2
            x_patch * y_patch,              # x*y
            y_patch**2                      # y^2
        ], dim=-1)                           # [M,3,n_patch, m_patch]

        # Mask out padded nodes (broadcast mask to last dim)
        P_moments = P_moments * patch_mask_elem.unsqueeze(-1).float()  # [M,3,n_patch,m_patch]

        # --- P_vector on evaluation points ---
        x_eval = x_physical[:, 0:1]  # [M,1]
        y_eval = x_physical[:, 1:2]  # [M,1]
        # Broadcast to element nodes
        x_eval_nodes = x_eval.expand(-1, three)  # [M,3]
        y_eval_nodes = y_eval.expand(-1, three)  # [M,3]

        P_vector = torch.stack([
            torch.ones_like(x_eval_nodes),  # 1
            x_eval_nodes,                   # x
            y_eval_nodes,                   # y
            x_eval_nodes**2,                # x^2
            x_eval_nodes * y_eval_nodes,    # x*y
            y_eval_nodes**2                 # y^2
        ], dim=-1)                           # [M,3,m_patch]

        # --- dP_dx w.r.t x_physical ---
        # derivative of each polynomial term w.r.t [x, y]
        dP_dx = torch.stack([
            torch.stack([torch.zeros_like(x_eval_nodes), torch.zeros_like(x_eval_nodes)], dim=-1),          # derivative of 1
            torch.stack([torch.ones_like(x_eval_nodes),  torch.zeros_like(x_eval_nodes)], dim=-1),          # derivative of x
            torch.stack([torch.zeros_like(x_eval_nodes), torch.ones_like(x_eval_nodes)], dim=-1),           # derivative of y
            torch.stack([2*x_eval_nodes,                 torch.zeros_like(x_eval_nodes)], dim=-1),          # derivative of x^2
            torch.stack([y_eval_nodes,                   x_eval_nodes], dim=-1),                            # derivative of x*y
            torch.stack([torch.zeros_like(x_eval_nodes), 2*y_eval_nodes], dim=-1)                           # derivative of y^2
        ], dim=-2)  # [M,3,m_patch*2]

        # Reshape last dimension to [m_patch, 2]
        #dP_dx = dP_dx.view(M, three, m_patch, 2)  # [M,3,6,2]

        return P_moments, P_vector, dP_dx

    def solve_patch_weights(self, R_moments: torch.Tensor, P_moments: torch.Tensor,
                            R_vector: torch.Tensor, P_vector: torch.Tensor,
                            dR_dx: torch.Tensor, dP_dx: torch.Tensor,
                            patch_mask: torch.Tensor):

        M, three, n_patch, _ = R_moments.shape
        m_patch = P_moments.shape[-1]

        # --- Build block matrix G: [M,3,n_patch+m_patch, n_patch+m_patch] ---
        # Upper-left: R_moments
        G_UL = R_moments
        # Upper-right: P_moments
        G_UR = P_moments
        # Lower-left: P_moments^T (transpose last two dims)
        G_LL = P_moments.transpose(-2, -1)
        # Lower-right: zeros [M,3,m_patch,m_patch]
        G_LR = torch.zeros(M, three, m_patch, m_patch, device=self.device, dtype=self.dtype)

        # Concatenate along last dimension
        G_top = torch.cat([G_UL, G_UR], dim=-1)    # [M,3,n_patch,n_patch+m_patch]
        G_bottom = torch.cat([G_LL, G_LR], dim=-1) # [M,3,m_patch,n_patch+m_patch]
        G = torch.cat([G_top, G_bottom], dim=-2)   # [M,3,n_patch+m_patch,n_patch+m_patch]
        #print("G matrix",G.shape)

        # --- Add small epsilon on diagonal for padded nodes ---
        eps = 1e-12
        # Create a mask for the upper-left diagonal: True where padded
        padded_diag_mask = ~patch_mask  # [M,3,n_patch]
        diag_indices = torch.arange(n_patch)
        # Broadcast to [M,3,n_patch]
        G[..., diag_indices, diag_indices] += eps * padded_diag_mask

        # --- Build vector b: [M,3,n_patch+m_patch] ---
        b = torch.cat([R_vector, P_vector], dim=-1)

        # --- Build derivative vector b: [M,3,n_patch+m_patch,2] ---
        db_dx = torch.cat([dR_dx, dP_dx], dim=-2)

        # --- Solve linear system G * W_tilde = b ---
        # We'll reshape batch dimensions to do batch solve
        G_flat = G.view(M*three, n_patch+m_patch, n_patch+m_patch)
        b_flat = b.view(M*three, n_patch+m_patch, 1)
        db_flat = db_dx.view(M*three, n_patch+m_patch, 2)
        # print("G_flat", G_flat[0].shape)
        # print("b_flat", b_flat[0].shape)
        # print("db_flat", db_flat[0].shape)

        # Solve
        W_tilde_flat = torch.linalg.solve(G_flat, b_flat)  # [M*3, n_patch+m_patch,1]
        W_tilde = W_tilde_flat.view(M, three, n_patch+m_patch).contiguous()

        dW_tilde_flat = torch.linalg.solve(G_flat, db_flat)  # [M*3, n_patch+m_patch,1]
        dW_tilde = dW_tilde_flat.view(M, three, n_patch+m_patch, 2).contiguous()

        # --- Extract patch weights W ---
        W = W_tilde[..., :n_patch]  # [M,3,n_patch]
        dW_dx = dW_tilde[..., :n_patch,:]  # [M,3,n_patch]

        return W, dW_dx
