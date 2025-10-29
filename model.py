import torch
import torch.nn as nn

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


class PiecewiseLinearShapeNN2D(nn.Module):
    def __init__(self, node_coords, connectivity, boundary_mask=None, dirichlet_mask=None, u_fixed=None, neumann_edges=None):
        super().__init__()

        self.scale = 1e-5
        self.dim_u = 2

        self.register_buffer("initial_node_coords", node_coords.clone())   # [N,2]
        self.Nnodes = node_coords.shape[0] #N

        # connectivity
        self.register_buffer("connectivity", connectivity.long().clone())  # [Ne,3]
        self.Nelems = connectivity.shape[0] #Ne

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

            # Gather nodal u values per element: [M,3, dim(u)]
            u_nodes = self.u_full[self.connectivity[elem_id]]  # [M,3, dim(u)]

            u_h = torch.sum(N.unsqueeze(2) * u_nodes, dim=1)# [M, dim(u)]

            # Compute 2x2 Jacobian for area / quadrature mapping using all 3 nodes
            v0 = coords_elem[:, 0, :]  # [M,2]
            v1 = coords_elem[:, 1, :]
            v2 = coords_elem[:, 2, :]
            Jmat = torch.stack([v1 - v0, v2 - v0], dim=2)  # [M,2,2]
            detJ = torch.linalg.det(Jmat)  # [M]

            # Inverse Jacobian: [M, 2, 2]
            Jinv = torch.linalg.inv(Jmat)

            # Shape function derivatives w.r.t local coords (ξ, η)
            # [3 nodes, 2 local derivatives]
            dN_dxi = torch.tensor([[-1, -1], [1, 0], [0, 1]], device=self.device, dtype=self.dtype)

            # Derivatives in physical coords: [M, 3, 2]
            dN_dx = torch.einsum('mij,nj->mni', Jinv, dN_dxi)

            # Compute ∂u/∂x and ∂u/∂y using nodal displacements
            # u_nodes: [M, 3, dim_u], dN_dx: [M, 3, 2]
            # Result: [M, dim_u, 2] -> last dim = [∂/∂x, ∂/∂y]
            grad_u = torch.einsum('mni,mnj->mji', dN_dx, u_nodes)

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
