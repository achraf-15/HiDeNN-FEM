import torch
import torch.nn as nn
import torch.nn.functional as F

class PiecewiseLinearShapeNN2D(nn.Module):
    def __init__(self, node_coords, connectivity, boundary_mask=None, dirichlet_mask=None, neumann_mask=None, neumann_map=None, u_fixed=None):
        super().__init__()
        device = node_coords.device
        dtype = node_coords.dtype

        self.register_buffer("initial_node_coords", node_coords.clone())   # [N,2]
        self.Nnodes = node_coords.shape[0] #N

        # connectivity
        self.register_buffer("connectivity", connectivity.long().clone())  # [Ne,3]
        self.Nelems = connectivity.shape[0] #Ne

        # boundary mask
        if boundary_mask is None:
            # default: treat nodes lying on convex hull edges? -> here user should pass mask
            boundary_mask = torch.zeros(self.Nnodes, dtype=torch.bool, device=device)
        self.register_buffer("boundary_mask", boundary_mask.clone())

        # Dirichlet mask
        if dirichlet_mask is None:
            dirichlet_mask = torch.zeros(self.Nnodes, dtype=torch.bool, device=device)
        self.register_buffer("dirichlet_mask", dirichlet_mask.clone().unsqueeze(1))

        # Neumann mask
        if neumann_mask is None:
            neumann_mask = torch.zeros(self.Nnodes, dtype=torch.bool, device=device)
        self.register_buffer("neumann_mask", neumann_mask.clone())
        if neumann_edges is not None:
            self.register_buffer("neumann_edges", neumann_edges) # [N_edges, 2]
            self.N_edges = neumann_edges.shape[0]
        if neumann_map is not None:
            self.register_buffer("neumann_map", neumann_map) # [N_edges]

        free_mask = ~boundary_mask 
        self.node_coords_free = nn.Parameter(node_coords[free_mask])
        self.register_buffer("node_coords_fixed", node_coords[boundary_mask])
        self.free_mask = free_mask

        # nodal DOFs (u at nodes)
        self.dim_u = 2
        scale = 10e-5
        self.u = nn.Parameter(scale * torch.randn(self.Nnodes, self.dim_u, device=device, dtype=dtype)) # may put dim(u) as an argument
        # optionally force boundary u values to u_fixed via property u_full -> Dirichlet boundaries
        if u_fixed is not None:
            u_fixed_val = float(u_fixed)
            self.register_buffer("u_fixed",  torch.full((self.Nnodes, self.dim_u), u_fixed_val, dtype=dtype, device=device))
        else:
            self.u_fixed = None

        self.eps = 1e-12

    @property
    def coords(self):
        """
        Return node coordinates with boundary nodes replaced by initial coords (kept fixed).
        """
        coords = torch.zeros_like(self.initial_node_coords)
        coords[self.free_mask] = self.node_coords_free
        coords[self.boundary_mask] = self.node_coords_fixed
        return coords
        
    @property
    def u_full(self):
        if self.u_fixed is None:
            return self.u
        else:
            # apply u_fixed to boundary DOFs
            return torch.where(self.dirichlet_mask, self.u_fixed, self.u)
        
    @property
    def nm_edges(self):
        x_i = self.coords[self.neumann_edges[:, 0]]
        x_ip1 = self.coords[self.neumann_edges[:, 1]]
        return x_i, x_ip1
        
            
    def forward(self, x_eval, elem_id):
        # Gather the 3 node coordinates per element
        coords_elem = self.coords[self.connectivity[elem_id]]  # [M,3,2]
        ones_col = torch.ones(coords_elem.shape[0], 3, 1, device=coords_elem.device, dtype=coords_elem.dtype)
        ref_Mat = torch.cat([coords_elem, ones_col], dim=2)  # [M,3,3]

        try:
            # Invert each matrix in batch: [M,3,3]
            ref_Mat_inv = torch.linalg.inv(ref_Mat)
        except:
            print(ref_Mat[7473])

        # Pad x_eval with ones: [M,3,1]
        x_eval_h = F.pad(x_eval, (0,1)).unsqueeze(2)  # [M,3,1]

        # Compute reference coords per point: [M,3,1] 
        x_ref = torch.bmm(ref_Mat_inv, x_eval_h) # [M,3,1]

        # Gather nodal u values per element: [M,3, dim(u)]
        u_nodes = self.u_full[self.connectivity[elem_id]]  # [M,3, dim(u)]

        u_h = torch.sum(x_ref * u_nodes, dim=1)# [M, dim(u)]
        
        #print(x_ref)
        return u_h
    
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
        # 1-point centroid
        rs = torch.tensor([[1/3, 1/3]], dtype=dtype, device=device)
        w = torch.tensor([0.5], dtype=dtype, device=device)
    elif order == 3:
        a = 1/6
        rs = torch.tensor([[a,a],[4*a,a],[a,4*a]], dtype=dtype, device=device)
        w = torch.tensor([1/6,1/6,1/6], dtype=dtype, device=device)
    else:
        raise NotImplementedError("Only orders 1 or 3 implemented.")
    return rs, w

def b_force_volume(x: torch.Tensor) -> torch.Tensor:
    """
    Body force per unit area (domain load).
    For this problem: no body force inside the domain (zero everywhere).
    """
    return torch.zeros_like(x)

def b_force_edge(x: torch.Tensor, L: float = 1.0, F_total: float = 100e3) -> torch.Tensor:
    """
    Traction (Neumann) force per unit length.
    Uniform load of 100 kN in +x direction.
    """
    device, dtype = x.device, x.dtype
    t_x = torch.full((x.shape[0],), F_total/L, device=device, dtype=dtype)
    t_y = torch.zeros_like(t_x)
    return torch.stack([t_x, t_y], dim=1)

# Energy based loss function (weak formulation)
def energy_loss(model, connectivity, b_force, t_force, E=10e9, nu=0.3):

    device = model.initial_node_coords.device
    dtype = model.initial_node_coords.dtype

    # --- construct plane-stress constitutive matrix C (3x3) in Voigt [ε_xx, ε_yy, γ_xy]
    factor = E / (1.0 - nu**2)
    C = torch.tensor([[1.0,   nu,        0.0],
                      [nu,    1.0,       0.0],
                      [0.0,   0.0,  (1.0-nu)/2.0]], dtype=dtype, device=device) * factor # C: [3,3]
    
    # --- Domain (area) quadrature setup
    gauss_order = 3
    xg, wg = triangle_gauss_points(order=gauss_order, device=device, dtype=dtype)
    n_elem = model.Nelems
    n_gauss = xg.shape[0] #gauss_order

    # xg: [n_gauss,2] in reference triangle coords; expand to all elements
    x_eval = xg.unsqueeze(0).expand(n_elem, n_gauss, 2).reshape(-1, 2).to(device=device, dtype=dtype)
    elem_id = torch.arange(n_elem, device=device).unsqueeze(1).repeat(1, n_gauss).reshape(-1)  # [n_elem*n_gauss]

    # Evaluate displacement vector at quadrature points
    x_eval.requires_grad_(True) # require grad on x_eval
    u_eval = model(x_eval, elem_id)   # [M,2] where M = n_elem*n_gauss

    # Compute displacement gradients via autograd (component-wise)
    # grad_u_x = [∂u_x/∂x, ∂u_x/∂y], grad_u_y = [∂u_y/∂x, ∂u_y/∂y]
    ones_x = torch.ones_like(u_eval[:, 0], device=device, dtype=dtype)
    ones_y = torch.ones_like(u_eval[:, 1], device=device, dtype=dtype)

    grad_u_x = torch.autograd.grad(u_eval[:, 0], x_eval, grad_outputs=ones_x, create_graph=True)[0]  # [M,2]
    grad_u_y = torch.autograd.grad(u_eval[:, 1], x_eval, grad_outputs=ones_y, create_graph=True)[0]  # [M,2]

    # Strain components (infinitesimal)
    eps_xx = grad_u_x[:, 0]              # ∂u_x/∂x
    eps_yy = grad_u_y[:, 1]              # ∂u_y/∂y
    eps_xy = 0.5 * (grad_u_x[:, 1] + grad_u_y[:, 0])  # 1/2(∂u_x/∂y + ∂u_y/∂x)

    # Voigt strain vector: [eps_xx, eps_yy, gamma_xy] where gamma_xy = 2*eps_xy
    eps_voigt = torch.stack([eps_xx, eps_yy, 2.0 * eps_xy], dim=1)  # [M,3]

    # Stress in Voigt: sigma = C * eps  -> vectorized as eps @ C^T
    sigma_voigt = eps_voigt @ C.T   # [M,3]

    # Elastic energy density at each quadrature point: 0.5 * eps^T * sigma
    elastic_density = 0.5 * torch.sum(eps_voigt * sigma_voigt, dim=1)  # [M]

    # Body force potential (vector): b(x) . u
    b_vec = b_force(x_eval)  # should return [M,2]
    body_potential_density = torch.sum(b_vec * u_eval, dim=1)  # [M]

    ### Jaccobian calculation, might want to put it inside the model
    # Jacobian / area mapping per element 
    coords = model.coords  # [Nnodes,2]
    v0 = coords[connectivity[:, 0]]  # [Ne,2]
    v1 = coords[connectivity[:, 1]]
    v2 = coords[connectivity[:, 2]]
    Jmat = torch.stack([v1 - v0, v2 - v0], dim=2)  # [Ne,2,2]
    detJ = (Jmat[:, 0, 0] * Jmat[:, 1, 1] - Jmat[:, 0, 1] * Jmat[:, 1, 0]).abs()  # [Ne]
    # Note: reference triangle area is 0.5; reference weights wg should be defined for that reference domain

    # Expand detJ and wg to pointwise weights
    detJ_flat = detJ.unsqueeze(1).repeat(1, n_gauss).reshape(-1)  # [M]
    wg_flat = wg.unsqueeze(0).repeat(n_elem, 1).reshape(-1).to(device=device, dtype=dtype)  # [M]
    ###

    quad_weights = wg_flat * detJ_flat  # final area weights [M]

    # Domain contribution to total potential: integrate density over domain
    domain_energy = torch.sum(quad_weights * elastic_density)  # scalar
    body_work = torch.sum(quad_weights * body_potential_density)  # scalar

    # Total domain potential: elastic - body
    total_domain = domain_energy - body_work

    # ----- Neumann boundary (edges) contribution -----
    # 1D gauss on interval [0,1]
    gauss_order_1d = 2
    xi_1d, wi_1d = interval_gauss_points(order=gauss_order_1d, device=device, dtype=dtype)  # xi_1d: [ng1], wi_1d: [ng1]
    ng1 = xi_1d.shape[0] #gauss_order_1d
    
    # endpoints of each neumann edge from the model
    x_i, x_ip1 = model.nm_edges  # each [N_edges,2]
    N_edges = model.N_edges

    # map 1D xi -> 2D physical points
    xq = (1.0 - xi_1d[None, :, None]) * x_i[:, None, :] + xi_1d[None, :, None] * x_ip1[:, None, :] # [N_edges, ng1, 2]
    xq_flat = xq.reshape(-1, 2)  # [N_edges * ng1, 2]

    # elem id per edge repeated for ng1 points
    neumann_map = model.neumann_map  # array-like [N_edges]
    elem_id_edge = torch.repeat_interleave(torch.tensor(neumann_map, device=device, dtype=torch.long), repeats=ng1)

    # evaluate displacement at edge gauss pts
    u_edge = model(xq_flat, elem_id_edge)  # [N_edges*ng1, 2]

    # traction at edge pts
    t_edge = t_force(xq_flat) #params L(edge_lenght) and F(froce applied)  # [N_edges*ng1, 2]

    # 1D weight includes physical Jacobian = edge length
    ds = torch.norm(x_ip1 - x_i, dim=1)  # [N_edges]
    w_edge = (wi_1d[None, :] * ds[:, None]).reshape(-1).to(device=device, dtype=dtype)  # [N_edges*ng1]

    # Neumann potential = integral t·u ds  (subtract in total potential)
    neumann_work = torch.sum( (u_edge * t_edge).sum(dim=1) * w_edge )

    # ----- Total potential energy
    total_potential = total_domain - neumann_work

    return total_potential 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

####################################
import meshzoo
import numpy as np
import torch

# Define rectangle and holes
length, height = 2.0, 1.0
holes = [(0.5,0.7,0.12), (1.0,0.3,0.15), (1.4,0.6,0.1)]
nx, ny = 50, 100

# Generate structured triangle mesh
x = np.linspace(0.0, length, nx)
y = np.linspace(0.0, height, ny)
points, cells = meshzoo.rectangle_tri(x, y, variant="zigzag")
points = np.array(points)
cells = np.array(cells)

# Remove nodes inside holes
mask = np.ones(len(points), dtype=bool)
for cx, cy, r in holes:
    dx, dy = points[:,0]-cx, points[:,1]-cy
    mask &= (dx**2 + dy**2) > r**2

points_kept = points[mask]

# Remap indices
old_to_new = -np.ones(len(points), dtype=int)
old_to_new[mask] = np.arange(points_kept.shape[0])

# Full geometric boundary mask
geom_boundary_mask = np.zeros(len(points_kept), dtype=bool)
tol = 1e-6

# Filter connectivity
cells_kept = []
for tri in cells:
    #print(tri)
    #print(mask[tri])
    if all(mask[tri]):
        cells_kept.append(old_to_new[tri])
    else:
        for point in tri:
            if mask[point]:
                geom_boundary_mask[old_to_new[point]] = 1
cells_kept = np.array(cells_kept)

# Rectangle edges
geom_boundary_mask |= np.abs(points_kept[:,0]-0.0)< tol
geom_boundary_mask |= np.abs(points_kept[:,0]-length)< tol
geom_boundary_mask |= np.abs(points_kept[:,1]-0.0)< tol
geom_boundary_mask |= np.abs(points_kept[:,1]-height)< tol

# BC mask (Dirichlet)
bc_mask = (points_kept[:,0] == 0.0)  # left edge fixed (Drichlet)

# MN mask (Neumann)
mn_mask = (points_kept[:,0] == length)  # right edge (Neumann)

# Extract all edges from triangles
all_edges = np.vstack([
    cells_kept[:, [0, 1]],
    cells_kept[:, [1, 2]],
    cells_kept[:, [2, 0]]
])

# Sort and unique edges
all_edges = np.sort(all_edges, axis=1)
unique_edges = np.unique(all_edges, axis=0)

# Filter unique edges with both nodes on the Neumann boundary
neumann_edges = unique_edges[np.all(mn_mask[unique_edges], axis=1)]

# Map each edge to its element
elem_ids = np.repeat(np.arange(cells_kept.shape[0]), 3)

# Vectorized matching using broadcasting
matches = (all_edges[:, None, :] == neumann_edges[None, :, :]).all(axis=2)
neumann_edge2elem = elem_ids[np.argmax(matches, axis=0)]

# Convert to torch tensors
node_coords = torch.tensor(points_kept, dtype=torch.float32)
connectivity = torch.tensor(cells_kept, dtype=torch.long)
geom_boundary_mask = torch.tensor(geom_boundary_mask, dtype=torch.bool)
bc_mask = torch.tensor(bc_mask, dtype=torch.bool)
mn_mask = torch.tensor(mn_mask, dtype=torch.bool)
neumann_edges  = torch.tensor(neumann_edges, dtype=torch.long)
neumann_edge2elem = torch.tensor(neumann_edge2elem, dtype=torch.long)


print("Nodes:", node_coords.shape)
print("Connectivity:", connectivity.shape)
print("Geometric boundary nodes:", geom_boundary_mask.sum().item())
print("Dirichlet BC nodes:", bc_mask.sum().item())
print("Neumann MN nodes:", mn_mask.sum().item())
print("Neumann edges:", neumann_edges.shape)
print("Neumann edges2elem mapping:", neumann_edge2elem.shape)


# import matplotlib.pyplot as plt
# plt.triplot(points_kept[:,0], points_kept[:,1], cells_kept)
# plt.scatter(points_kept[geom_boundary_mask,0], points_kept[geom_boundary_mask,1], color='red')
# #plt.scatter(points_kept[mn_mask,0], points_kept[mn_mask,1], color='purple')
# # # Draw Neumann edges in red
# # for e in neumann_edges.numpy():
# #     x = points_kept[e, 0]
# #     y = points_kept[e, 1]
# #     plt.plot(x, y, color='red', linewidth=2)
# plt.gca().set_aspect('equal')
# plt.show()
##########

# Model
model = PiecewiseLinearShapeNN2D(
    node_coords, connectivity,
    boundary_mask=geom_boundary_mask, 
    dirichlet_mask = bc_mask,
    neumann_mask=mn_mask,
    neumann_map=neumann_edge2elem,
    u_fixed=0.0
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)


for epoch in range(1500):
    optimizer.zero_grad()
    loss = energy_loss(model, connectivity, b_force=b_force_volume, t_force=b_force_edge)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6e}")

print("Training finished.")
u_vals = model.u_full.cpu().detach().numpy()       # [Nnodes, 2]
print("Nodal values u_x:", np.mean(u_vals[:,0]), np.min(u_vals[:,0]), np.max(u_vals[:,0]))
print("Nodal values u_y:", np.mean(u_vals[:,1]), np.min(u_vals[:,1]), np.max(u_vals[:,1]))


import matplotlib.pyplot as plt
import numpy as np

# Extract data from model
coords = model.coords.cpu().detach().numpy()        # [Nnodes, 2]
triangles = model.connectivity.cpu().detach().numpy()  # [Nelems, 3]
u_vals = model.u_full.cpu().detach().numpy()       # [Nnodes, 2]

# Compute displacement magnitude at each node
u_mag = np.linalg.norm(u_vals, axis=1)  # [Nnodes]

# Compute triangle face values: mean of node magnitudes per triangle
tri_face_vals = np.mean(u_mag[triangles], axis=1)  # [Nelems]

# Plot using tripcolor
plt.figure(figsize=(8,4))
plt.tripcolor(coords[:,0], coords[:,1], triangles, facecolors=tri_face_vals, edgecolors='k', cmap='viridis')
plt.colorbar(label='Displacement magnitude ||u||')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('HiDeNN displacement field (magnitude)')
plt.gca().set_aspect('equal')
plt.show()