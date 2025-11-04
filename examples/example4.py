import torch
import numpy as np

from src.models import PiecewiseLinearShapeNN2D
from src.loss import EnergyLoss2D
from src.mesh import generate_mesh_gmsh, generate_mesh, plot_mesh
from src.plots import plot_displacement_magnitude, plot_von_mises, plot_model_mesh
from src.utils import test_gradients


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# Define rectangle and holes
length, height = 2.0, 1.0
holes = [(0.5,0.7,0.12), (1.0,0.3,0.15), (1.4,0.6,0.1)]
#holes = [(0.5,0.5,0.2)]
boundaries = {
    'up': 0,    # no conditions
    'down': 0,  # no conditions
    'right': 2, # Neumann boundaries
    'left': 1   # Drichlet boundaries
}
nx, ny = 200, 100
lc = 0.05
node_coords, connectivity, geom_boundary_mask, bc_mask, mn_mask, neumann_edges = generate_mesh_gmsh(length, height, holes, boundaries, lc)
#node_coords, connectivity, geom_boundary_mask, bc_mask, mn_mask, neumann_edges = generate_mesh(length,height,holes,boundaries,nx,ny)


print("Nodes:", node_coords.shape)
print("Connectivity:", connectivity.shape)
print("Geometric boundary nodes:", geom_boundary_mask.sum().item())
print("Dirichlet BC nodes:", bc_mask.sum().item())
print("Neumann MN nodes:", mn_mask.sum().item())
print("Neumann edges:", neumann_edges.shape)

#plot_mesh(node_coords, connectivity, geom_boundary_mask, bc_mask, mn_mask, neumann_edges)

# Model
model = PiecewiseLinearShapeNN2D(
    node_coords, connectivity,
    boundary_mask=geom_boundary_mask, 
    dirichlet_mask = bc_mask,
    u_fixed=0.0,
    neumann_edges=neumann_edges,
).to(device)

# Loss function
loss_fn = EnergyLoss2D(E=10e9, nu=0.3, length=length, height=height, device=device, dtype=dtype)

#test_gradients(model, loss_fn)

### Adam optimizer
# optimizer = torch.optim.Adam([
#     {'params': model.u_free, 'lr': 1e-4},
#     {'params': model.node_coords_free, 'lr': 1e-5}  # Much smaller!
# ], lr=1e-4)

# for epoch in range(2000):
#     optimizer.zero_grad()
#     loss = loss_fn(model)
#     loss.backward()
#     optimizer.step()
#     if epoch % 200 == 0:
#         print(f"Epoch {epoch}: Loss = {loss.item():.6e}")

### LBFGS optimizer 
optimizer = torch.optim.LBFGS(model.parameters())

for epoch in range(30):

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(model)
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    if epoch % 5 == 0:
        print(f"Epoch {epoch:04d}: Loss = {loss.item():.6e}")


# ### Alternating scheme
# optimizer = torch.optim.Adam([
#     {'params': model.u_free, 'lr': 1e-6},
#     {'params': model.node_coords_free, 'lr': 1e-7}  # Much smaller!
# ], lr=1e-6)
# # optimizer = torch.optim.LBFGS(model.parameters())

# for epoch in range(500):
#     # Step 1: Optimize displacements (freeze mesh)
#     model.node_coords_free.requires_grad = False
#     model.u_free.requires_grad = True
    
#     for _ in range(10):  # Inner iterations
#         optimizer.zero_grad()
#         loss = loss_fn(model)
#         loss.backward()
#         optimizer.step()
    
#     # Step 2: Optimize mesh (freeze displacements)
#     model.u_free.requires_grad = False
#     model.node_coords_free.requires_grad = True
    
#     for _ in range(5):  # Fewer iterations for mesh
#         optimizer.zero_grad()
#         loss = loss_fn(model) #+ 0.1 * mesh_quality_loss(model.coords, model.connectivity)
#         loss.backward()
#         optimizer.step()

#     if epoch % 50 == 0:
#          print(f"Epoch {epoch}: Loss = {loss.item():.6e}")

# ### Two phase scheme
# optimizer = torch.optim.Adam([
#     {'params': model.u_free, 'lr': 1e-6},
#     {'params': model.node_coords_free, 'lr': 1e-7}  # Much smaller!
# ], lr=1e-6)
# for epoch in range(1000):
#     optimizer.zero_grad()
#     loss = loss_fn(model)
#     loss.backward()
#     optimizer.step()
#     if epoch % 200 == 0:
#         print(f"Epoch {epoch}: Loss = {loss.item():.6e}")

# optimizer = torch.optim.LBFGS(model.parameters())
# for epoch in range(40):

#     def closure():
#         optimizer.zero_grad()
#         loss = loss_fn(model)
#         loss.backward()
#         return loss

#     loss = optimizer.step(closure)
#     if epoch % 10 == 0:
#         print(f"Epoch {epoch:04d}: Loss = {loss.item():.6e}")



print("Training finished.")
u_vals = model.u_full.cpu().detach().numpy()       # [Nnodes, 2]
print("Nodal values u", u_vals.shape)
print("Nodal values u_x:", np.mean(u_vals[:,0]), np.min(u_vals[:,0]), np.max(u_vals[:,0]))
print("Nodal values u_y:", np.mean(u_vals[:,1]), np.min(u_vals[:,1]), np.max(u_vals[:,1]))

#test_gradients(model, loss_fn)

#plot_mesh(node_coords, connectivity, geom_boundary_mask, bc_mask, mn_mask, neumann_edges)
plot_model_mesh(model)
plot_displacement_magnitude(model)
plot_von_mises(model)