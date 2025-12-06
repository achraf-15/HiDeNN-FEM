import torch
import numpy as np

from src.c_model import PiecewiseLinearShapeNN2D
from src.c_loss import EnergyLoss2D
from src.c_mesh import generate_mesh_gmsh, mesh_to_patch, plot_mesh, plot_mesh_with_patches
from src.c_optimization import TestOptimizer
from src.c_plots import plot_displacement_magnitude, plot_von_mises, plot_von_mises_tricontourfd
from src.utils import test_gradients


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
#torch.backends.cuda.matmul.allow_tf32 = False

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
lc = 0.032

node_coords, connectivity, geom_boundary_mask, bc_mask, mn_mask, neumann_edges, dimless_scale = generate_mesh_gmsh(length, height, holes, boundaries, lc, dtype)
patch = mesh_to_patch(connectivity, s=2)

print("-----"*10)
print("Nodes:", node_coords.shape)
print("Connectivity:", connectivity.shape)
print("-----"*5)
print("Geometric boundary nodes:", geom_boundary_mask.sum().item())
print("Dirichlet BC nodes:", bc_mask.sum().item())
print("-----"*5)
print("Neumann MN nodes:", mn_mask.sum().item())
print("Neumann edges:", neumann_edges.shape)
print("-----"*5)
print("Patch tensor shape:", patch.shape)
print("-----"*10)

#plot_mesh_with_patches(node_coords, connectivity, geom_boundary_mask, bc_mask, mn_mask, neumann_edges, patch)

# --- Define Model and Loss function --- 
E = 10e9
nu = 0.3
F_total = 100e3

#characteristic scales
L0 = dimless_scale
T0 = (F_total / L0)            
U0 = (T0 * L0) / E 

# Model
model = PiecewiseLinearShapeNN2D(
    node_coords/L0, # model coordinates should be dimensionless
    connectivity, patch, 
    boundary_mask=geom_boundary_mask, 
    dirichlet_mask = bc_mask,
    u_fixed=0.0,
    neumann_edges=neumann_edges,
).to(device)

# Loss function
loss_fn = EnergyLoss2D(E=E, nu=nu, length=length, height=height, F_total=F_total,
                       gauss_order=37, gauss_order_1d=7, 
                       device=device, dtype=dtype)

optimizer = TestOptimizer(model, loss_fn)

stages = [    
    #{"optimizer": "SGD", "lr": 1e-5, "epochs": 500},
    #{"optimizer": "Adam", "lr": 3e-1, "epochs": 500},
    #{"optimizer": "AdamW", "lr": 3e-1, "epochs": 500},
    #{"optimizer": "RMSprop", "lr": 3e-2, "epochs": 500},
    {"optimizer": "LBFGS", "epochs": 200},
] 

optimizer.optimize(stages)
print("Training finished.\n")

test_gradients(model, loss_fn)

u_vals = model.values.cpu().detach().numpy()       # [Nnodes, 2]
print("Nodal values u", u_vals.shape)
print("Nodal values u_x:", np.mean(u_vals[:,0]), np.min(u_vals[:,0]), np.max(u_vals[:,0]))
print("Nodal values u_y:", np.mean(u_vals[:,1]), np.min(u_vals[:,1]), np.max(u_vals[:,1]))

optimizer.plot_loss()
#plot_mesh(node_coords, connectivity, geom_boundary_mask, bc_mask, mn_mask, neumann_edges)
plot_displacement_magnitude(model, L0=L0, U0=U0)
plot_von_mises(model, E=E, nu=nu, L0=L0, U0=U0)
plot_von_mises_tricontourfd(model, E=E, nu=nu, L0=L0, U0=U0)
