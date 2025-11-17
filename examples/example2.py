import torch
import torch.nn as nn
import torch
import torch.optim as optim

from src.models import GridLinearShapeNN2D
from src.plots import plot_2d_solution, plot_2d_derivatives


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Structured Cartesian grid
Nx, Ny = 25, 25
grid_x = torch.linspace(0, 1, Nx, device=device)
grid_y = torch.linspace(0, 1, Ny, device=device)

# Training points
nx_train, ny_train = 100, 100
M = 1000  # number of collocation points per epoch
x_train_1d = torch.linspace(0, 1, nx_train, device=device)
y_train_1d = torch.linspace(0, 1, ny_train, device=device)

# Tensor-product grid of training points
XX, YY = torch.meshgrid(x_train_1d, y_train_1d, indexing="ij")
x_train = torch.stack([XX.flatten(), YY.flatten()], dim=1)  # shape (nx_train*ny_train, 2)

# True values
u_true = torch.sin(2 * torch.pi * x_train[:, 0]) * torch.cos(2 * torch.pi * x_train[:, 1])

# Model & optimizer
model = GridLinearShapeNN2D(grid_x=grid_x,
                                 grid_y=grid_y,
                                 boundary_mask_x=None, 
                                 boundary_mask_y=None,
                                 r_adapt=True,
                            ).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005)

# L2 projection training
for epoch in range(3000):
    optimizer.zero_grad()
    indices = torch.randint(0, x_train.shape[0], (M,), device=device)
    x_train_batch = x_train[indices]
    u_true_batch = u_true[indices]
    pred = model(x_train_batch)
    loss = ((pred - u_true_batch) ** 2).mean()
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.6f}")

# Visualization 
exact_solution_2d = lambda X, Y: (torch.sin(2*torch.pi*X) * torch.cos(2*torch.pi*Y)).cpu().numpy()
plot_2d_solution(model, u_exact=exact_solution_2d)
plot_2d_derivatives(model, n_eval=50, title="FEM Derivatives")