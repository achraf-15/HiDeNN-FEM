import torch
import torch.optim as optim

from models import PiecewiseLinearShapeNN
from plots import plot_fem_solution, plot_fem_derivative

# ----------------------------------------------------------------------
# L² Projection using Piecewise Linear Finite Elements
# ----------------------------------------------------------------------
# Problem:
# Find u_h ∈ V_h (piecewise linear FE space) such that:
#
#       ∫ (u_h - u_true) v dx = 0    ∀ v ∈ V_h
#
# Equivalent to minimizing the functional:
#       J(u_h) = ∫ (u_h - u_true)² dx
#
# This represents the orthogonal projection of u_true(x)
# onto the finite element space spanned by the model’s shape functions.
# ----------------------------------------------------------------------

# # Generate synthetic data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_grid = torch.linspace(0, 1, 100).to(device)     # FE grid (nodes)        
x_train = torch.linspace(0, 1, 1000).to(device)   # Training samples
u_true = torch.sin(2 * torch.pi * x_train)        # Target function 

# Model and Optimizer Setup
model = PiecewiseLinearShapeNN(x_grid, r_adapt=True).to(device)  
optimizer = optim.Adam(model.parameters(), lr=0.005)

# L² Projection via Minimization
# The loss corresponds to ∫ (u_h - u_true)² dx ≈ mean((pred - u_true)²) 
for epoch in range(500):
    optimizer.zero_grad()
    pred = model(x_train)
    loss = ((pred - u_true) ** 2).mean()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.6f}")

# Analytical Target Function (for visualization)
exact_solution = lambda x: torch.sin(2 * torch.pi * x)
exact_derivative_solution = lambda x: 2 * torch.pi * torch.cos(2 * torch.pi * x)

# Visualization
plot_fem_solution(model, u_exact=exact_solution, title="L² Projection of sin(2πx)")
plot_fem_derivative(model, u_exact=exact_derivative_solution, title="Derivative of L² Projection (du/dx)")


