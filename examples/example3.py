import torch
import torch.optim as optim

from src.models import PiecewiseLinearShapeNN
from src.utils import gauss_legendre_points_weights
from src.plots import plot_fem_solution, plot_fem_derivative

# ----------------------------------------------------------------------
# Problem Definition: 1D Bar under distributed load
# Governed by:  -(E * u')' = b(x)
# Dirichlet BCs: u(0) = u(L) = 0
# Minimize total potential energy Π(u) = ∫ [ (1/2)E(u')² - b(x)u ] dx
# ----------------------------------------------------------------------

# Body Force
def b_force(x):
    """
    Body force distribution b(x) consisting of two Gaussian-like bumps.
    """
    N1 = 4*torch.pi**2 * (x-2.5)**2 - 2*torch.pi
    D1 = torch.exp(torch.pi*(x-2.5)**2)
    N2 = 8*torch.pi**2 * (x-7.5)**2 - 4*torch.pi
    D2 = torch.exp(torch.pi*(x-7.5)**2)
    return -N1/D1 - N2/D2

# Energy based loss function (weak formulation)
def energy_loss(model, xi, wi, b_force, E, L=10.0):
    """
    Compute the energy-based loss for a rod under a body force.

    Parameters:
    - model: The neural network model representing the displacement field u(x).
    - xi, wi: Gauss Quadrature points and weights.
    - b: Function for body force.
    - E: Young's modulus.
    - L: Lenght of the bar.

    Returns:
    - loss: The computed energy-based loss.
    """
    with torch.no_grad():

        grid = model.grid             # [N]

        x_i = grid[:-1].unsqueeze(1)   # [nelem,1]
        x_ip1 = grid[1:].unsqueeze(1)  # [nelem,1]

        # Map Gauss points to physical coordinates
        xq = 0.5 * (x_ip1 - x_i) * xi + 0.5 * (x_ip1 + x_i)  # [nelem, n_gauss]
        wq = 0.5 * (x_ip1 - x_i) * wi                          # [nelem, n_gauss]

    xq.requires_grad_(True)
    u = model(xq)  # Displacement field u(x)

    # Compute the derivative of u with respect to x
    du_dx = torch.autograd.grad(u, xq, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    # Compute the elastic strain energy
    elastic_energy = 0.5 * E * du_dx**2

    # Compute the potential energy due to body force
    potential_energy = b_force(xq) * u

    # Compute the total potential energy
    total_energy = elastic_energy - potential_energy

    # weighted sum: ∑ w_i f(x_i)
    loss = torch.sum(wq * total_energy)

    return loss 


# Problem Parameters
L = 10.0          # Bar length
E = 175.0         # Young's modulus
u0, uN = 0.0, 0.0 # Dirichlet BCs

grid_pts = 89     # Number of nodal points
n_gauss = 2       # Quadrature points per element
r_adapt = True    # Adaptive refinement flag

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_grid = torch.linspace(0, L, grid_pts).to(device)     
xi, wi = gauss_legendre_points_weights(n_gauss, device=device)
  
# Model and Optimizer Setup
model = PiecewiseLinearShapeNN(x_grid, r_adapt=r_adapt, u0=u0, uN=uN).to(device)  
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Optimization Loop (Energy Minimization)
for epoch in range(4000):
    optimizer.zero_grad()
    loss = energy_loss(model, xi, wi, b_force, E=E)
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.6f}")

# Analytical Solution (for validation)
def u_true(x, E):
    pi = torch.tensor(torch.pi)
    term1 = (1 / E) * (torch.exp(-pi * (x - 2.5)**2) - torch.exp(-6.25 * pi))
    term2 = (2 / E) * (torch.exp(-pi * (x - 7.5)**2) - torch.exp(-56.25 * pi))
    constant = torch.exp(-6.25 * pi) - torch.exp(-56.25 * pi)
    linear = constant * x / (10 * E)
    return term1 + term2 - linear

def du_dx_true(x, E):
    pi = torch.tensor(torch.pi)
    term1 = (2 / E) * (-pi * (x - 2.5) * torch.exp(-pi * (x - 2.5)**2))
    term2 = (4 / E) * (-pi * (x - 7.5) * torch.exp(-pi * (x - 7.5)**2))
    constant = torch.exp(-6.25 * pi) - torch.exp(-56.25 * pi)
    linear = constant * x / (10 * E)
    return term1 + term2 - linear

# Visualization
u_exact = lambda x: u_true(x, E)
plot_fem_solution(model, u_exact=u_exact, title="FEM Solution (Displacement)")
du_exact = lambda x: du_dx_true(x, E)
plot_fem_derivative(model, u_exact=du_exact, title="FEM Derivative (du/dx)")