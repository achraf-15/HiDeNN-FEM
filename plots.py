import torch
import matplotlib.pyplot as plt

def compute_du_dx_per_element(model):
    """
    Compute du/dx per element using torch.autograd, respecting the actual grid segments.
    Returns:
        x_start: start coordinates of each element (for staircase plotting)
        du_dx_elem: derivative per element (piecewise constant)
    """
    grid = model.grid.detach()
    du_dx_elem = []

    for i in range(len(grid) - 1):
        x0 = grid[i].unsqueeze(0)  # start of element
        x1 = grid[i + 1]

        # Evaluate gradient at both endpoints to ensure correct autograd
        # Take a representative point inside the element (we can pick start or a tiny offset inside)
        x = ((x0 + x1) / 2).clone().requires_grad_(True)
        u = model(x)
        du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=False)[0]

        du_dx_elem.append(du_dx.item())

    return torch.tensor(du_dx_elem)


def plot_fem_solution(model, u_exact=None, title="FEM Solution"):
    """
    Plot FEM displacement solution.
    """
    with torch.no_grad():
        x_eval = torch.linspace(model.grid[0], model.grid[-1], 1000, device=model.grid.device)
        u_pred = model(x_eval).detach().cpu().numpy()
        x_eval_np = x_eval.cpu().numpy()

    plt.figure(figsize=(8,5))
    plt.plot(x_eval_np, u_pred, label="FEM solution", color="blue")

    if u_exact is not None:
        if callable(u_exact):
            u_exact_vals = u_exact(x_eval).detach().cpu().numpy()
        else:
            u_exact_vals = u_exact.detach().cpu().numpy()
        plt.plot(x_eval_np, u_exact_vals, '--', label="Exact solution", color="red")

    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_fem_derivative(model, u_exact=None, title="FEM Derivative du/dx"):
    """
    Plot FEM derivative du/dx as staircase using torch.autograd,
    respecting each element segment.
    """
    du_dx_elem = compute_du_dx_per_element(model)
    du_dx_elem_np = du_dx_elem.numpy()

    # Build staircase by repeating the derivative for each element segment
    x_plot = []
    y_plot = []
    grid = model.grid.cpu().detach().numpy()
    for i in range(len(du_dx_elem_np)):
        x_plot.extend([grid[i], grid[i+1]])
        y_plot.extend([du_dx_elem_np[i], du_dx_elem_np[i]])

    plt.figure(figsize=(8,5))
    plt.plot(x_plot, y_plot, label="FEM derivative", color="green")

    if u_exact is not None:
        if callable(u_exact):
            u_exact_vals = u_exact(torch.tensor(grid)).detach().cpu().numpy()
        else:
            u_exact_vals = u_exact.cpu().numpy()
        plt.plot(grid, u_exact_vals, '--', label="Exact derivative", color="orange")

    plt.xlabel("x")
    plt.ylabel("du/dx")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
