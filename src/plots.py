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


def plot_2d_solution(model, u_exact=None, n_eval=100):
    device = model.u.device
    grid_x, grid_y = model.grid

    X = torch.linspace(grid_x[0], grid_x[-1], n_eval, device=device)
    Y = torch.linspace(grid_y[0], grid_y[-1], n_eval, device=device)
    XX, YY = torch.meshgrid(X, Y, indexing="ij")
    XY = torch.stack([XX.flatten(), YY.flatten()], dim=1)
    u_pred = model(XY).detach().cpu().numpy().reshape(n_eval, n_eval)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(XX.cpu().numpy(), YY.cpu().numpy(), u_pred, cmap="viridis", alpha=0.8)

    if u_exact is not None:
        u_true_vals = u_exact(XX, YY)
        ax.plot_surface(XX.cpu().numpy(), YY.cpu().numpy(), u_true_vals, cmap="coolwarm", alpha=0.5)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u(x,y)")
    plt.title("2D Piecewise Linear FEM Approximation")
    plt.show()

def plot_2d_derivatives(model, n_eval=50, title="FEM Derivatives"):
    device = model.u.device
    grid_x, grid_y = model.grid

    # create evaluation points
    X = torch.linspace(grid_x[0], grid_x[-1], n_eval, device=device)
    Y = torch.linspace(grid_y[0], grid_y[-1], n_eval, device=device)
    XX, YY = torch.meshgrid(X, Y, indexing="ij")
    XY = torch.stack([XX.flatten(), YY.flatten()], dim=1).clone().requires_grad_(True)

    # evaluate model
    u_pred = model(XY)
    ones = torch.ones_like(u_pred)

    # compute derivatives
    du_dXY = torch.autograd.grad(u_pred, XY, grad_outputs=ones, create_graph=False)[0]  # shape (M, 2)
    du_dx = du_dXY[:,0].reshape(n_eval, n_eval).detach().cpu().numpy()
    du_dy = du_dXY[:,1].reshape(n_eval, n_eval).detach().cpu().numpy()

    XX_np = XX.cpu().numpy()
    YY_np = YY.cpu().numpy()

    # plot du/dx
    fig = plt.figure(figsize=(14,6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(XX_np, YY_np, du_dx, cmap="viridis", alpha=0.8)
    ax1.set_title("du/dx")
    ax1.set_xlabel("x"); ax1.set_ylabel("y")

    # plot du/dy
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(XX_np, YY_np, du_dy, cmap="viridis", alpha=0.8)
    ax2.set_title("du/dy")
    ax2.set_xlabel("x"); ax2.set_ylabel("y")

    plt.suptitle(title)
    plt.show()
