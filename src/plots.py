import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from .utils import triangle_gauss_points, interval_gauss_points

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


def plot_displacement_magnitude(model, L0=1.0, U0=1.0):
    # Extract nodal data
    coords = L0 * model.coords.cpu().detach().numpy()     # [Nnodes, 2] # Convert from dimensionless to physical 
    triangles = model.connectivity.cpu().detach().numpy() # [Nelems, 3]
    u_vals = U0 * model.values.cpu().detach().numpy()     # [Nnodes, 2] # Convert from dimensionless to physical

    # Compute displacement magnitude per node
    u_mag = np.linalg.norm(u_vals, axis=1)  # [Nnodes]

    triang = tri.Triangulation(coords[:,0], coords[:,1], triangles)

    # Plot using tripcolor
    plt.figure(figsize=(8,4))
    pc = plt.tripcolor(triang, u_mag, shading='flat', edgecolors='k', linewidth=0.2, cmap='viridis')
    plt.colorbar(pc, label='||u|| (m)')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Displacement field (magnitude)')
    plt.gca().set_aspect('equal')
    plt.show()


def plot_von_mises(model, E=10e9, nu=0.3, L0=1.0, U0=1.0):
    # Extract data
    coords = L0  *model.coords.cpu().detach().numpy()     # [Nnodes,2] # Convert from dimensionless to physical
    triangles = model.connectivity.cpu().detach().numpy() # [Nelems,3]

    # Evaluate displacement gradients at element centroids
    Nelems = model.Nelems
    Nnodes = model.Nnodes
    n_g = 37
    xg, wg = triangle_gauss_points(order=n_g, device=model.device, dtype=model.dtype)

    x_eval = xg.unsqueeze(0).expand(Nelems, n_g, 2).reshape(-1, 2).to(model.device, dtype=model.dtype)
    elem_id = torch.arange(Nelems, device=model.device).unsqueeze(1).repeat(1, n_g).reshape(-1)
    w_eval = wg.unsqueeze(0).repeat(Nelems, 1).to(model.device, dtype=model.dtype).cpu().detach().numpy()

    _, _, grad_u =  model(x_eval, elem_id)  # [Nelems, n_g, dim_u, 2]

    # Strain components (infinitesimal)
    grad_u = U0 * grad_u.cpu().detach().numpy().reshape(Nelems, n_g, 2, 2)  # Convert from dimensionless to physical 
    eps_xx = grad_u[:, :, 0, 0]
    eps_yy = grad_u[:, :, 1, 1]
    eps_xy = 0.5 * (grad_u[:, :, 0, 1] + grad_u[:, :, 1, 0])

    # Plane stress von Mises stress
    sigma_xx = E/(1-nu**2)*(eps_xx + nu*eps_yy)
    sigma_yy = E/(1-nu**2)*(eps_yy + nu*eps_xx)
    sigma_xy = E/(1+nu)*eps_xy

    von_mises = np.sqrt(sigma_xx**2 - sigma_xx*sigma_yy + sigma_yy**2 + 3*sigma_xy**2)
    von_mises_mean = np.sum(von_mises * w_eval, axis=1)

    triang = tri.Triangulation(coords[:,0], coords[:,1], triangles)

    # Plot von Mises stress
    plt.figure(figsize=(8,4))
    pc = plt.tripcolor(triang, facecolors=von_mises_mean, edgecolors='b', linewidth=0.2, cmap='inferno')
    plt.colorbar(pc, label='Von Mises stress [Pa]')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('von Mises stress concentration (Element averaging)')
    plt.gca().set_aspect('equal')
    plt.show()

def plot_von_mises_tricontourfd(model, E=10e9, nu=0.3, L0=1.0, U0=1.0):
    # Extract data
    coords = L0  *model.coords.cpu().detach().numpy()     # [Nnodes,2] # Convert from dimensionless to physical
    triangles = model.connectivity.cpu().detach().numpy() # [Nelems,3]

    # Evaluate displacement gradients at element centroids
    Nelems = model.Nelems
    Nnodes = model.Nnodes
    n_g = 37
    xg, wg = triangle_gauss_points(order=n_g, device=model.device, dtype=model.dtype)

    x_eval = xg.unsqueeze(0).expand(Nelems, n_g, 2).reshape(-1, 2).to(model.device, dtype=model.dtype)
    elem_id = torch.arange(Nelems, device=model.device).unsqueeze(1).repeat(1, n_g).reshape(-1)
    w_eval = wg.unsqueeze(0).repeat(Nelems, 1).to(model.device, dtype=model.dtype).cpu().detach().numpy()

    _, _, grad_u =  model(x_eval, elem_id)  # [Nelems, n_g, dim_u, 2]

    # Strain components (infinitesimal)
    grad_u = U0 * grad_u.cpu().detach().numpy().reshape(Nelems, n_g, 2, 2)  # Convert from dimensionless to physical 
    eps_xx = grad_u[:, :, 0, 0]
    eps_yy = grad_u[:, :, 1, 1]
    eps_xy = 0.5 * (grad_u[:, :, 0, 1] + grad_u[:, :, 1, 0])

    # Plane stress von Mises stress
    sigma_xx = E/(1-nu**2)*(eps_xx + nu*eps_yy)
    sigma_yy = E/(1-nu**2)*(eps_yy + nu*eps_xx)
    sigma_xy = E/(1+nu)*eps_xy

    von_mises = np.sqrt(sigma_xx**2 - sigma_xx*sigma_yy + sigma_yy**2 + 3*sigma_xy**2)
    sigma_elem = np.sum(von_mises * w_eval, axis=1)

    sigma_node = np.zeros(Nnodes)
    counts = np.zeros(Nnodes)

    for e in range(Nelems):
        val = sigma_elem[e]
        for n in triangles[e]:
            sigma_node[n] += val
            counts[n] += 1

    sigma_node /= np.maximum(counts, 1)

    triang = tri.Triangulation(coords[:,0], coords[:,1], triangles)

    plt.figure(figsize=(8,4))
    cf = plt.tricontourf(triang, sigma_node, levels=50, cmap='inferno')
    plt.triplot(triang, color='k', linewidth=0.3, alpha=0.3)
    plt.colorbar(cf, label='Von Mises stress [Pa]')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('von Mises stress concentration (Node averaging)')
    plt.gca().set_aspect('equal')
    plt.show()


def plot_model_mesh(model, L0=1.0):
    # convert tensors to numpy
    
    points = L0 * model.coords.cpu().detach().numpy() # Convert from dimensionless to physical coordinates
    cells = model.connectivity.cpu().detach().numpy()
    geom_boundary_mask = model.boundary_mask.cpu().detach().numpy()
    bc_mask = model.dirichlet_mask.cpu().detach().numpy()
    neumann_edges = model.neumann_edges.cpu().detach().numpy()

    plt.figure(figsize=(8, 4))
    
    # thin blue internal mesh lines
    plt.triplot(points[:, 0], points[:, 1], cells, color='blue', linewidth=0.3, alpha=0.6)
    
    # transparent black geometric boundaries
    plt.scatter(points[geom_boundary_mask, 0], points[geom_boundary_mask, 1],
                color='black', s=10, alpha=0.7, label='Geom Boundary')
    
    # red Dirichlet boundary nodes
    plt.scatter(points[bc_mask, 0], points[bc_mask, 1],
                color='red', s=15, label='Dirichlet Nodes')
    
    # purple Neumann edges
    for e in neumann_edges:
        x, y = points[e, 0], points[e, 1]
        plt.plot(x, y, color='purple', linewidth=1.5, alpha=0.9)
    
    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()