import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from .utils import triangle_gauss_points, interval_gauss_points


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
    pc = plt.tripcolor(triang, u_mag, shading='flat', edgecolors='b', linewidth=0.2, cmap='inferno')
    plt.colorbar(pc, label='||u|| (m)')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('c-HiDeNN displacement field (magnitude)')
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
    plt.title('HiDeNN von Mises stress concentration (Element averaging)')
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
    plt.title('HiDeNN von Mises stress concentration (Node averaging)')
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