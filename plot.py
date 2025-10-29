import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_displacement_magnitude(model):
    # Extract nodal data
    coords = model.coords.cpu().detach().numpy()           # [Nnodes, 2]
    triangles = model.connectivity.cpu().detach().numpy() # [Nelems, 3]
    u_vals = model.u_full.cpu().detach().numpy()          # [Nnodes, 2]

    # Compute displacement magnitude per node
    u_mag = np.linalg.norm(u_vals, axis=1)  # [Nnodes]

    # Compute mean magnitude per triangle
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


def plot_von_mises(model):
    # Extract data
    coords = model.coords.cpu().detach().numpy()           # [Nnodes,2]
    triangles = model.connectivity.cpu().detach().numpy() # [Nelems,3]

    # Evaluate displacement gradients at element centroids
    n_elem = model.Nelems
    xi = np.array([[1/3, 1/3]])  # centroid
    x_eval = torch.tensor(xi, dtype=model.coords.dtype, device=model.device).expand(n_elem, 2)
    elem_id = torch.arange(n_elem, device=model.device)
    _, _, grad_u = model(x_eval, elem_id)  # [Nelems, dim_u, 2]

    # Strain components (infinitesimal)
    grad_u = grad_u.cpu().detach().numpy()
    eps_xx = grad_u[:,0,0]
    eps_yy = grad_u[:,1,1]
    eps_xy = 0.5*(grad_u[:,0,1] + grad_u[:,1,0])

    # Plane stress von Mises stress
    E = 10e9
    nu = 0.3
    sigma_xx = E/(1-nu**2)*(eps_xx + nu*eps_yy)
    sigma_yy = E/(1-nu**2)*(eps_yy + nu*eps_xx)
    sigma_xy = E/(1+nu)*eps_xy
    von_mises = np.sqrt(sigma_xx**2 - sigma_xx*sigma_yy + sigma_yy**2 + 3*sigma_xy**2)

    # Plot von Mises stress
    plt.figure(figsize=(8,4))
    plt.tripcolor(coords[:,0], coords[:,1], triangles, facecolors=von_mises, edgecolors='b', linewidth=0.2, cmap='inferno')
    plt.colorbar(label='Von Mises stress [Pa]')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('HiDeNN von Mises stress concentration')
    plt.gca().set_aspect('equal')
    plt.show()