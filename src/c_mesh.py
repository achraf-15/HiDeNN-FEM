from typing import List, Tuple, Dict
import meshzoo
import gmsh
import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_mesh_gmsh(
    length: float = 2.0,
    height: float = 1.0,
    holes: List[Tuple[float, float, float]] = [(0.5, 0.7, 0.12), (1.0, 0.3, 0.15), (1.4, 0.6, 0.1)],
    boundaries: Dict[str, int] = {"up": 0, "down": 0, "right": 2, "left": 1},
    lc: float = 1e-1,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Generate a 2D rectangular triangular mesh with circular holes using gmsh.
    """
    
    gmsh.initialize()
    gmsh.model.add("mesh_with_holes")
    
    # Create rectangle
    rect = gmsh.model.occ.addRectangle(0, 0, 0, length, height)
    
    # Create and subtract holes
    hole_tags = []
    for cx, cy, r in holes:
        hole = gmsh.model.occ.addDisk(cx, cy, 0, r, r)
        hole_tags.append((2, hole))
    
    if hole_tags:
        rect_domain = gmsh.model.occ.cut([(2, rect)], hole_tags)
        domain = rect_domain[0][0][1]
    else:
        domain = rect
    
    gmsh.model.occ.synchronize()
    
    # Set mesh size
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)
    
    # Generate 2D mesh
    gmsh.model.mesh.generate(2)
    
    # Get nodes and elements
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_coords = node_coords.reshape(-1, 3)[:, :2]  # Extract x, y
    
    # Create tag to index mapping
    tag_to_idx = {tag: idx for idx, tag in enumerate(node_tags)}
    
    # Get triangles
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
    triangles = []
    for i, elem_type in enumerate(elem_types):
        if elem_type == 2:  # Triangle
            nodes = elem_node_tags[i].reshape(-1, 3)
            triangles.append(nodes)
    
    if triangles:
        triangles = np.vstack(triangles)
    else:
        triangles = np.array([]).reshape(0, 3)
    
    # Remap to indices
    connectivity = np.array([[tag_to_idx[t] for t in tri] for tri in triangles])
    
    # Get all boundary entities recursively
    geom_boundary_nodes = set()
    
    # Get boundary curves (1D)
    boundary_curves = gmsh.model.getBoundary([(2, domain)], oriented=False, recursive=False)
    
    # For each curve, get all nodes (including endpoints)
    for dim, tag in boundary_curves:
        # Get nodes on the curve itself
        curve_nodes = gmsh.model.mesh.getNodes(dim, tag)[0]
        geom_boundary_nodes.update(curve_nodes)
        
        # Get boundary points (0D) of the curve
        curve_points = gmsh.model.getBoundary([(dim, tag)], oriented=False, recursive=False)
        for pdim, ptag in curve_points:
            point_nodes = gmsh.model.mesh.getNodes(pdim, ptag)[0]
            geom_boundary_nodes.update(point_nodes)
    
    geom_boundary_mask = np.array([tag in geom_boundary_nodes for tag in node_tags])
    
    # Additional check: mark nodes on hole boundaries geometrically
    tol_hole = 1e-6
    for cx, cy, r in holes:
        dist = np.sqrt((node_coords[:, 0] - cx)**2 + (node_coords[:, 1] - cy)**2)
        on_hole = np.abs(dist - r) < tol_hole
        geom_boundary_mask |= on_hole
    
    # Initialize BC masks
    bc_mask = np.zeros(len(node_tags), dtype=bool)
    mn_mask = np.zeros(len(node_tags), dtype=bool)
    
    tol = 1e-6
    
    # Assign boundary conditions
    for face, condition in boundaries.items():
        if condition == 0:
            continue
        
        if face == "up":
            mask_face = np.abs(node_coords[:, 1] - height) < tol
        elif face == "down":
            mask_face = np.abs(node_coords[:, 1] - 0.0) < tol
        elif face == "left":
            mask_face = np.abs(node_coords[:, 0] - 0.0) < tol
        elif face == "right":
            mask_face = np.abs(node_coords[:, 0] - length) < tol
        else:
            continue
        
        if condition == 1:
            bc_mask |= mask_face
        elif condition == 2:
            mn_mask |= mask_face
    
    # Extract Neumann edges
    all_edges = np.vstack([
        connectivity[:, [0, 1]],
        connectivity[:, [1, 2]],
        connectivity[:, [2, 0]]
    ])
    all_edges = np.sort(all_edges, axis=1)
    unique_edges = np.unique(all_edges, axis=0)
    
    # Neumann edges: both nodes in Neumann mask
    neumann_edges = unique_edges[np.all(mn_mask[unique_edges], axis=1)]
    
    gmsh.finalize()
    
    # Convert to torch tensors
    node_coords = torch.tensor(node_coords, dtype=torch.float32)
    connectivity = torch.tensor(connectivity, dtype=torch.long)
    geom_boundary_mask = torch.tensor(geom_boundary_mask, dtype=torch.bool)
    bc_mask = torch.tensor(bc_mask, dtype=torch.bool)
    mn_mask = torch.tensor(mn_mask, dtype=torch.bool)
    neumann_edges = torch.tensor(neumann_edges, dtype=torch.long)

    # Convert the coordinates to dimensionless and return the characteristic scale
    dimless_scale = max(length, height)
    
    return (
        node_coords,
        connectivity,
        geom_boundary_mask,
        bc_mask,
        mn_mask,
        neumann_edges,
        dimless_scale,
    )

import scipy.sparse as sp

def mesh_to_patch(connectivity: torch.Tensor, s: int = 1) -> Tuple[sp.csr_matrix, torch.Tensor]:
    """
    Given mesh connectivity, returns the adjacency matrix (sparse) and the patch tensor.
    
    Args:
        connectivity: (num_elements, nodes_per_element) tensor of mesh connectivity
        s: number of steps to expand the adjacency
    
    Returns:
        A: scipy.sparse.csr_matrix adjacency matrix (num_nodes x num_nodes)
        patch: torch.LongTensor of shape (num_nodes, max_connections) containing
               indices of connected nodes (0-padded)
    """
    num_nodes = connectivity.max().item() + 1

    # Build adjacency matrix
    rows = []
    cols = []
    for tri in connectivity.numpy():
        for i in range(len(tri)):
            for j in range(len(tri)):
                rows.append(tri[i])
                cols.append(tri[j])
    
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    A.setdiag(1)  # Ensure self-connection
    A = A.tocsr()

    # Compute A^s
    As = A.copy()
    for _ in range(s - 1):
        As = As.dot(A)
        As.data = np.ones_like(As.data)  # keep it binary
    
    # Build patch tensor
    patch_list = []
    max_connections = max(As.getnnz(axis=1))
    for i in range(num_nodes):
        neighbors = As[i].indices
        padded = -np.ones(max_connections, dtype=np.int64)
        padded[:len(neighbors)] = neighbors
        patch_list.append(padded)
    
    patch = torch.tensor(np.vstack(patch_list), dtype=torch.long)
    
    return patch


def plot_mesh(node_coords, connectivity, geom_boundary_mask, bc_mask, mn_mask, neumann_edges):
    # convert tensors to numpy
    points = node_coords.cpu().numpy()
    cells = connectivity.cpu().numpy()
    geom_boundary_mask = geom_boundary_mask.cpu().numpy()
    bc_mask = bc_mask.cpu().numpy()
    mn_mask = mn_mask.cpu().numpy()
    neumann_edges = neumann_edges.cpu().numpy()

    plt.figure(figsize=(8, 4))
    
    # thin blue internal mesh lines
    plt.triplot(points[:, 0], points[:, 1], cells, color='blue', linewidth=0.3, alpha=0.6)
    
    # transparent black geometric boundaries
    plt.scatter(points[geom_boundary_mask, 0], points[geom_boundary_mask, 1],
                color='black', s=10, alpha=0.7, label='Geom Boundary')
    
    # red Dirichlet boundary nodes
    plt.scatter(points[bc_mask, 0], points[bc_mask, 1],
                color='red', s=15, label='Dirichlet')
    
    # purple Neumann nodes
    plt.scatter(points[mn_mask, 0], points[mn_mask, 1],
                color='purple', s=20, label='Neumann Nodes')
    
    # purple Neumann edges
    for e in neumann_edges:
        x, y = points[e, 0], points[e, 1]
        plt.plot(x, y, color='purple', linewidth=1.5, alpha=0.9)
    
    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


import random

def plot_mesh_with_patches(node_coords, connectivity, geom_boundary_mask, bc_mask, mn_mask, neumann_edges, patch):
    # convert tensors to numpy
    points = node_coords.cpu().numpy()
    cells = connectivity.cpu().numpy()
    geom_boundary_mask_np = geom_boundary_mask.cpu().numpy()
    bc_mask_np = bc_mask.cpu().numpy()
    mn_mask_np = mn_mask.cpu().numpy()
    neumann_edges_np = neumann_edges.cpu().numpy()
    patch_np = patch.cpu().numpy()

    plt.figure(figsize=(8, 4))
    
    # thin blue internal mesh lines
    plt.triplot(points[:, 0], points[:, 1], cells, color='blue', linewidth=0.3, alpha=0.6)
    
    # transparent black geometric boundaries
    plt.scatter(points[geom_boundary_mask_np, 0], points[geom_boundary_mask_np, 1],
                color='black', s=10, alpha=0.7, label='Geom Boundary')
    
    # red Dirichlet boundary nodes
    plt.scatter(points[bc_mask_np, 0], points[bc_mask_np, 1],
                color='red', s=15, label='Dirichlet')
    
    # purple Neumann nodes
    plt.scatter(points[mn_mask_np, 0], points[mn_mask_np, 1],
                color='purple', s=20, label='Neumann Nodes')
    
    # purple Neumann edges
    for e in neumann_edges_np:
        x, y = points[e, 0], points[e, 1]
        plt.plot(x, y, color='purple', linewidth=1.5, alpha=0.9)
    
    # --- Sample a random internal node ---
    all_nodes = np.arange(points.shape[0])
    internal_nodes = all_nodes[~geom_boundary_mask_np]
    random_internal = random.choice(internal_nodes)
    internal_patch_nodes = patch_np[random_internal]
    internal_patch_nodes = internal_patch_nodes[internal_patch_nodes != -1]  # remove padding
    
    plt.scatter(points[random_internal, 0], points[random_internal, 1], color='green', s=50, label='Random Node')
    plt.scatter(points[internal_patch_nodes, 0], points[internal_patch_nodes, 1], 
                color='lime', s=35, alpha=0.7, label='Patch Nodes (internal)')
    
    # --- Sample a random geometric boundary node ---
    boundary_nodes = all_nodes[geom_boundary_mask_np]
    random_boundary = random.choice(boundary_nodes)
    boundary_patch_nodes = patch_np[random_boundary]
    boundary_patch_nodes = boundary_patch_nodes[boundary_patch_nodes != -1]  # remove padding
    
    plt.scatter(points[random_boundary, 0], points[random_boundary, 1], color='orange', s=50, label='Boundary Node')
    plt.scatter(points[boundary_patch_nodes, 0], points[boundary_patch_nodes, 1], 
                color='yellow', s=35, alpha=0.7, label='Patch Nodes (boundary)')
    
    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.legend(loc='upper right', markerscale=1.2)
    plt.show()

