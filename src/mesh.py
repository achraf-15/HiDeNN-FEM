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
    
    return (
        node_coords,
        connectivity,
        geom_boundary_mask,
        bc_mask,
        mn_mask,
        neumann_edges,
    )

def generate_mesh(
    length: float = 2.0,
    height: float = 1.0,
    holes: List[Tuple[float, float, float]] = [(0.5, 0.7, 0.12), (1.0, 0.3, 0.15), (1.4, 0.6, 0.1)],
    boundaries: Dict[str, int] = {"up": 0, "down": 0, "right": 2, "left": 1},
    nx: int = 100,
    ny: int = 50,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Generate a 2D rectangular triangular mesh with circular holes and boundary condition masks.

    Returns
    -------
    node_coords : (N,2) torch.Tensor
        Coordinates of mesh nodes.
    connectivity : (M,3) torch.Tensor
        Triangle connectivity.
    geom_boundary_mask : (N,) torch.BoolTensor
        Geometric boundary nodes (outer domain or hole).
    bc_mask : (N,) torch.BoolTensor
        Dirichlet boundary mask.
    mn_mask : (N,) torch.BoolTensor
        Neumann boundary mask.
    neumann_edges : (E,2) torch.LongTensor
        Edges under Neumann conditions.
    """

    # --- Base rectangle mesh
    x = np.linspace(0.0, length, nx)
    y = np.linspace(0.0, height, ny)
    points, cells = meshzoo.rectangle_tri(x, y, variant="zigzag")
    points = np.array(points)
    cells = np.array(cells)

    # --- Remove points inside holes
    mask = np.ones(len(points), dtype=bool)
    for cx, cy, r in holes:
        dx, dy = points[:, 0] - cx, points[:, 1] - cy
        mask &= (dx**2 + dy**2) > r**2

    points_kept = points[mask]

    # --- Remap indices
    old_to_new = -np.ones(len(points), dtype=int)
    old_to_new[mask] = np.arange(points_kept.shape[0])

    # --- Keep valid triangles
    cells_kept = []
    geom_boundary_mask = np.zeros(len(points_kept), dtype=bool)
    tol = 1e-6

    for tri in cells:
        if np.all(mask[tri]):
            cells_kept.append(old_to_new[tri])
        else:
            # partially inside triangle → boundary node
            for p in tri:
                if mask[p]:
                    geom_boundary_mask[old_to_new[p]] = True
    cells_kept = np.array(cells_kept)

    # --- Mark outer rectangle boundaries
    geom_boundary_mask |= (
        (np.abs(points_kept[:, 0] - 0.0) < tol)
        | (np.abs(points_kept[:, 0] - length) < tol)
        | (np.abs(points_kept[:, 1] - 0.0) < tol)
        | (np.abs(points_kept[:, 1] - height) < tol)
    )

    # --- Initialize boundary condition masks
    bc_mask = np.zeros(len(points_kept), dtype=bool)  # Dirichlet
    mn_mask = np.zeros(len(points_kept), dtype=bool)  # Neumann

    # --- Assign boundary condition masks
    for face, condition in boundaries.items():
        if condition == 0:
            continue

        if face == "up":
            mask_face = np.abs(points_kept[:, 1] - height) < tol
        elif face == "down":
            mask_face = np.abs(points_kept[:, 1] - 0.0) < tol
        elif face == "left":
            mask_face = np.abs(points_kept[:, 0] - 0.0) < tol
        elif face == "right":
            mask_face = np.abs(points_kept[:, 0] - length) < tol
        else:
            continue

        if condition == 1:
            bc_mask |= mask_face
        elif condition == 2:
            mn_mask |= mask_face

    # --- Extract boundary edges
    all_edges = np.vstack(
        [cells_kept[:, [0, 1]], cells_kept[:, [1, 2]], cells_kept[:, [2, 0]]]
    )
    all_edges = np.sort(all_edges, axis=1)
    unique_edges = np.unique(all_edges, axis=0)

    # Neumann edges → both nodes in Neumann mask
    neumann_edges = unique_edges[np.all(mn_mask[unique_edges], axis=1)]

    # --- Convert to torch tensors
    node_coords = torch.tensor(points_kept, dtype=torch.float32)
    connectivity = torch.tensor(cells_kept, dtype=torch.long)
    geom_boundary_mask = torch.tensor(geom_boundary_mask, dtype=torch.bool)
    bc_mask = torch.tensor(bc_mask, dtype=torch.bool)
    mn_mask = torch.tensor(mn_mask, dtype=torch.bool)
    neumann_edges = torch.tensor(neumann_edges, dtype=torch.long)

    return (
        node_coords,
        connectivity,
        geom_boundary_mask,
        bc_mask,
        mn_mask,
        neumann_edges,
    )


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
