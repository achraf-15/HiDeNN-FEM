import torch
import torch.nn as nn
import torch.nn.functional as F


class PiecewiseLinearShapeNN(nn.Module):
    def __init__(self, node_coords, r_adapt=False, u0=None, uN=None):
        super().__init__()
        self.N = len(node_coords)
        self.r_adapt = r_adapt

        # Fixed boundary nodes 
        self.register_buffer('x0', node_coords[0:1])      # first node
        self.register_buffer('xN', node_coords[-1:])      # last node

        if self.r_adapt and self.N > 2: #r-adaptivity (only for inner nodes, via positive increments)
            # learn increments 
            init_diff = node_coords[1:] - node_coords[:-1]
            self.x_increments = nn.Parameter(init_diff)  # inner increments
        else:
            self.register_buffer('x_inner', node_coords[1:-1])

        # Optional boundary nodal values
        if u0 is not None:
            self.register_buffer("u0_fixed", torch.tensor([u0], dtype=torch.float32))
        else:
            self.u0_fixed = None

        if uN is not None:
            self.register_buffer("uN_fixed", torch.tensor([uN], dtype=torch.float32))
        else:
            self.uN_fixed = None

        # nodal DOFs
        if (self.u0_fixed is not None) and (self.uN_fixed is not None):  
            self.u = nn.Parameter(torch.zeros(self.N-2))
        elif (self.u0_fixed is not None) ^ (self.uN_fixed is not None):  
            self.u = nn.Parameter(torch.zeros(self.N-1))
        else :
            self.u = nn.Parameter(torch.zeros(self.N))

        # eps to avoid division by 0
        self.epsilon = 1e-10

    @property
    def grid(self):
        if self.r_adapt and self.N > 2:
            # first/last node
            x0, xN = self.x0, self.xN
            increments = torch.clamp(F.softplus(self.x_increments), min=1e-6)
            cum_increments = torch.cumsum(increments, dim=0) # length N-2
            x_inner = x0 + (xN - x0) * cum_increments / cum_increments[-1]
            return torch.cat([x0, x_inner], dim=0)
        else:
            x_inner = self.x_inner
            return torch.cat([self.x0, x_inner, self.xN], dim=0)
        
    @property
    def u_full(self):
        if self.u0_fixed is not None and self.uN_fixed is not None:
            return torch.cat([self.u0_fixed, self.u.view(-1), self.uN_fixed])
        elif self.u0_fixed is not None:
            return torch.cat([self.u0_fixed, self.u.view(-1)])
        elif self.uN_fixed is not None:
            return torch.cat([self.u, self.uN_fixed])
        else:
            return self.u.view(-1)
        
            
    def forward(self, x_eval):
        # 1. Find element index for each x_eval
        grid = self.grid  # compute only once
        elem_idx = torch.searchsorted(grid, x_eval) - 1  
        elem_idx = elem_idx.clamp(0, self.N-2)  # enforce valid range

        # 2. Get local node coordinates and nodal values
        x_i = grid[elem_idx]                # [M]
        x_ip1 = grid[elem_idx + 1]          # [M]
        u_full = self.u_full                # [N]
        u_i = u_full[elem_idx]              # [1]
        u_ip1 = u_full[elem_idx + 1]        # [1]

        # 3. Compute local shape functions
        N1 = (x_ip1 - x_eval) / (x_ip1 - x_i).clamp(self.epsilon)  # [M]
        N2 = (x_eval - x_i) / (x_ip1 - x_i).clamp(self.epsilon)   # [M]

        # 4. Compute local field
        u_per_elem = u_i * N1 + u_ip1 * N2

        return u_per_elem
    

class PiecewiseLinearShapeNN2D(nn.Module):
    def __init__(self,  grid_x, grid_y, boundary_mask_x=None, boundary_mask_y=None, r_adapt=False, u_fixed=None):
        super().__init__()
        # Store separate 1D grids
        self.Nx = grid_x.numel()
        self.Ny = grid_y.numel()
        self.r_adapt = r_adapt

        # Store initial grids for potential regularization or reference
        self.register_buffer("initial_x_grid", grid_x.clone())
        self.register_buffer("initial_y_grid", grid_y.clone())

        # Boundary coordinates (first and last points)
        self.register_buffer("x0", grid_x.flatten()[0:1])     
        self.register_buffer("xN", grid_x.flatten()[-1:])  
        self.register_buffer("y0", grid_y.flatten()[0:1])     
        self.register_buffer("yN", grid_y.flatten()[-1:])   

        if self.r_adapt and max(self.Nx,self.Ny) > 2: #r-adaptivity (only for inner nodes, via positive increments)
            # learn increments 
            init_diff_x = grid_x[1:] - grid_x[:-1]  
            self.increments_x = nn.Parameter(init_diff_x) 
            init_diff_y = grid_y[1:] - grid_y[:-1]  
            self.increments_y = nn.Parameter(init_diff_y) 
        else:
            self.register_buffer("x_grid_inner", grid_x[1:-1])
            self.register_buffer("y_grid_inner", grid_y[1:-1])


         # --- Boundary masks (1D for x and y) ---
        if boundary_mask_x is None:
            boundary_mask_x = torch.zeros(self.Nx, dtype=torch.bool)
            boundary_mask_x[0] = boundary_mask_x[-1] = True
        if boundary_mask_y is None:
            boundary_mask_y = torch.zeros(self.Ny, dtype=torch.bool)
            boundary_mask_y[0] = boundary_mask_y[-1] = True

        self.register_buffer("boundary_mask_x", boundary_mask_x)
        self.register_buffer("boundary_mask_y", boundary_mask_y)

        # 2D node mask (True = any boundary in x or y)
        self.register_buffer("node_mask", self.boundary_mask_x[:, None] | self.boundary_mask_y[None, :])

        if u_fixed is not None:
            self.register_buffer("u_fixed", torch.tensor([u_fixed], dtype=torch.float32))
        else:
            self.u_fixed = None

        # nodal DOFs 
        self.u = nn.Parameter(torch.randn(self.Nx, self.Ny))

        self.epsilon = 1e-10

    @property
    def grid(self):
        if self.r_adapt and max(self.Nx,self.Ny) > 2:
            # compute increments and cumulative sum per dim
            incr_x = torch.clamp(F.softplus(self.increments_x), min=1e-6)
            cum_x = torch.cumsum(incr_x, dim=0)
            x_grid_inner = self.x0 + (self.xN - self.x0) * cum_x / cum_x[-1]
            incr_y = torch.clamp(F.softplus(self.increments_y), min=1e-6)
            cum_y = torch.cumsum(incr_y, dim=0)
            y_grid_inner = self.y0 + (self.yN - self.y0) * cum_y / cum_y[-1]
        else:
            x_grid_inner = torch.cat([self.x_grid_inner, self.xN], dim=0)
            y_grid_inner = torch.cat([self.y_grid_inner, self.yN], dim=0)

        # concatenate first and last nodes
        x_grid_full = torch.cat([self.x0, x_grid_inner], dim=0)
        y_grid_full = torch.cat([self.y0, y_grid_inner], dim=0)

        # Apply 1D boundary masks (keep coordinates fixed)
        x_grid_full = torch.where(self.boundary_mask_x, self.initial_x_grid, x_grid_full)
        y_grid_full = torch.where(self.boundary_mask_y, self.initial_y_grid, y_grid_full)

        return x_grid_full, y_grid_full  
        
    @property
    def u_full(self):
        if self.u_fixed is not None:
            # concatenate fixed nodes and free DOFs
            return torch.where(self.node_mask, self.u_fixed, self.u)
        else:
            u_full = self.u
        return u_full
        
            
    def forward(self, x_eval):
        # 1. Find element indices in each dim, using searchsorted
        grid_x, grid_y = self.grid  # compute only once 
        idx_x = torch.searchsorted(grid_x, x_eval[:,0].contiguous()) - 1
        idx_y = torch.searchsorted(grid_y, x_eval[:,1].contiguous()) - 1  
        idx_x = idx_x.clamp(0, self.Nx-2)  
        idx_y = idx_y.clamp(0, self.Ny-2)

        # 2. Get local node coordinates and nodal values
        # Local nodes in each dim
        x_i = grid_x[idx_x]      # (M,)
        x_ip1 = grid_x[idx_x+1]  # (M,)
        y_i = grid_y[idx_y]
        y_ip1 = grid_y[idx_y+1]
                                      
        # Node values
        u_full = self.u_full
        u00 = u_full[idx_x,   idx_y]
        u10 = u_full[idx_x+1, idx_y]
        u01 = u_full[idx_x,   idx_y+1]
        u11 = u_full[idx_x+1, idx_y+1]

        # 3. Compute local shape functions
        N1x = (x_ip1 - x_eval[:,0]) / (x_ip1 - x_i).clamp(self.epsilon)
        N2x = (x_eval[:,0] - x_i) / (x_ip1 - x_i).clamp(self.epsilon)
        N1y = (y_ip1 - x_eval[:,1]) / (y_ip1 - y_i).clamp(self.epsilon)
        N2y = (x_eval[:,1] - y_i) / (y_ip1 - y_i).clamp(self.epsilon)

        # 4. Compute local field
        # Bilinear interpolation
        u_h = N1x * N1y * u00 + N2x * N1y * u10 + N1x * N2y * u01 + N2x * N2y * u11

        return u_h