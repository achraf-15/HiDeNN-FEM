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
        """
        Return the full nodal displacement vector including boundary conditions.
        """
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
        elem_idx = torch.searchsorted(self.grid, x_eval) - 1  
        elem_idx = elem_idx.clamp(0, len(self.grid)-2)  # enforce valid range

        # 2. Get local node coordinates and nodal values
        x_i = self.grid[elem_idx]           # [M]
        x_ip1 = self.grid[elem_idx + 1]     # [M]
        u_full = self.u_full                # [N]
        u_i = u_full[elem_idx]              # [1]
        u_ip1 = u_full[elem_idx + 1]        # [1]

        # 3. Compute local shape functions
        N1 = (x_ip1 - x_eval) / (x_ip1 - x_i).clamp(self.epsilon)  # [M]
        N2 = (x_eval - x_i) / (x_ip1 - x_i).clamp(self.epsilon)   # [M]

        # 4. Compute local field
        u_per_elem = u_i * N1 + u_ip1 * N2

        return u_per_elem