import torch
from torch.autograd import Function
import torch.nn as nn
import cuda_kernel.invert_cuda as invert
import cuda_kernel.solve_cuda as solve


class SolveMaskedFunction(Function):
    @staticmethod
    def forward(ctx, A, b, mask):
        """
        A: (..., n_total, n_total)
        b: (..., n_total)
        mask: (..., n_total) boolean
        """
        x = torch.zeros_like(b)
        # call CUDA launcher
        solve.solve_batched_launcher(
            A.contiguous(),
            b.contiguous(),
            mask.contiguous(),
            x.contiguous()
        )
        # save for backward
        ctx.save_for_backward(A, x, mask)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        A, x, mask = ctx.saved_tensors
        grad_A = torch.zeros_like(A)
        grad_b = torch.zeros_like(x)
        # call CUDA backward launcher
        solve.solve_batched_backward_launcher(
            A.contiguous(),
            x.contiguous(),
            grad_output.contiguous(),
            mask.contiguous(),
            grad_A.contiguous(),
            grad_b.contiguous()
        )
        return grad_A, grad_b, None  # mask has no gradient

class SolveMasked(nn.Module):
    def forward(self, A, b, mask):
        return SolveMaskedFunction.apply(A, b, mask)
    

class InvertMaskedFunction(Function):
    @staticmethod
    def forward(ctx, G, valid_idx):
        G_inv = torch.zeros_like(G)
        invert.invert_batched_launcher(G.contiguous(), valid_idx.to(torch.int32).contiguous(), G_inv.contiguous())
        ctx.save_for_backward(G_inv, valid_idx)
        return G_inv

    @staticmethod
    def backward(ctx, grad_output):
        G_inv, valid_idx = ctx.saved_tensors
        grad_G = torch.zeros_like(G_inv)
        invert.invert_batched_backward_launcher(G_inv.contiguous(), valid_idx.to(torch.int32).contiguous(), grad_output.contiguous(), grad_G.contiguous())
        return grad_G, None

class InvertMasked(nn.Module):
    def forward(self, G, valid_idx):
        return InvertMaskedFunction.apply(G, valid_idx)
