import torch
from torch.autograd import Function
import torch.nn as nn
import cuda_kernel.invert_cuda as invert

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
