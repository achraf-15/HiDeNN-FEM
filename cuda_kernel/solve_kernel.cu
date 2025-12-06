#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cmath>

#include "algo.cuh"  // You can reuse small LDLT/LU solver functions here

// ----------------------
// Forward kernel
// ----------------------
template <typename scalar_t>
__global__ void solve_batched_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ b,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ x,
    const int M,
    const int node_per_elem,
    const int n_total)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_nodes = M * node_per_elem;
    if (tid >= total_nodes) return;

    int elem = tid / node_per_elem;
    int node = tid % node_per_elem;

    const scalar_t* A_ptr = A + elem*node_per_elem*n_total*n_total + node*n_total*n_total;
    const scalar_t* b_ptr = b + elem*node_per_elem*n_total + node*n_total;
    const bool* mask_ptr = mask + elem*node_per_elem*n_total + node*n_total;
    scalar_t* x_ptr = x + elem*node_per_elem*n_total + node*n_total;

    // Count valid entries
    int k = 0;
    for(int i=0;i<n_total;i++) if(mask_ptr[i]) k++;

    int sel_idx[32];  // assume k <= 32
    int cnt = 0;
    for(int i=0;i<n_total;i++)
        if(mask_ptr[i])
            sel_idx[cnt++] = i;

    double A_sub[32*32], b_sub[32], x_sub[32];

    // Copy masked submatrix
    for(int i=0;i<k;i++)
        for(int j=0;j<k;j++)
            A_sub[i*k+j] = static_cast<double>(A_ptr[sel_idx[i]*n_total + sel_idx[j]]);

    // Copy masked vector
    for(int i=0;i<k;i++)
        b_sub[i] = static_cast<double>(b_ptr[sel_idx[i]]);

    // Solve small linear system
    solve_small_double(A_sub, b_sub, x_sub, k);  // Implement LDLT/LU solve

    // Scatter back
    for(int i=0;i<k;i++)
        x_ptr[sel_idx[i]] = static_cast<scalar_t>(x_sub[i]);
}

// ----------------------
// Backward kernel
// ----------------------
template <typename scalar_t>
__global__ void solve_batched_backward_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ grad_output,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ grad_A,
    scalar_t* __restrict__ grad_b,
    const int M,
    const int node_per_elem,
    const int n_total)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_nodes = M * node_per_elem;
    if (tid >= total_nodes) return;

    int elem = tid / node_per_elem;
    int node = tid % node_per_elem;

    const scalar_t* A_ptr = A + elem*node_per_elem*n_total*n_total + node*n_total*n_total;
    const scalar_t* x_ptr = x + elem*node_per_elem*n_total + node*n_total;
    const scalar_t* grad_out_ptr = grad_output + elem*node_per_elem*n_total + node*n_total;
    const bool* mask_ptr = mask + elem*node_per_elem*n_total + node*n_total;

    scalar_t* grad_A_ptr = grad_A + elem*node_per_elem*n_total*n_total + node*n_total*n_total;
    scalar_t* grad_b_ptr = grad_b + elem*node_per_elem*n_total + node*n_total;

    // Count valid entries
    int k = 0;
    for(int i=0;i<n_total;i++) if(mask_ptr[i]) k++;

    int sel_idx[32];
    int cnt = 0;
    for(int i=0;i<n_total;i++)
        if(mask_ptr[i])
            sel_idx[cnt++] = i;

    double A_sub[32*32], x_sub[32], grad_out_sub[32], grad_A_sub[32*32], grad_b_sub[32];

    // Copy masked submatrix and x
    for(int i=0;i<k;i++){
        x_sub[i] = static_cast<double>(x_ptr[sel_idx[i]]);
        grad_out_sub[i] = static_cast<double>(grad_out_ptr[sel_idx[i]]);
        for(int j=0;j<k;j++)
            A_sub[i*k+j] = static_cast<double>(A_ptr[sel_idx[i]*n_total + sel_idx[j]]);
    }

    // Compute gradients: standard formula
    // grad_b_sub = A^{-T} * grad_out_sub
    solve_small_double(A_sub, grad_out_sub, grad_b_sub, k); // Since A is symmetric, we can reuse LDLT decomposition

    // grad_A_sub = - grad_b_sub * x_sub^T
    for(int i=0;i<k;i++)
        for(int j=0;j<k;j++)
            grad_A_sub[i*k+j] = -grad_b_sub[i]*x_sub[j];

    // Scatter back
    for(int i=0;i<k;i++){
        grad_b_ptr[sel_idx[i]] = static_cast<scalar_t>(grad_b_sub[i]);
        for(int j=0;j<k;j++)
            grad_A_ptr[sel_idx[i]*n_total + sel_idx[j]] = static_cast<scalar_t>(grad_A_sub[i*k+j]);
    }
}

// ----------------------
// Launcher functions
// ----------------------
void solve_batched_launcher(torch::Tensor A, torch::Tensor b, torch::Tensor mask, torch::Tensor x){
    const int M = A.size(0);
    const int node_per_elem = A.size(1);
    const int n_total = A.size(2);

    int total_nodes = M * node_per_elem;
    int threads = 128;
    int blocks = (total_nodes + threads - 1)/threads;

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "solve_batched_launcher", ([&] {
        solve_batched_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            mask.data_ptr<bool>(),
            x.data_ptr<scalar_t>(),
            M, node_per_elem, n_total
        );
    }));
}

void solve_batched_backward_launcher(torch::Tensor A, torch::Tensor x,
                                     torch::Tensor grad_output, torch::Tensor mask,
                                     torch::Tensor grad_A, torch::Tensor grad_b)
{
    const int M = A.size(0);
    const int node_per_elem = A.size(1);
    const int n_total = A.size(2);

    int total_nodes = M * node_per_elem;
    int threads = 128;
    int blocks = (total_nodes + threads - 1)/threads;

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "solve_batched_backward_launcher", ([&] {
        solve_batched_backward_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            grad_output.data_ptr<scalar_t>(),
            mask.data_ptr<bool>(),
            grad_A.data_ptr<scalar_t>(),
            grad_b.data_ptr<scalar_t>(),
            M, node_per_elem, n_total
        );
    }));
}

// ----------------------
// PyBind11 binding
// ----------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve_batched_launcher", &solve_batched_launcher, "Batched masked linear solve forward kernel");
    m.def("solve_batched_backward_launcher", &solve_batched_backward_launcher, "Batched masked linear solve backward kernel");
}
