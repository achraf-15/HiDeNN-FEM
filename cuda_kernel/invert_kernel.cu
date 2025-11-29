#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cmath>

#include "algo.cuh"

// Forward kernel
template <typename scalar_t>
__global__ void invert_batched_kernel(
    const scalar_t* __restrict__ G,
    const int* __restrict__ valid_idx,
    scalar_t* __restrict__ G_inv,
    const int Nelems,
    const int node_per_elem,
    const int n_tot
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_nodes = Nelems * node_per_elem;
    if(tid >= total_nodes) return;

    int elem = tid / node_per_elem;
    int node = tid % node_per_elem;

    const scalar_t* G_ptr = G + elem*node_per_elem*n_tot*n_tot + node*n_tot*n_tot;
    const int* idx_ptr = valid_idx + elem*node_per_elem*n_tot + node*n_tot;
    scalar_t* Ginv_ptr = G_inv + elem*node_per_elem*n_tot*n_tot + node*n_tot*n_tot;

    // Count valid entries
    int k = 0;
    for(int i=0;i<n_tot;i++) if(idx_ptr[i] == i) k++;

    int sel_idx[32];
    int cnt = 0;
    for(int i=0;i<n_tot;i++){
        if(idx_ptr[i] == i){
            sel_idx[cnt++] = idx_ptr[i];
        }
    }

    double G_sub[32*32], G_sub_inv[32*32];  // internal calculations are always in double

    // Copy valid submatrix
    for(int i=0;i<k;i++)
        for(int j=0;j<k;j++){
            G_sub[i*k+j] = static_cast<double>(G_ptr[sel_idx[i]*n_tot + sel_idx[j]]);
            //printf("G_sub: Element=%d at Node=%d index=(%d,%d), value=%f, count=%d \n",
            //       elem, node, i, j, G_sub[i*k+j], k);
            }

    // invert_small_double(G_sub, G_sub_inv, k);
    invert_ldlt_double(G_sub, G_sub_inv, k);
    // invert_cholesky(G_sub, G_sub_inv, k); // doesnt work since G_sub is not definite

    // Scatter back
    for(int i=0;i<k;i++)
        for(int j=0;j<k;j++){
            Ginv_ptr[sel_idx[i]*n_tot + sel_idx[j]] = static_cast<scalar_t>(G_sub_inv[i*k+j]);
            //printf("G_inv: Element=%d at Node=%d index=(%d,%d), value=%f, count=%d \n",
            //       elem, node, i, j, G_sub[i*k+j], k);
            }
}

// Backward kernel
template <typename scalar_t>
__global__ void invert_batched_backward_kernel(
    const scalar_t* __restrict__ G_inv,
    const int* __restrict__ valid_idx,
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_G,
    const int Nelems,
    const int node_per_elem,
    const int n_tot
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_nodes = Nelems * node_per_elem;
    if(tid >= total_nodes) return;

    int elem = tid / node_per_elem;
    int node = tid % node_per_elem;

    const scalar_t* Ginv_ptr = G_inv + elem*node_per_elem*n_tot*n_tot + node*n_tot*n_tot;
    const int* idx_ptr = valid_idx + elem*node_per_elem*n_tot + node*n_tot;
    const scalar_t* grad_out_ptr = grad_output + elem*node_per_elem*n_tot*n_tot + node*n_tot*n_tot;
    scalar_t* grad_G_ptr = grad_G + elem*node_per_elem*n_tot*n_tot + node*n_tot*n_tot;

    // Count valid entries
    int k = 0;
    for(int i=0;i<n_tot;i++) if(idx_ptr[i] == i) k++;

    int sel_idx[32];
    int cnt = 0;
    for(int i=0;i<n_tot;i++)
        if(idx_ptr[i] == i)
            sel_idx[cnt++] = idx_ptr[i];

    double Ginv_sub[32*32], grad_out_sub[32*32], grad_sub[32*32];

    for(int i=0;i<k;i++)
        for(int j=0;j<k;j++){
            Ginv_sub[i*k+j] = static_cast<double>(Ginv_ptr[sel_idx[i]*n_tot + sel_idx[j]]);
            grad_out_sub[i*k+j] = static_cast<double>(grad_out_ptr[sel_idx[i]*n_tot + sel_idx[j]]);
        }

    matmul_transpose_grad_double(Ginv_sub, grad_out_sub, grad_sub, k);

    // Scatter back
    for(int i=0;i<k;i++)
        for(int j=0;j<k;j++)
            grad_G_ptr[sel_idx[i]*n_tot + sel_idx[j]] = static_cast<scalar_t>(grad_sub[i*k+j]);
}

// Launcher functions
void invert_batched_launcher(torch::Tensor G, torch::Tensor valid_idx, torch::Tensor G_inv){
    const int Nelems = G.size(0);
    const int node_per_elem = G.size(1);

    int total_nodes = Nelems * node_per_elem;
    int threads = 128;
    int blocks = (total_nodes + threads - 1)/threads;

    AT_DISPATCH_FLOATING_TYPES(G.scalar_type(), "invert_batched_launcher", ([&] {
        invert_batched_kernel<scalar_t><<<blocks, threads>>>(
            G.data_ptr<scalar_t>(),
            valid_idx.data_ptr<int>(),
            G_inv.data_ptr<scalar_t>(),
            Nelems, node_per_elem, G.size(2)
        );
    }));
}

void invert_batched_backward_launcher(torch::Tensor G_inv, torch::Tensor valid_idx,
                                      torch::Tensor grad_output, torch::Tensor grad_G){
    const int Nelems = G_inv.size(0);
    const int node_per_elem = G_inv.size(1);

    int total_nodes = Nelems * node_per_elem;
    int threads = 128;
    int blocks = (total_nodes + threads - 1)/threads;

    AT_DISPATCH_FLOATING_TYPES(G_inv.scalar_type(), "invert_batched_backward_launcher", ([&] {
        invert_batched_backward_kernel<scalar_t><<<blocks, threads>>>(
            G_inv.data_ptr<scalar_t>(),
            valid_idx.data_ptr<int>(),
            grad_output.data_ptr<scalar_t>(),
            grad_G.data_ptr<scalar_t>(),
            Nelems, node_per_elem, G_inv.size(2)
        );
    }));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("invert_batched_launcher", &invert_batched_launcher, "Batched inversion kernel");
    m.def("invert_batched_backward_launcher", &invert_batched_backward_launcher, "Batched inversion backward kernel");
}
