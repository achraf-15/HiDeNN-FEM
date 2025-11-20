#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cmath>

#include "algo.cuh"


// Forward kernel
__global__ void invert_batched_kernel(
    const float* __restrict__ G,
    const int* __restrict__ valid_idx,
    float* __restrict__ G_inv,
    const int Nelems,
    const int node_per_elem,
    const int n_tot
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_nodes = Nelems * node_per_elem;
    if(tid >= total_nodes) return;

    int elem = tid / node_per_elem;
    int node = tid % node_per_elem;

    const float* G_ptr = G + elem*node_per_elem*n_tot*n_tot + node*n_tot*n_tot;
    const int* idx_ptr = valid_idx + elem*node_per_elem*n_tot + node*n_tot;
    float* Ginv_ptr = G_inv + elem*node_per_elem*n_tot*n_tot + node*n_tot*n_tot;

    // Count valid entries
    int k = 0;
    for(int i=0;i<n_tot;i++) if(idx_ptr[i] == i) k++;

    float G_sub[32*32], G_sub_inv[32*32];  // adjust max size if needed
    int sel_idx[32];

    int cnt = 0;
    for(int i=0;i<n_tot;i++){
        if(idx_ptr[i] == i){
            sel_idx[cnt++] = idx_ptr[i];
        }
    }

    // Copy valid submatrix
    for(int i=0;i<k;i++)
        for(int j=0;j<k;j++){
            G_sub[i*k+j] = G_ptr[sel_idx[i]*n_tot + sel_idx[j]];
            //printf("G_sub: Element=%d at Node=%d index=(%d,%d), value=%f, count=%d \n",
            //       elem, node, i, j, G_sub[i*k+j], k);
            }

    // invert_small(G_sub, G_sub_inv, k);

    // double G_sub64[32*32], G_sub_inv64[32*32];
    // for(int i=0;i<k*k;i++)
    //     G_sub64[i] = (double) G_sub[i];
    // invert_small_double(G_sub64, G_sub_inv64, k);
    // for(int i=0;i<k*k;i++)
    //     G_sub_inv[i] = (float) G_sub_inv64[i];


    // invert_ldlt(G_sub, G_sub_inv, k);

    double G_sub64[32*32], G_sub_inv64[32*32];
    for(int i=0;i<k*k;i++)
        G_sub64[i] = (double) G_sub[i];
    invert_ldlt_double(G_sub64, G_sub_inv64, k);
    for(int i=0;i<k*k;i++)
        G_sub_inv[i] = (float) G_sub_inv64[i];


    // invert_cholesky(G_sub, G_sub_inv, k); // doesnt work since G_sub is not definite


    // Scatter back
    for(int i=0;i<k;i++)
        for(int j=0;j<k;j++){
            Ginv_ptr[sel_idx[i]*n_tot + sel_idx[j]] = G_sub_inv[i*k+j];
            //printf("G_inv: Element=%d at Node=%d index=(%d,%d), value=%f, count=%d \n",
            //       elem, node, i, j, G_sub[i*k+j], k);
            }
}

// Backward helper: dG = -Ginv^T * dOut * Ginv^T
__device__ void matmul_transpose_grad(float* Ginv, float* dOut, float* dG, int k){
    for(int i=0;i<k;i++)
        for(int j=0;j<k;j++){
            float sum = 0.f;
            for(int p=0;p<k;p++)
                for(int q=0;q<k;q++)
                    sum -= Ginv[p*k + i] * dOut[p*k + q] * Ginv[q*k + j];
            dG[i*k+j] = sum;
        }
}

// Backward kernel
__global__ void invert_batched_backward_kernel(
    const float* __restrict__ G_inv,
    const int* __restrict__ valid_idx,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_G,
    const int Nelems,
    const int node_per_elem,
    const int n_tot
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_nodes = Nelems * node_per_elem;
    if(tid >= total_nodes) return;

    int elem = tid / node_per_elem;
    int node = tid % node_per_elem;

    const float* Ginv_ptr = G_inv + elem*node_per_elem*n_tot*n_tot + node*n_tot*n_tot;
    const int* idx_ptr = valid_idx + elem*node_per_elem*n_tot + node*n_tot;
    const float* grad_out_ptr = grad_output + elem*node_per_elem*n_tot*n_tot + node*n_tot*n_tot;
    float* grad_G_ptr = grad_G + elem*node_per_elem*n_tot*n_tot + node*n_tot*n_tot;

    // Count valid entries
    int k = 0;
    for(int i=0;i<n_tot;i++) if(idx_ptr[i] == i) k++;

    float Ginv_sub[32*32], grad_out_sub[32*32], grad_sub[32*32];
    int sel_idx[32];
    int cnt = 0;
    for(int i=0;i<n_tot;i++)
        if(idx_ptr[i] == i)
            sel_idx[cnt++] = idx_ptr[i];

    for(int i=0;i<k;i++)
        for(int j=0;j<k;j++){
            Ginv_sub[i*k+j] = Ginv_ptr[sel_idx[i]*n_tot + sel_idx[j]];
            grad_out_sub[i*k+j] = grad_out_ptr[sel_idx[i]*n_tot + sel_idx[j]];
        }

    matmul_transpose_grad(Ginv_sub, grad_out_sub, grad_sub, k);

    // Scatter back
    for(int i=0;i<k;i++)
        for(int j=0;j<k;j++)
            grad_G_ptr[sel_idx[i]*n_tot + sel_idx[j]] = grad_sub[i*k+j];
}

// Launcher functions
void invert_batched_launcher(torch::Tensor G, torch::Tensor valid_idx, torch::Tensor G_inv){
    const int Nelems = G.size(0);
    const int node_per_elem = G.size(1);

    int total_nodes = Nelems * node_per_elem;
    int threads = 128;
    int blocks = (total_nodes + threads - 1)/threads;

    invert_batched_kernel<<<blocks, threads>>>(G.data_ptr<float>(),
                                               valid_idx.data_ptr<int>(),
                                               G_inv.data_ptr<float>(),
                                               Nelems, node_per_elem, G.size(2));
}

void invert_batched_backward_launcher(torch::Tensor G_inv, torch::Tensor valid_idx,
                                      torch::Tensor grad_output, torch::Tensor grad_G){
    const int Nelems = G_inv.size(0);
    const int node_per_elem = G_inv.size(1);

    int total_nodes = Nelems * node_per_elem;
    int threads = 128;
    int blocks = (total_nodes + threads - 1)/threads;

    invert_batched_backward_kernel<<<blocks, threads>>>(G_inv.data_ptr<float>(),
                                                        valid_idx.data_ptr<int>(),
                                                        grad_output.data_ptr<float>(),
                                                        grad_G.data_ptr<float>(),
                                                        Nelems, node_per_elem, G_inv.size(2));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("invert_batched_launcher", &invert_batched_launcher, "Batched inversion kernel");
    m.def("invert_batched_backward_launcher", &invert_batched_backward_launcher, "Batched inversion backward kernel");
}
