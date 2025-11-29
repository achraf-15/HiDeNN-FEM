// algo.cuh
#pragma once
#include <cuda_runtime.h>
#include <cmath>


// Cholesky inversion for symmetric positive definite matrix
// A: input k x k matrix
// Ainv: output k x k inverse matrix
__device__ void invert_cholesky_double(const double* A, double* Ainv, int k) {
    double L[32*32];    // lower-triangular Cholesky factor
    double Y[32*32];    // intermediate solution
    double I[32*32];    // identity matrix

    // --- Step 0: initialize identity matrix ---
    for(int i=0;i<k;i++)
        for(int j=0;j<k;j++)
            I[i*k + j] = (i==j ? 1.0f : 0.0f);

    // --- Step 1: compute Cholesky factor L (lower-triangular) ---
    for(int i=0;i<k;i++) {
        for(int j=0;j<=i;j++) {
            double sum = A[i*k + j];
            for(int t=0;t<j;t++)
                sum -= L[i*k + t] * L[j*k + t];
            if(i == j) {
                // diagonal element
                L[i*k + j] = sqrtf(sum);
            } else {
                L[i*k + j] = sum / L[j*k + j];
            }
        }
        // fill upper triangle with 0 (not used)
        for(int j=i+1;j<k;j++)
            L[i*k + j] = 0.0f;
    }

    // --- Step 2: solve L * Y = I (forward substitution) ---
    for(int col=0; col<k; col++) {
        for(int i=0; i<k; i++) {
            double sum = I[i*k + col];
            for(int t=0; t<i; t++)
                sum -= L[i*k + t] * Y[t*k + col];
            Y[i*k + col] = sum / L[i*k + i];
        }
    }

    // --- Step 3: solve L^T * Ainv = Y (backward substitution) ---
    for(int col=0; col<k; col++) {
        for(int i=k-1; i>=0; i--) {
            double sum = Y[i*k + col];
            for(int t=i+1; t<k; t++)
                sum -= L[t*k + i] * Ainv[t*k + col];
            Ainv[i*k + col] = sum / L[i*k + i];
        }
    }
}


// Symmetric indefinite inversion using LDL^T (no pivoting)
// A: input symmetric matrix (k x k) in row-major
// Ainv: output inverse (k x k)
__device__ void invert_ldlt_double(const double* A, double* Ainv, int k)
{
    // Storage for L (unit lower triangle) and D (diagonal)
    double L[32*32];
    double D[32];

    // Copy A into L; we'll overwrite into LDL^T
    for(int i=0;i<k;i++)
        for(int j=0;j<k;j++)
            L[i*k + j] = A[i*k + j];

    // --- LDL^T Factorization ---
    for(int j=0;j<k;j++){
        // Compute D[j]
        double sum = 0.0f;
        for(int s=0;s<j;s++){
            double Ljs = L[j*k + s];
            sum += Ljs * Ljs * D[s];
        }
        D[j] = L[j*k + j] - sum;

        // Compute column j of L
        for(int i=j+1;i<k;i++){
            double sum2 = 0.0f;
            for(int s=0;s<j;s++){
                sum2 += L[i*k + s] * L[j*k + s] * D[s];
            }
            L[i*k + j] = (L[i*k + j] - sum2) / D[j];
        }

        // Fill diagonal with ones
        L[j*k + j] = 1.0f;
        // Zero upper triangle
        for(int t=j+1;t<k;t++)
            L[j*k + t] = 0.0f;
    }

    // --- Inverse via solving L D L^T x = e_r ---
    double y[32], z[32], x[32];

    for(int r=0; r<k; r++){
        // Solve L y = e_r
        for(int i=0;i<k;i++){
            double rhs = (i == r ? 1.0f : 0.0f);
            double acc = 0.0f;
            for(int j=0;j<i;j++){
                acc += L[i*k + j] * y[j];
            }
            y[i] = rhs - acc;
        }

        // Solve D z = y  (D diagonal)
        for(int i=0;i<k;i++){
            z[i] = y[i] / D[i];
        }

        // Solve L^T x = z
        for(int i=k-1; i>=0; i--){
            double acc = 0.0f;
            for(int j=i+1;j<k;j++){
                acc += L[j*k + i] * x[j];
            }
            x[i] = z[i] - acc;
        }

        // Store column r into Ainv
        for(int i=0;i<k;i++)
            Ainv[i*k + r] = x[i];
    }
}


__device__ void invert_small_double(double* A, double* Ainv, int n){
    // Initialize Ainv = identity
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            Ainv[i*n+j] = (i==j) ? 1.f : 0.f;

    for(int i=0;i<n;i++){
        // Find pivot
        int max_row = i;
        double max_val = fabsf(A[i*n + i]);
        for(int r=i+1;r<n;r++){
            double val = fabsf(A[r*n + i]);
            if(val > max_val){
                max_val = val;
                max_row = r;
            }
        }

        // Swap rows in both A and Ainv
        if(max_row != i){
            for(int j=0;j<n;j++){
                double tmp = A[i*n+j];    A[i*n+j] = A[max_row*n+j];    A[max_row*n+j] = tmp;
                tmp = Ainv[i*n+j];       Ainv[i*n+j] = Ainv[max_row*n+j]; Ainv[max_row*n+j] = tmp;
            }
        }

        // Normalize pivot row
        double pivot = A[i*n + i];
        for(int j=0;j<n;j++){
            A[i*n+j] /= pivot;
            Ainv[i*n+j] /= pivot;
        }

        // Eliminate other rows
        for(int ii=0;ii<n;ii++){
            if(ii==i) continue;
            double factor = A[ii*n + i];
            for(int j=0;j<n;j++){
                A[ii*n+j] -= factor*A[i*n+j];
                Ainv[ii*n+j] -= factor*Ainv[i*n+j];
            }
        }
    }
}


__device__ void matmul_transpose_grad_double(double* Ginv, double* dOut, double* dG, int k){
    for(int i=0;i<k;i++)
        for(int j=0;j<k;j++){
            double sum = 0.0;
            for(int p=0;p<k;p++)
                for(int q=0;q<k;q++)
                    sum -= (double)Ginv[p*k + i] * (double)dOut[p*k + q] * (double)Ginv[q*k + j];
            dG[i*k+j] = (float)sum;
        }
}