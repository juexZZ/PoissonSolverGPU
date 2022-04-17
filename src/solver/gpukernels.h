// Poisson solver cuda kernel implementation
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

/* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)

__global__ void PoissonSolverDense(double* b0, double* A, double* x0, unsigned int N, unsigned int iters);
/* CUDA Kernel for Dense Poisson problem
Hyper Param:
float eps: minimum residue
unsigned int iters: maximum iters
Params:
float* A: matrix
float* b0: initial b
float* xk: initial x, and subsequent x

intermediate variables:
float* rk: residue
float* pk
float r_k_norm: r_k^T r_k initialize to 0. pass pointer for atomic ops
float r_k1_norm: r_{k+1}^T r_{k+1}, intialize to 0. pass pointer for atomic ops
float pAp_k: pk^T A pk: initialize to 0
alpha_k should be globally shared, or in each thread, individually
beta_k shuold be globally shared, or in each thread, individually
*/
__global__ 
void PoissonSolverDense(float* b0, float* A, float* xk, float* r_k_norm, float* pAp_k, float* r_k1_norm,
                        float eps,
                        unsigned int N, unsigned int iters){
    // 1 dimesion geometry
    int ri = blockDim.x * blockIdx.x + threadIdx.x;
    // initial, compute r0 and p0
    float Ax0_r = 0;
    float x0_r = xk[ri];
    for(int j=0; j<N; j++){
        float Aik = A[index(ri, j, N)];
        Ax0_r += Aik * x0_r;
    }
    float b_r = b0[ri];
    rk[ri] = b_r - Ax0_r;
    pk[ri] = b_r - Ax0_r;
    atomicAdd(r_k_norm, rk[ri]*rk[ri]);
    __syncthreads();
    // iterate here
    for(unsigned itk = 0; itk<iters; itk++){
        // compute \alpha_k
        float Ap_r = 0;
        float pk_r = pk[ri];
        for(int j=0; j<N; j++){
            float Aik = A[index(ri, j, N)];
            Ap_r += Aik * pk_r;
        }
        // atomicAdd(r_k_norm, rk[ri]*rk[ri]);
        atomicAdd(pAp_k, pk_r*Ap_r);
        __syncthreads();
        float alpha_k = r_k_norm / pAp_k; //solve contension by getting one for each thread
        // update x_k to x_{k+1}, r_k to r_{k+1}
        xk[ri] = xk[ri] + alpha_k * pk_r;
        rk[ri] = rk[ri] - alpha_k * Ap_r;
        // compute beta k
        atomicAdd(r_k1_norm, rk[ri]*rk[ri]);
        __syncthreads();
        // terminating condition:
        if(r_k1_norm < eps){
            break;
        }
        float beta_k = r_k1_norm / r_k_norm; //solve contension by getting one for each thread
        // update pk to pk1
        pk[ri] = rk[ri] + beta_k * pk[ri];
        // update rk norm to r_k1_norm, set r_k1_norm to 0 before next iter.
        r_k_norm = r_k1_norm;
        r_k1_norm = 0.0;
    }
}