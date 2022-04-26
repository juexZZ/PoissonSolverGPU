// Poisson solver cuda kernel implementation
#include <cuda.h>
#include <Eigen/Dense>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "../cudautil.cuh"


/* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)

template<typename T>
__global__ void PoissonSolverDense(T* b0, T* A, T* xk, T* rk, T* pk,
    T* r_k_norm, T* r_k1_norm, T* pAp_k,
    T eps, unsigned int N, unsigned int iters);

/* CUDA Kernel for Dense Poisson problem
Hyper Param:
T eps: minimum residue
unsigned int iters: maximum iters
Params:
T* A: matrix
T* b0: initial b
T* xk: initial x, and subsequent x

intermediate variables:
T* rk: residue
T* pk
T r_k_norm: r_k^T r_k initialize to 0. pass pointer for atomic ops
T r_k1_norm: r_{k+1}^T r_{k+1}, intialize to 0. pass pointer for atomic ops
T pAp_k: pk^T A pk: initialize to 0
alpha_k should be globally shared, or in each thread, individually
beta_k shuold be globally shared, or in each thread, individually
*/
// __device__ T r_k_norm = 0.0;
// __device__ T r_k1_norm = 0.0;
// __device__ T pAp_k = 0.0;
// __device__ bool norm_updated = false;

template<typename T>
__global__ void PoissonSolverDense(T* b0, T* A, T* xk, T* rk, T* pk, 
    T* r_k_norm, T* r_k1_norm, T* pAp_k,
    T eps, unsigned int N, unsigned int iters) {

    // 1 dimesion geometry
    int ri = blockDim.x * blockIdx.x + threadIdx.x;
    if (ri < N) {
        // initial, compute r0 and p0
        T Ax0_r = 0;
        //T x0_r = xk[ri];
        for (unsigned int j = 0; j < N; j++) {
            T Aik = A[index(ri, j, N)];
            Ax0_r += Aik * xk[j];
        }
        T b_r = b0[ri];
        rk[ri] = b_r - Ax0_r;
        pk[ri] = b_r - Ax0_r;
        atomicAdd(r_k_norm, rk[ri] * rk[ri]); //&r_k_norm takes addr
        __syncthreads();
        // iterate here
        for (unsigned int itk = 0; itk < iters; itk++) {
            // compute \alpha_k
            T Ap_r = 0;
            T pk_r = pk[ri];
            for (unsigned int j = 0; j < N; j++) {
                T Aik = A[index(ri, j, N)];
                Ap_r += Aik * pk[j];
            }
            atomicAdd(pAp_k, pk_r * Ap_r);
            selfatomicCAS(r_k1_norm, 0.0);//r_k1_norm = 0.0;
            __syncthreads();
            T alpha_k = *r_k_norm / *pAp_k; //solve contension by getting one for each thread
            // update x_k to x_{k+1}, r_k to r_{k+1}
            xk[ri] = xk[ri] + alpha_k * pk_r;
            rk[ri] = rk[ri] - alpha_k * Ap_r;
            // compute beta k
            atomicAdd(r_k1_norm, rk[ri] * rk[ri]);
            __syncthreads();
            // terminating condition:
            if (*r_k1_norm < eps) {
                break;
            }
            T temp_r_k1_norm = *r_k1_norm;
            T beta_k = *r_k1_norm / *r_k_norm; //solve contension by getting one for each thread
            // update pk to pk1
            pk[ri] = rk[ri] + beta_k * pk[ri];
            __syncthreads();
            selfatomicCAS(pAp_k, 0.0);
            selfatomicCAS(r_k_norm, temp_r_k1_norm); // update rk norm to r_k1_norm, set r_k1_norm to 0 before next iter.
        }
    }
}

// wrapper function
template <typename T> 
void wrapper_PoissonSolverDense(unsigned int blocksPerGrid,
    unsigned int threadsPerBlock,
    T* rhs_dptr,
    T* A_dptr,
    T* x_dptr,
    Eigen::Matrix<T, Eigen::Dynamic, 1> &root,
    T abstol,
    unsigned int N,
    int maxIter) {
    
    cudaError_t cudaStatus;
    unsigned int matrix_bytesize = N * N * sizeof(T); // NxN
	unsigned int vector_bytesize = N * sizeof(T);
    // allocate and move to device
	T* rhs_d; // b(rhs) on device
	T* A_d; // A on device
	T* x_d; //x on device
	cudaStatus = cudaMalloc((void**)&A_d, matrix_bytesize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}
	cudaStatus = cudaMalloc((void**)&rhs_d, vector_bytesize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}
	cudaStatus = cudaMalloc((void**)&x_d, vector_bytesize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

    cudaStatus = cudaMemcpy(A_d, A_dptr, matrix_bytesize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
	}
	cudaStatus = cudaMemcpy(rhs_d, rhs_dptr, vector_bytesize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
	}
	cudaStatus = cudaMemcpy(x_d, x_dptr, vector_bytesize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
	}

    // create intermediate variables
	T* rk; //residue
	T* pk;
	cudaStatus = cudaMalloc((void**)&rk, vector_bytesize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}
	cudaStatus = cudaMalloc((void**)&pk, vector_bytesize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

    T* r_k_norm;
    T* r_k1_norm;
    T* pAp_k;
    T initial = 0.0;
    
    cudaStatus = cudaMalloc((void**)&r_k_norm, sizeof(T));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}
	cudaStatus = cudaMalloc((void**)&r_k1_norm, sizeof(T));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}
	cudaStatus = cudaMalloc((void**)&pAp_k, sizeof(T));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

    cudaStatus = cudaMemcpy(r_k_norm, &initial,  sizeof(T), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
	}
	cudaStatus = cudaMemcpy(r_k1_norm, &initial, sizeof(T), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
	}
	cudaStatus = cudaMemcpy(pAp_k, &initial, sizeof(T), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
	}

    printf("N = %d, kernel has %d blocks each has %d threads\n", N, blocksPerGrid, threadsPerBlock);
    PoissonSolverDense<T> << <blocksPerGrid, threadsPerBlock >> > (rhs_d, A_d, x_d, rk, pk, r_k_norm, r_k1_norm, pAp_k, abstol, N, maxIter);
    // Eigen::Matrix<T, Eigen::Dynamic, 1> root(N);
	cudaMemcpy(root.data(), x_d, vector_bytesize, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
	cudaFree(rhs_d);
	cudaFree(x_d);
	cudaFree(rk);
	cudaFree(pk);
    cudaFree(r_k_norm);
    cudaFree(r_k1_norm);
    cudaFree(pAp_k);
}