// Poisson solver cuda kernel implementation
#include <cuda.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "gpukernels.h"

typedef long long int int64_cu;
typedef unsigned long long int uint64_cu;

/* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)

__device__  void selfatomicCAS(double* address, double val) {
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val));

		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);
}

__global__ void PoissonSolverSparse(double* b0, double* A, double* xk, int* ia, int* ja, double* rk, double* pk,
	double eps, unsigned int N, unsigned int iters);

/* CUDA Kernel for Dense Poisson problem
Hyper Param:
double eps: minimum residue
unsigned int iters: maximum iters
Params:
double* A: matrix
double* b0: initial b
double* xk: initial x, and subsequent x

intermediate variables:
double* rk: residue
double* pk
double r_k_norm: r_k^T r_k initialize to 0. pass pointer for atomic ops
double r_k1_norm: r_{k+1}^T r_{k+1}, intialize to 0. pass pointer for atomic ops
double pAp_k: pk^T A pk: initialize to 0
alpha_k should be globally shared, or in each thread, individually
beta_k shuold be globally shared, or in each thread, individually
*/
__device__ double r_k_norm = 0.0;
__device__ double r_k1_norm = 0.0;
__device__ double pAp_k = 0.0;
__device__ bool norm_updated = false;

__global__
void PoissonSolverSparse(double* b0, double* A,int* ia, int* ja,double* xk, double* rk, double* pk,
	double eps, unsigned int N, unsigned int iters) {
	// 1 dimesion geometry
	int ri = blockDim.x * blockIdx.x + threadIdx.x;
	//ia[ri+1]-ia[ri] is the number of i-th row entries
	if (ri < N)
	{
		double* row;
		row = (double*)malloc(ia[ri + 1] - ia[ri]);
		for (size_t i = 0; i < ia[ri + 1]-ia[ri]; i++)
		{
			row[i] = A[ia[ri]+i];
		}
		// initial, compute r0 and p0
		double Ax0_r = 0;
		//double x0_r = xk[ri];
		for (unsigned int j = 0; j < ia[ri + 1] - ia[ri]; j++) {
			Ax0_r += row[j] * xk[j]; // x0_r;
		}
		double b_r = b0[ri];
		rk[ri] = b_r - Ax0_r;
		pk[ri] = b_r - Ax0_r;
		atomicAdd(&r_k_norm, rk[ri] * rk[ri]); //&r_k_norm takes addr
		__syncthreads();
		// iterate here
		for (unsigned int itk = 0; itk < iters; itk++) {
			// compute \alpha_k
			double Ap_r = 0;
			double pk_r = pk[ri];
			for (unsigned int j = 0; j < ia[ri + 1] - ia[ri]; j++) {
				Ap_r += row[j] * pk[j]; //pk_r;
			}
			// atomicAdd(r_k_norm, rk[ri]*rk[ri]);
			atomicAdd(&pAp_k, pk_r * Ap_r);
			selfatomicCAS(&r_k1_norm,0.0);
			__syncthreads();
			double alpha_k = r_k_norm / pAp_k; //solve contension by getting one for each thread
			// update x_k to x_{k+1}, r_k to r_{k+1}
			xk[ri] = xk[ri] + alpha_k * pk_r;
			rk[ri] = rk[ri] - alpha_k * Ap_r;
			// compute beta k
			atomicAdd(&r_k1_norm, rk[ri] * rk[ri]);
			__syncthreads();
			// terminating condition:
			if (r_k1_norm < eps) {
				break;
			}
			double temp_r_k1_norm = r_k1_norm;
			double beta_k = temp_r_k1_norm / r_k_norm; //solve contension by getting one for each thread
			// update pk to pk1
			pk[ri] = rk[ri] + beta_k * pk[ri];
			selfatomicCAS(&pAp_k, 0.0);
			selfatomicCAS(&r_k_norm, temp_r_k1_norm);
		}
	}

}

// wrapper function 
void wrapper_PoissonSolverSparse(unsigned int blocksPerGrid,
	unsigned int threadsPerBlock,
	double* rhs_d,
	double* A_d,
	int* ia_d,
	int* ja_d,
	double* x_d,
	double* rk,
	double* pk,
	double abstol,
	unsigned int N,
	int maxIter) {
	// call cuda kernel functions here
	printf("N = %d, kernel has %d blocks each has %d threads\n", N, blocksPerGrid, threadsPerBlock);
	PoissonSolverSparse << <blocksPerGrid, threadsPerBlock >> > (rhs_d, A_d, ia_d, ja_d, x_d, rk, pk, abstol, N, maxIter);
}