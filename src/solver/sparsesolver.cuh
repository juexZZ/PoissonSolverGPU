// Poisson solver cuda kernel implementation
#include <cuda.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "../cudautil.cuh"

/* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)


__global__ void PoissonSolverSparse_init(double* b0, double* A, int* ia, int* ja, double* xk, double* rk, double* pk, double eps,
	unsigned int N, double* r_k_norm, double* r_k1_norm, double* pAp_k);
__global__ void PoissonSolverSparse_iter1(double* b0, double* A, int* ia, int* ja, double* xk, double* rk, double* pk,
	double eps, unsigned int N, double* Ap_rd, double* r_k_norm, double* r_k1_norm, double* pAp_k);
__global__ void PoissonSolverSparse_iter2(double* b0, double* A, int* ia, int* ja, double* xk, double* rk, double* pk,
	double eps, unsigned int N, double* Ap_rd, double* r_k_norm, double* r_k1_norm, double* pAp_k);
__global__ void PoissonSolverSparse_iter3(double* b0, double* A, int* ia, int* ja, double* xk, double* rk, double* pk,
	double eps, unsigned int N, double* Ap_rd, double* r_k_norm, double* r_k1_norm, double* pAp_k);

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


__global__
void PoissonSolverSparse_init(double* b0, double* A, int* ia, int* ja, double* xk, double* rk, double* pk, double eps,
	unsigned int N, double* r_k_norm, double* r_k1_norm, double* pAp_k) {
	// 1 dimesion geometry
	int ri = blockDim.x * blockIdx.x + threadIdx.x;
	//ia[ri+1]-ia[ri] is the number of i-th row entries
	if (ri < N)
	{
		// initial, compute r0 and p0
		double Ax0_r = 0;
		//double x0_r = xk[ri];
		for (unsigned int j = 0; j < ia[ri + 1] - ia[ri]; j++) {
			Ax0_r += A[j + ia[ri]] * xk[ja[j + ia[ri]]]; // x0_r;
		}
		double b_r = b0[ri];
		rk[ri] = b_r - Ax0_r;
		pk[ri] = b_r - Ax0_r;
		atomicAdd(r_k_norm, rk[ri] * rk[ri]); //&r_k_norm takes addr
		selfatomicCAS(r_k1_norm, *r_k_norm);
	}
}

__global__
void PoissonSolverSparse_iter1(double* b0, double* A, int* ia, int* ja, double* xk, double* rk, double* pk,
	double eps, unsigned int N,double* Ap_rd, double* r_k_norm, double* r_k1_norm, double* pAp_k) {
	int ri = blockDim.x * blockIdx.x + threadIdx.x;
	if (ri < N)
	{
		selfatomicCAS(r_k_norm, *r_k1_norm);
		selfatomicCAS(pAp_k, 0.0);		
		// compute \alpha_k
		double Ap_r = 0;
		double pk_r = pk[ri];
		for (unsigned int j = 0; j < ia[ri + 1] - ia[ri]; j++) {
			Ap_r += A[j + ia[ri]] * pk[ja[j + ia[ri]]]; //pk_r;
		}
		Ap_rd[ri] = Ap_r;
		// atomicAdd(r_k_norm, rk[ri]*rk[ri]);
		atomicAdd(pAp_k, pk_r * Ap_r);		
	}
}
__global__
void PoissonSolverSparse_iter2(double* b0, double* A, int* ia, int* ja, double* xk, double* rk, double* pk,
	double eps, unsigned int N, double* Ap_rd, double* r_k_norm, double* r_k1_norm, double* pAp_k) {
	int ri = blockDim.x * blockIdx.x + threadIdx.x;
	if (ri < N)
	{
		selfatomicCAS(r_k1_norm, 0.0);
		double alpha_k = *r_k_norm / *pAp_k; //solve contension by getting one for each thread
		// update x_k to x_{k+1}, r_k to r_{k+1}
		xk[ri] = xk[ri] + alpha_k * pk[ri];
		rk[ri] = rk[ri] - alpha_k * Ap_rd[ri];
		// compute beta k
		atomicAdd(r_k1_norm, rk[ri] * rk[ri]);
	}
}
__global__
void PoissonSolverSparse_iter3(double* b0, double* A, int* ia, int* ja, double* xk, double* rk, double* pk,
	double eps, unsigned int N, double* Ap_rd, double* r_k_norm, double* r_k1_norm, double* pAp_k) {
	int ri = blockDim.x * blockIdx.x + threadIdx.x;
	if (ri < N) {
		// terminating condition:
		double temp_r_k1_norm = *r_k1_norm;
		double beta_k = temp_r_k1_norm / *r_k_norm; //solve contension by getting one for each thread
		// update pk to pk1
		pk[ri] = rk[ri] + beta_k * pk[ri];
	}
}


void wrapper_PoissonSolverSparse_multiblock(unsigned int blocksPerGrid,
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
	int maxIter,double* Ap_rd,double* r_k_norm,double* r_k1_norm,double* pAp_k) {
	double temp = 0;
	printf("N = %d, kernel has %d blocks each has %d threads\n", N, blocksPerGrid, threadsPerBlock);
	PoissonSolverSparse_init<<<blocksPerGrid, threadsPerBlock >>> (rhs_d, A_d, ia_d, ja_d, x_d, rk, pk, abstol, N, r_k_norm, r_k1_norm, pAp_k);
	for (size_t i = 0; i < maxIter; i++)
	{
		PoissonSolverSparse_iter1 <<<blocksPerGrid, threadsPerBlock >> > (rhs_d, A_d, ia_d, ja_d, x_d, rk, pk, abstol, N,Ap_rd, r_k_norm, r_k1_norm, pAp_k);
		cudaDeviceSynchronize();
		PoissonSolverSparse_iter2 <<<blocksPerGrid, threadsPerBlock >> > (rhs_d, A_d, ia_d, ja_d, x_d, rk, pk, abstol, N,Ap_rd, r_k_norm, r_k1_norm, pAp_k);
		cudaDeviceSynchronize();
		PoissonSolverSparse_iter3 <<<blocksPerGrid, threadsPerBlock >> > (rhs_d, A_d, ia_d, ja_d, x_d, rk, pk, abstol, N,Ap_rd, r_k_norm, r_k1_norm, pAp_k);
		cudaDeviceSynchronize();
		cudaMemcpy(&temp, r_k1_norm, sizeof(double), cudaMemcpyDeviceToHost);
		if (temp<abstol)
		{
			printf("Early Stop");
			break;
		}
	}
}