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

template <typename T>
__global__ void PoissonSolverSparse_init(T* b0, T* A, int* ia, int* ja, T* xk, T* rk, T* pk, T eps,
	unsigned int N, T* r_k_norm, T* r_k1_norm, T* pAp_k);

template <typename T>
__global__ void PoissonSolverSparse_iter1(T* b0, T* A, int* ia, int* ja, T* xk, T* rk, T* pk,
	T eps, unsigned int N, T* Ap_rd, T* r_k_norm, T* r_k1_norm, T* pAp_k);

template <typename T>
__global__ void PoissonSolverSparse_iter2(T* b0, T* A, int* ia, int* ja, T* xk, T* rk, T* pk,
	T eps, unsigned int N, T* Ap_rd, T* r_k_norm, T* r_k1_norm, T* pAp_k);
	
template <typename T>
__global__ void PoissonSolverSparse_iter3(T* b0, T* A, int* ia, int* ja, T* xk, T* rk, T* pk,
	T eps, unsigned int N, T* Ap_rd, T* r_k_norm, T* r_k1_norm, T* pAp_k);

	

template <typename T>
__global__ void PoissonSolverSparse_init(T* b0, T* A, int* ia, int* ja, T* xk, T* rk, T* pk, T eps,
	unsigned int N, T* r_k_norm, T* r_k1_norm, T* pAp_k) {
	// 1 dimesion geometry
	int ri = blockDim.x * blockIdx.x + threadIdx.x;
	//ia[ri+1]-ia[ri] is the number of i-th row entries
	if (ri < N)
	{
		// initial, compute r0 and p0
		T Ax0_r = 0.0;
		//T x0_r = xk[ri];
		for (unsigned int j = 0; j < ia[ri + 1] - ia[ri]; j++) {
			Ax0_r += A[j + ia[ri]] * xk[ja[j + ia[ri]]]; // x0_r;
		}
		T b_r = b0[ri];
		rk[ri] = b_r - Ax0_r;
		pk[ri] = b_r - Ax0_r;
		atomicAdd(r_k_norm, rk[ri] * rk[ri]); //&r_k_norm takes addr
		selfatomicCAS(r_k1_norm, *r_k_norm);
	}
}

template <typename T>
__global__ void PoissonSolverSparse_iter1(T* b0, T* A, int* ia, int* ja, T* xk, T* rk, T* pk,
	T eps, unsigned int N,T* Ap_rd, T* r_k_norm, T* r_k1_norm, T* pAp_k) {
	int ri = blockDim.x * blockIdx.x + threadIdx.x;
	if (ri < N)
	{
		selfatomicCAS(r_k_norm, *r_k1_norm);
		selfatomicCAS(pAp_k, 0.0);		
		// compute \alpha_k
		T Ap_r = 0;
		T pk_r = pk[ri];
		for (unsigned int j = 0; j < ia[ri + 1] - ia[ri]; j++) {
			Ap_r += A[j + ia[ri]] * pk[ja[j + ia[ri]]]; //pk_r;
		}
		Ap_rd[ri] = Ap_r;
		// atomicAdd(r_k_norm, rk[ri]*rk[ri]);
		atomicAdd(pAp_k, pk_r * Ap_r);		
	}
}

template <typename T>
__global__ void PoissonSolverSparse_iter2(T* b0, T* A, int* ia, int* ja, T* xk, T* rk, T* pk,
	T eps, unsigned int N, T* Ap_rd, T* r_k_norm, T* r_k1_norm, T* pAp_k) {
	int ri = blockDim.x * blockIdx.x + threadIdx.x;
	if (ri < N)
	{
		selfatomicCAS(r_k1_norm, 0.0);
		T alpha_k = ( * r_k_norm) / ( * pAp_k); //solve contension by getting one for each thread
		// update x_k to x_{k+1}, r_k to r_{k+1}
		xk[ri] = xk[ri] + alpha_k * pk[ri];
		rk[ri] = rk[ri] - alpha_k * Ap_rd[ri];
		// compute beta k
		atomicAdd(r_k1_norm, rk[ri] * rk[ri]);
	}
}

template <typename T>
__global__ void PoissonSolverSparse_iter3(T* b0, T* A, int* ia, int* ja, T* xk, T* rk, T* pk,
	T eps, unsigned int N, T* Ap_rd, T* r_k_norm, T* r_k1_norm, T* pAp_k) {
	int ri = blockDim.x * blockIdx.x + threadIdx.x;
	if (ri < N) {
		// terminating condition:
		T beta_k = *r_k1_norm / *r_k_norm; //solve contension by getting one for each thread
		// update pk to pk1
		pk[ri] = rk[ri] + beta_k * pk[ri];
	}
}

template <typename T>
void wrapper_PoissonSolverSparse_multiblock(unsigned int blocksPerGrid,
	unsigned int threadsPerBlock,
	Eigen::Matrix<T, Eigen::Dynamic, 1> &rhs,
	Eigen::SparseMatrix<T> &A,
	Eigen::Matrix<T, Eigen::Dynamic, 1> &x,
	Eigen::Matrix<T, Eigen::Dynamic, 1> &root,
	unsigned int N,
	int maxIter,
	T abstol,
	T &iter_residual,
	int& iters){

	unsigned int vector_bytesize = N * sizeof(T);
	T* rhs_d; // b(rhs) on device
	// Triplet for A
	T* A_d;
	int* ia_d;
	int* ja_d;
	T* r_k_norm;
	T* r_k1_norm;
	T* pAp_k;
	T* x_d;
	T* rk;
	T* pk;
	T* Ap_rd;

	T temp = 0;
	T initial = 0.0;
	T* Ap_r=(T*)calloc(N,sizeof(T));
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void**)&A_d, A.nonZeros() * sizeof(T));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}
	cudaStatus = cudaMalloc((void**)&ia_d, (A.rows() + 1) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}
	cudaStatus = cudaMalloc((void**)&ja_d, A.nonZeros() * sizeof(int));
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
	cudaStatus = cudaMalloc((void**)&Ap_rd, vector_bytesize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}

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

	cudaStatus = cudaMemcpy(A_d, A.valuePtr(), A.nonZeros() * sizeof(T), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
	}

	cudaStatus = cudaMemcpy(ia_d, A.outerIndexPtr(), (A.rows() + 1) * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
	}

	cudaStatus = cudaMemcpy(ja_d, A.innerIndexPtr(), A.nonZeros() * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
	}


	cudaStatus = cudaMemcpy(rhs_d, rhs.data(), vector_bytesize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
	}
	cudaStatus = cudaMemcpy(x_d, x.data(), vector_bytesize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
	}
	cudaStatus = cudaMemcpy(Ap_rd, Ap_r, vector_bytesize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
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

	// create intermediate variables
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


	printf("N = %d, kernel has %d blocks each has %d threads\n", N, blocksPerGrid, threadsPerBlock);
	PoissonSolverSparse_init<T><<<blocksPerGrid, threadsPerBlock >>> (rhs_d, A_d, ia_d, ja_d, x_d, rk, pk, abstol, N, r_k_norm, r_k1_norm, pAp_k);
	cudaDeviceSynchronize();
	for (size_t i = 0; i < maxIter; i++)
	{
		iters = i;
		PoissonSolverSparse_iter1<T> <<<blocksPerGrid, threadsPerBlock >> > (rhs_d, A_d, ia_d, ja_d, x_d, rk, pk, abstol, N,Ap_rd, r_k_norm, r_k1_norm, pAp_k);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSychronize failed!");
			//goto Error;
		}
		PoissonSolverSparse_iter2<T> <<<blocksPerGrid, threadsPerBlock >> > (rhs_d, A_d, ia_d, ja_d, x_d, rk, pk, abstol, N,Ap_rd, r_k_norm, r_k1_norm, pAp_k);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSychronize failed!");
			//goto Error;
		}
		PoissonSolverSparse_iter3<T> <<<blocksPerGrid, threadsPerBlock >> > (rhs_d, A_d, ia_d, ja_d, x_d, rk, pk, abstol, N,Ap_rd, r_k_norm, r_k1_norm, pAp_k);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSychronize failed!");
			//goto Error;
		}
		cudaMemcpy(&temp, r_k1_norm, sizeof(T), cudaMemcpyDeviceToHost);
		iter_residual = temp;
		if (temp<abstol)
		{
			cout << "Early Stop" << endl;
			break;
		}
	}
	cudaMemcpy(root.data(), x_d, vector_bytesize, cudaMemcpyDeviceToHost);
	cudaFree(A_d);
	cudaFree(ia_d);
	cudaFree(ja_d);
	cudaFree(rhs_d);
	cudaFree(x_d);
	cudaFree(rk);
	cudaFree(pk);
}