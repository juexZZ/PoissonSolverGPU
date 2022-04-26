#include <cuda.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cusparse.h>
#include <cublas_v2.h>


#include <stdlib.h>
#include <stdio.h>
#include <time.h>



#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        exit(0);                                                  \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        exit(0);                                                   \
    }                                                                          \
}
using namespace std;

void wrapper_PoissonSolverSparse_cusparese_device_kernel(unsigned int blocksPerGrid,
	unsigned int threadsPerBlock,
	unsigned int nnz,
	double* rhs_d,
	double* A_d,
	int* ia_d,
	int* ja_d,
	double* x_d,
	double* rk,
	double* pk,
	double abstol,
	unsigned int N,
	int maxIter,
	double* Ap_rd,
	double& iter_residual,
	int& iters) {
	printf("N = %d, kernel has %d blocks each has %d threads\n", N, blocksPerGrid, threadsPerBlock);
	cusparseSpMatDescr_t A_sparse = 0;
	cusparseCreateCsr(&A_sparse,
		N,
		N,
		nnz,
		ia_d,
		ja_d,
		A_d,
		CUSPARSE_INDEX_32I,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
	cusparseHandle_t handle = 0;
	cusparseCreate(&handle);
	cublasHandle_t handle_blas = 0;
	cublasCreate(&handle_blas);
	cusparseDnVecDescr_t rhs_vec;
	cusparseDnVecDescr_t x_vec;
	cusparseDnVecDescr_t rk_vec;
	cusparseDnVecDescr_t pk_vec;
	cusparseDnVecDescr_t Ap_r_vec;
	cusparseDnVecDescr_t tempvec;
	double* tempvec_d;
	CHECK_CUDA(cudaMalloc((void**)&tempvec_d, N * sizeof(double)));
	cusparseCreateDnVec(&tempvec, N, tempvec_d, CUDA_R_64F);
	cusparseCreateDnVec(&rhs_vec, N, rhs_d, CUDA_R_64F);
	cusparseCreateDnVec(&x_vec, N, x_d, CUDA_R_64F);
	cusparseCreateDnVec(&rk_vec, N, rk, CUDA_R_64F);
	cusparseCreateDnVec(&pk_vec, N, pk, CUDA_R_64F);
	cusparseCreateDnVec(&Ap_r_vec, N, Ap_rd, CUDA_R_64F);
	size_t buffersize = 0;
	//Use to receive the multiplication result
	double temp = 0.0;
	double alpha_k = 0.0;
	double beta_k = 0.0;
	double r_k_norm = 0.0;
	double r_k1_norm = 0.0;
	double pAp_k = 0.0;
	//Initialize 
	{
		double alpha = -1.0;
		double beta = 1.0;
		cublasDcopy(handle_blas, N, rhs_d, 1, tempvec_d, 1);
		cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A_sparse, x_vec, &beta, tempvec, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buffersize);
		void* dBuffer = NULL;
		CHECK_CUDA(cudaMalloc(&dBuffer, buffersize));
		cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A_sparse, x_vec, &beta, tempvec, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, dBuffer);
		cublasDcopy(handle_blas, N, tempvec_d, 1, rk, 1);
		cublasDcopy(handle_blas, N, rk, 1, pk, 1);
		cublasDdot(handle_blas, N, rk, 1, rk, 1, &r_k_norm);
		r_k1_norm = r_k_norm;
	}
	//Iteration
	for (size_t i = 0; i < maxIter; i++)
	{
		iters = i;
		{//Ap_r_vec=A*pk
			r_k_norm = r_k1_norm;
			pAp_k = 0;
			double alpha = 1.0;
			double beta = 0.0;
			cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A_sparse, pk_vec, &beta, Ap_r_vec, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buffersize);
			void* dBuffer = NULL;
			CHECK_CUDA(cudaMalloc(&dBuffer, buffersize));
			cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A_sparse, pk_vec, &beta, Ap_r_vec, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, dBuffer);
			cublasDdot(handle_blas, N, pk, 1, Ap_rd, 1, &pAp_k);
		}
		{
			r_k1_norm = 0.0;
			alpha_k = r_k_norm / (pAp_k);
			cublasDaxpy(handle_blas, N, &alpha_k, pk, 1, x_d, 1);
			double temp = -alpha_k;
			cublasDaxpy(handle_blas, N, &temp, Ap_rd, 1, rk, 1);
			cublasDdot(handle_blas, N, rk, 1, rk, 1, &r_k1_norm);
			iter_residual = r_k1_norm;
			if (r_k1_norm < abstol)
			{
				cout << "Early Stop" << endl;
				break;
			}
		}
		{
			double alpha = 1.0;
			beta_k = r_k1_norm / r_k_norm;
			cublasDscal(handle_blas, N, &beta_k, pk, 1);
			cublasDaxpy(handle_blas, N, &alpha, rk, 1, pk, 1);
		}
	}

}



void wrapper_PoissonSolverSparse_cusparse(unsigned int blocksPerGrid,
	unsigned int threadsPerBlock,
	Eigen::Matrix<double, Eigen::Dynamic, 1>& rhs,
	Eigen::SparseMatrix<double>& A,
	Eigen::Matrix<double, Eigen::Dynamic, 1>& x,
	Eigen::Matrix<double, Eigen::Dynamic, 1>& root,
	unsigned int N,
	int maxIter,
	double abstol,
	double& iter_residual,
	int& iters) {
	unsigned int vector_bytesize = N * sizeof(double);
	cudaError_t cudaStatus;
	// convert matrix to row-major storage
	// allocate and move to device
	double* rhs_d; // b(rhs) on device
	// Triplet for A
	double* A_d; //csrValA
	int* ia_d; //csrRowPtrA
	int* ja_d;//csrColIndA 

	double* x_d; //x on device
	double* Ap_rd;
	double initial = 0.0;
	cusparseHandle_t handle;


	cudaStatus = cudaMalloc((void**)&A_d, A.nonZeros() * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}
	//csrRowPtrA
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


	/*cudaStatus = cudaMemcpy(a_d, a.begin_, A.nonZeros() * sizeof(double), cudaMemcpyHostToDevice);*/
	cudaStatus = cudaMemcpy(A_d, A.valuePtr(), A.nonZeros() * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
	}
	/*cudaStatus = cudaMemcpy(ia_d, ia.begin_, (A.rows() + 1) * sizeof(int), cudaMemcpyHostToDevice);*/
	cudaStatus = cudaMemcpy(ia_d, A.outerIndexPtr(), (A.rows() + 1) * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
	}
	/*cudaStatus = cudaMemcpy(ja_d, ja.begin_, A.nonZeros() * sizeof(int), cudaMemcpyHostToDevice);*/
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


	// create intermediate variables
	double* rk; //residue
	double* pk;
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

	//setup geometry

	// solve at device side
	wrapper_PoissonSolverSparse_cusparese_device_kernel(blocksPerGrid, threadsPerBlock, A.nonZeros(), rhs_d, A_d, ia_d, ja_d, x_d, rk, pk, abstol, N, maxIter, Ap_rd, iter_residual, iters);

	// move back and write to the root vector
	cudaMemcpy(root.data(), x_d, vector_bytesize, cudaMemcpyDeviceToHost);
	//free and error handle

	cudaFree(A_d);
	cudaFree(ia_d);
	cudaFree(ja_d);
	cudaFree(rhs_d);
	cudaFree(x_d);
	cudaFree(rk);
	cudaFree(pk);
}

