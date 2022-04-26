#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cusparse.h"
#include "cublas_v2.h"
#include <cuda.h>
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include "../utils.h"
#include "../solver/cusparsesolver.cuh"
using namespace std;
using namespace Eigen;

int main(int argc, char* argv[]) {
	if (argc != 2) {
		cout << "Usage: ./cudatest num_iter" << endl;
	}

	/*string PATH = "../../data/";*/
	string PATH = "C:/NYU/gpu/PoissonSolver/data/";
	Eigen::SparseMatrix<double> A;
	readSymetric(A, PATH + "nos6.mtx");
	A.makeCompressed();
	VectorXd rhs(A.cols()), x(A.rows()), x_cg(A.rows());
	rhs.setOnes();
	x.setZero();
	double reTol = 1e-8; //Relative error tolerence
	int maxIter = atoi(argv[1]);
	cudaError_t cudaStatus;
	unsigned int N = A.rows();
	unsigned int vector_bytesize = N * sizeof(double);
	double abstol = reTol * reTol * rhs.norm();
	// convert matrix to row-major storage
	// allocate and move to device
	double* rhs_d; // b(rhs) on device
	// Triplet for A
	double* A_d; //csrValA
	int* ia_d; //csrRowPtrA
	int* ja_d;//csrColIndA 
	double* r_k_norm;
	double* r_k1_norm;
	double* pAp_k;

	double* x_d; //x on device
	double* Ap_rd;
	double* Ap_r;
	double initial = 0.0;
	cusparseHandle_t handle;
	cusparseSpMatDescr_t* A_sparse;
	cusparseCreateCsr(A_sparse,
		N,
		N,
		A.nonZeros(),
		ia_d,
		ja_d,
		A_d,
		CUSPARSE_INDEX_32I,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

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

	cudaStatus = cudaMalloc((void**)&r_k_norm, sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}
	cudaStatus = cudaMalloc((void**)&r_k1_norm, sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
	}
	cudaStatus = cudaMalloc((void**)&pAp_k, sizeof(double));
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
	cudaStatus = cudaMemcpy(Ap_rd, Ap_r, vector_bytesize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
	}
	cudaStatus = cudaMemcpy(r_k_norm, &initial, sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
	}
	cudaStatus = cudaMemcpy(r_k1_norm, &initial, sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
	}
	cudaStatus = cudaMemcpy(pAp_k, &initial, sizeof(double), cudaMemcpyHostToDevice);
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
	unsigned int threadsPerBlock = 256;
	unsigned int blocksPerGrid = ceil((double)N / 256.0);

	// solve at device side
	wrapper_PoissonSolverSparse_cusparese(blocksPerGrid, threadsPerBlock,A_sparse,rhs_d, A_d, ia_d, ja_d, x_d, rk, pk, abstol, N, maxIter, Ap_rd, r_k_norm, r_k1_norm, pAp_k);

	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(error));
	}
	else {
		printf("CUDA NO error\n");
	}
	// move back and write to the root vector
	VectorXd root(N);
	cudaMemcpy(root.data(), x_d, vector_bytesize, cudaMemcpyDeviceToHost);
	//free and error handle

	cudaFree(A_d);
	cudaFree(ia_d);
	cudaFree(ja_d);
	cudaFree(rhs_d);
	cudaFree(x_d);
	cudaFree(rk);
	cudaFree(pk);

	// check
	double err = (A * root - rhs).norm();
	if (err < 1e-5) {
		cout << "pass with err:" << err << endl;
	}
	else {
		cout << "not pass with err:" << err << endl;
		cout << "current solution: \n" << root << endl;
	}