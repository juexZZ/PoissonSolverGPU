#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <iostream>
#include <string>
#include "../utils.h"
#include "../solver/texturesolver.cuh"

using namespace std;



int main(int argc, char* argv[]) {
	if(argc!=2){
		cout<<"Usage: ./cudatest num_iter"<<endl;
	}
	/*string PATH = "../../data/";*/
	string PATH = "C:/NYU/gpu/PoissonSolver/data/";
	Eigen::MatrixXf A = openData<float>(PATH + "test_100.csv");
	Eigen::VectorXf rhs(A.cols()), x(A.rows()), x_cg(A.rows());
	rhs.setOnes();
	x.setZero();
	//for (size_t i = 0; i < A.rows(); i++)
	//{
	//	A.row(i) = A.row(i) / A(i,i);
	//	rhs[i] = rhs[i] / A(i,i);
	//}
	float reTol = 1e-8; //Relative error tolerence
	//float absTol = 0;
	int maxIter = atoi(argv[1]);
	cudaError_t cudaStatus;
	unsigned int matrix_bytesize = A.size() * sizeof(float); // NxN
	unsigned int N = A.rows();
	unsigned int vector_bytesize = N * sizeof(float);
	float abstol = reTol * reTol * rhs.norm();
	// convert matrix to row-major storage
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Arowmajor = A;
	// allocate and move to device
	float* rhs_d; // b(rhs) on device
	float* A_d; // A on device
	float* x_d; //x on device
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

	cudaStatus = cudaMemcpy(A_d, Arowmajor.data(), matrix_bytesize, cudaMemcpyHostToDevice);
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
	float* rk; //residue
	float* pk;
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
	unsigned int threadsPerBlock = N;
	unsigned int blocksPerGrid = 1;

	// solve at device side
	wrapper_PoissonSolverTexture(blocksPerGrid, threadsPerBlock, rhs_d, A_d, x_d, rk, pk, abstol, N, maxIter);

	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if(error!=cudaSuccess){
		printf("CUDA error: %s\n", cudaGetErrorString(error));
    }else{
		printf("CUDA NO error\n");
	}
	// move back and write to the root vector
	Eigen::VectorXf root(N);
	cudaMemcpy(root.data(), x_d, vector_bytesize, cudaMemcpyDeviceToHost);
	//free and error handle

	cudaFree(A_d);
	cudaFree(rhs_d);
	cudaFree(x_d);
	cudaFree(rk);
	cudaFree(pk);

	// check
	float err = (A * root - rhs).norm();
	if (err < 1e-5){
		cout << "pass with err:"<< err << endl;
	}
	else {
		cout << "not pass with err:"<< err << endl;
		cout << "current solution: \n"<<root<< endl;
	}
}