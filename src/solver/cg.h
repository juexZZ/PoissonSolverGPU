#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <cuda.h>
#include "cuda_runtime.h"
#include "gpukernels.h"

using namespace std;
using namespace Eigen;

class cgDenseSolver
{
public:
	cgDenseSolver(MatrixXd _A, VectorXd _rhs) 
	{
		if (_A.cols()!=_rhs.size())
		{
			cout << "The Matrix cannot match the rhs vector, construction fail" << endl;
		}
		else {
			A = _A;
			rhs = _rhs;
			_initx.resize(A.rows());
			_initx.setOnes();
		}
	};
	cgDenseSolver(MatrixXd _A, VectorXd _rhs,VectorXd _x)
	{
		if (_A.cols() != _rhs.size() || _A.rows()!=_x.size())
		{
			cout << "The Matrix cannot match the rhs vector, construction fail" << endl;
		}
		else {
			A = _A;
			rhs = _rhs;
			_initx.resize(A.rows());
			_initx = _x;
		}
	};
	~cgDenseSolver() {};
public:
	MatrixXd A;
	VectorXd rhs;
	VectorXd _initx;
	int maxIter=500;
	double reTol=1e-8; //Relative error tolerence
	double absTol = 0;// If absTol!=0, the solver will stop when satisfy the absTolr, otherwise the solver will satisfy the relative error tolerence;
public:
	VectorXd solve() {
		double tol;// Square of absTol, then only compare the (f-Ax).norm with tol 
		if (absTol>0)
		{
			tol = absTol*absTol;
		}
		else
		{
			tol = reTol*reTol * rhs.norm();
		}
		int iter = 0;
		VectorXd x = _initx;
		VectorXd rk=rhs-A*x;
		VectorXd pk=rk;
		double alpha;
		double beta;
		for (iter=0; iter < maxIter; iter++)
		{
			VectorXd rk1;
			alpha = rk.squaredNorm() / (pk.transpose() * A * pk)[0];
			x = x + alpha * pk;
			rk1 = rk - alpha * A * pk;
			if (rk1.norm()<tol)
			{
				rk = rk1;
				break;
			}
			beta = rk1.squaredNorm() / (rk.squaredNorm());
			pk = rk1 + beta * pk;
			rk = rk1;
		}
		if (absTol>0)
		{
			cout << "Iterations  " << iter  << "Absolute Residual Error  " << rk.norm() << endl;
		}
		else
		{
			cout << "Iterations  " << iter << " Relative Residual Error  " << rk.norm()/rhs.norm() << endl;
		}
		return x;
		
	}

	VectorXd solve_gpu_dense(){
		/*
		A GPU solver for dense case, 1st version
		*/
		cudaError_t cudaStatus;

		unsigned int matrix_bytesize = A.size()*sizeof(double); // NxN
		unsigned int N = A.rows();
		unsigned int vector_bytesize = N * sizeof(double);
		double abstol = reTol * reTol * rhs.norm();
		// convert matrix to row-major storage
		Matrix<double, Eigen::Dynamic, Eigen::Dynamic, RowMajor> Arowmajor = A;
		// allocate and move to device
		double* rhs_d; // b(rhs) on device
		double* A_d; // A on device
		double* x_d; //x on device
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
		cudaStatus = cudaMemcpy(x_d, _initx.data(), vector_bytesize, cudaMemcpyHostToDevice);
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
		unsigned int threadsPerBlock = N;
		unsigned int blocksPerGrid = 1;
		
		// solve at device side
		wrapper_PossionSolverDense(blocksPerGrid, threadsPerBlock, rhs_d, A_d, x_d, rk, pk, abstol, N, maxIter);

		// move back and write to the root vector
		VectorXd root(N);
		cudaMemcpy(root.data(), x_d, vector_bytesize, cudaMemcpyDeviceToHost);
		//free and error handle
	
		cudaFree(A_d);
		cudaFree(rhs_d);
		cudaFree(x_d);
		cudaFree(rk);
		cudaFree(pk);

		return root;
	}
};


