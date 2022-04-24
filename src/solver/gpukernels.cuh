#pragma once
// gpu kernel function wrappers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <Eigen/Dense>

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> wrapper_PoissonSolverDense(unsigned int blocksPerGrid,
    unsigned int threadsPerBlock,
    T* rhs_dptr,
    T* A_dptr,
    T* x_dptr,
    T abstol,
    unsigned int N,
    int maxIter);

void wrapper_PoissonSolverSparse(unsigned int blocksPerGrid,
	unsigned int threadsPerBlock,
	double* rhs_d,
	double* A_d,
	double* x_d,
	double* rk,
	double* pk,
	double abstol,
	unsigned int N,
	int maxIter);

void wrapper_PoissonSolverTexture(unsigned int blocksPerGrid,
    unsigned int threadsPerBlock,
    float* rhs_d,
    float* A_d,
    float* x_d,
    float* rk,
    float* pk,
    float abstol,
    unsigned int N,
    int maxIter);

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
	int maxIter,double* Ap_rd,double* r_k_norm,double* r_k1_norm,double* pAp_k);