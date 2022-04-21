#pragma once
// gpu kernel function wrappers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void wrapper_PoissonSolverDense(unsigned int blocksPerGrid,
	unsigned int threadsPerBlock,
	double* rhs_d,
	double* A_d,
	double* x_d,
	double* rk,
	double* pk,
	double abstol,
	unsigned int N,
	int maxIter);