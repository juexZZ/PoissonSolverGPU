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


void wrapper_PoissonSolverSparse_cusparese(unsigned int blocksPerGrid,
	unsigned int threadsPerBlock,
	cusparseSpMatDescr_t* A_sparse,
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
	double* r_k_norm,
	double* r_k1_norm,
	double* pAp_k) {
	double temp = 0;
	printf("N = %d, kernel has %d blocks each has %d threads\n", N, blocksPerGrid, threadsPerBlock);
	cusparseHandle_t handle;
	cublasHandle_t handle_blas;
	cusparseDnVecDescr_t* rhs_vec;
	cusparseDnVecDescr_t* x_vec;
	cusparseDnVecDescr_t* rk_vec;
	cusparseDnVecDescr_t* pk_vec;
	cusparseDnVecDescr_t* Ap_r_vec;
	CHECK_CUSPARSE(cusparseCreateDnVec(rhs_vec, N, rhs_d, CUDA_R_64F) );
	CHECK_CUSPARSE(cusparseCreateDnVec(x_vec, N, x_d, CUDA_R_64F));
	CHECK_CUSPARSE(cusparseCreateDnVec(rk_vec, N, rk, CUDA_R_64F));
	CHECK_CUSPARSE(cusparseCreateDnVec(pk_vec, N, pk, CUDA_R_64F));
	CHECK_CUSPARSE(cusparseCreateDnVec(Ap_r_vec, N, Ap_rd, CUDA_R_64F));
	size_t buffersize=0;
	cusparseDnVecDescr_t* tempvec; //Use to receive the multiplication result
	//Initialize 
	{ 
		double alpha = -1.0;
		double beta = 1.0;	
		*tempvec = *rhs_vec;
		CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, *A_sparse, *x_vec, &beta, *tempvec, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,&buffersize));
		void* dBuffer = NULL;
		CHECK_CUDA(cudaMalloc(&dBuffer, buffersize));
		CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, *A_sparse,*x_vec,&beta,*tempvec,CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT,dBuffer));
		*rk_vec = *tempvec;
		*pk_vec = *rk_vec;
		cublasDdot(handle_blas, N, rk, 1, rk, 1, r_k_norm);
		cublasDcopy(handle_blas, 1, r_k_norm,1, r_k1_norm,1);

	}
	//Iteration
	for (size_t i = 0; i < maxIter; i++)
	{
		{//Ap_r_vec=A*pk
			const double alpha = 1.0;
			const double beta = 0;
			CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, *A_sparse, *pk_vec, &beta, *Ap_r_vec, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buffersize));
			void* dBuffer = NULL;
			CHECK_CUDA(cudaMalloc(&dBuffer, buffersize));
			CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, *A_sparse, *pk_vec, &beta, *Ap_r_vec, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, dBuffer));
			
		}
	}
	
}