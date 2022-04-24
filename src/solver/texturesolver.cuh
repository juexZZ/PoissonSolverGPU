/* 
A cuda Poisson solver kernel implementation that 
uses texture memory
*/
#include <cuda.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "gpukernels.h"

/* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)

__device__  void selfatomicCAS(float* address, float val) {
	unsigned int* address_as_ull =
		(unsigned int*)address;
	unsigned int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
            __float_as_uint(val));

		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);
}

__global__ void PoissonSolverTexture(float* b0, float* xk, float* rk, float* pk,
    float eps, unsigned int N, unsigned int iters);

__device__ float r_k_norm = 0.0;
__device__ float r_k1_norm = 0.0;
__device__ float pAp_k = 0.0;
// __device__ bool norm_updated = false;

texture<float, cudaTextureType1D, cudaReadModeElementType> Atext;

__global__ 
void PoissonSolverTexture(float* b0, float* xk, float* rk, float* pk,
                        float eps, unsigned int N, unsigned int iters){
    
    // 1 dimesion geometry
    int ri = blockDim.x * blockIdx.x + threadIdx.x;
    if (ri < N){
        // initial, compute r0 and p0
        float Ax0_r = 0;
        //float x0_r = xk[ri];
        for(unsigned int j=0; j<N; j++){
            float Aik = tex1Dfetch<float>(Atext, index(ri,j,N));//A[index(ri, j, N)];
            Ax0_r += Aik * xk[j];
        }
        float b_r = b0[ri];
        rk[ri] = b_r - Ax0_r;
        pk[ri] = b_r - Ax0_r;
        atomicAdd(&r_k_norm, rk[ri]*rk[ri]); //&r_k_norm takes addr
        __syncthreads();
        // iterate here
        for(unsigned int itk = 0; itk<iters; itk++){
            // compute \alpha_k
            float Ap_r = 0;
            float pk_r = pk[ri];
            for(unsigned int j=0; j<N; j++){
                float Aik = tex1Dfetch<float>(Atext, index(ri,j,N)); //A[index(ri, j, N)];
                Ap_r += Aik * pk[j];
            }
            atomicAdd(&pAp_k, pk_r*Ap_r);
            selfatomicCAS(&r_k1_norm, 0.0);//r_k1_norm = 0.0;
            __syncthreads();
            float alpha_k = r_k_norm / pAp_k; //solve contension by getting one for each thread
            // update x_k to x_{k+1}, r_k to r_{k+1}
            xk[ri] = xk[ri] + alpha_k * pk_r;
            rk[ri] = rk[ri] - alpha_k * Ap_r;
            // compute beta k
            atomicAdd(&r_k1_norm, rk[ri]*rk[ri]);
            __syncthreads();
            // terminating condition:
            if(r_k1_norm < eps){
                break;
            }
            float temp_r_k1_norm = r_k1_norm;
            float beta_k = r_k1_norm / r_k_norm; //solve contension by getting one for each thread
            // update pk to pk1
            pk[ri] = rk[ri] + beta_k * pk[ri];
            __syncthreads();
            selfatomicCAS(&pAp_k, 0.0);
            selfatomicCAS(&r_k_norm, temp_r_k1_norm); // update rk norm to r_k1_norm, set r_k1_norm to 0 before next iter.
        }
    }
}
    
// wrapper function 
void wrapper_PoissonSolverTexture(unsigned int blocksPerGrid,
    unsigned int threadsPerBlock,
    float* rhs_d,
    float* A_d,
    float* x_d,
    float* rk,
    float* pk,
    float abstol,
    unsigned int N,
    int maxIter) {
    // call cuda kernel functions here
    // option1 whole matrix as texture
    // texture<float, cudaTextureType1D, cudaReadModeElementType> Atext;
    cudaBindTexture(NULL, Atext, A_d, N*N*sizeof(float));
    // option2 each row as texture, separately (maybe need an array of texture reference)
    // TODO? difficult to decide how many reference to use.
    // printf("N = %d, kernel has %d blocks each has %d threads\n", N, blocksPerGrid, threadsPerBlock);
    PoissonSolverTexture<<<blocksPerGrid, threadsPerBlock>>>(rhs_d, x_d, rk, pk, abstol, N, maxIter);
    cudaUnbindTexture(Atext);
}
