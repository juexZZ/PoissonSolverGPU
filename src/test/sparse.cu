#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <iostream>
#include <string>
#include "../utils.h"
#include "../solver/sparsesolver.cuh"
#include "../solver/sparsetexturesolver.cuh"
#include "../solver/cusparsesolver.cuh"
#include <time.h>

using namespace std;



//Sparse solver use row major matrix, every row is assigned one thread to do the iteration

int main(int argc, char* argv[]) {
	if(argc!=6){
		cout<<"usage: ./cudatest path_to_data num_iter precision(0/1) method scale"<<endl;
       exit(1);
	}
	/*string PATH = "../../data/";*/
    //string PATH = "C:/NYU/gpu/PoissonSolver/data/";
	string PATH = string(argv[1]);
    int maxIter = atoi(argv[2]);
    bool double_precision = atoi(argv[3]); // 0 false, 1 true
    string algo = string(argv[4]);
    bool scale = atoi(argv[5]);
    bool icho = true;
    int iters = 0;
    cout << PATH << endl;
    clock_t gpu_s, gpu_e;
	if(double_precision){
        cout<<"using double precision.."<<endl;
	    Eigen::SparseMatrix<double> A;
	    readSymetric(A, PATH);
	    A.makeCompressed();
	    VectorXd rhs(A.cols()), x(A.rows());
	    rhs.setOnes();
	    x.setOnes();
        double iter_residual=0.0;
        if (scale)
        {
            std::vector<double> D(A.rows(), 1.0);
            for (ptrdiff_t i = 0; i < A.rows(); ++i) {
                for (ptrdiff_t j = A.outerIndexPtr()[i], e = A.outerIndexPtr()[i + 1]; j < e; ++j) {
                    if (A.innerIndexPtr()[j] == i) {
                        D[i] = 1 / sqrt(A.valuePtr()[j]);
                        break;
                    }
                }
            }
            for (ptrdiff_t i = 0; i < A.rows(); ++i) {
                for (ptrdiff_t j = A.outerIndexPtr()[i], e = A.outerIndexPtr()[i + 1]; j < e; ++j) {
                    A.valuePtr()[j] *= D[i] * D[A.innerIndexPtr()[j]];
                }
                rhs[i] *= D[i];
            }
        }
        
	    double reTol = 1e-8; //Relative error tolerence
	    unsigned int N = A.rows();
	    double abstol = reTol * reTol * rhs.norm();
        unsigned int threadsPerBlock = 256;
        unsigned int blocksPerGrid = ceil((double)N / 256.0);

        VectorXd root(N);

        // solve at device side
        gpu_s = clock();
        if (algo == "s"){
            wrapper_PoissonSolverSparse_multiblock<double>(blocksPerGrid, threadsPerBlock, 
                rhs, A, x, root, N, maxIter,abstol,iter_residual,iters);
        }
        else if (algo == "st"){
            Eigen::SparseMatrix<float> Af = A.cast<float>();
            wrapper_PoissonSolverSparse_texture_multiblock<double>(blocksPerGrid, threadsPerBlock, 
                rhs, Af, x, root, N, maxIter,abstol, iter_residual, iters);
        }
        else if (algo == "sc")
        {
            wrapper_PoissonSolverSparse_cusparse(blocksPerGrid, threadsPerBlock,
                rhs, A, x, root, N, maxIter, abstol, iter_residual, iters);
        }
        else{
            cout << "method not recognised! we support sparse(s), sparse texture(t),.."<<endl;
            exit(1);
        }
        gpu_e = clock();
        cout<<(double)(gpu_e-gpu_s)/ CLOCKS_PER_SEC <<endl;

        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
        }
        else {
            printf("CUDA NO error\n");
        }
        // check
        double err = (A * root - rhs).norm()/rhs.norm();
        if (err < 1e-5) {
            cout << "pass with err:" << err << endl;
        }
        else {
            cout << "not pass with err:" << err << endl;
        }
        cout << "iteration times " << iters << endl;
        cout << "Residual: " << iter_residual << endl;
        /*cout << "current solution: \n" << root << endl;*/
    }
    else{
        cout<<"using single precision..."<<endl;
        Eigen::SparseMatrix<double> A;
        readSymetric(A, PATH );
        VectorXf rhs(A.cols()), x(A.rows());
        rhs.setOnes();
        x.setOnes();
        float iter_residual=0.0;
        if (scale)
        {
            std::vector<float> D(A.rows(), 1.0);
            for (ptrdiff_t i = 0; i < A.rows(); ++i) {
                for (ptrdiff_t j = A.outerIndexPtr()[i], e = A.outerIndexPtr()[i + 1]; j < e; ++j) {
                    if (A.innerIndexPtr()[j] == i) {
                        D[i] = 1 / sqrt(A.valuePtr()[j]);
                        break;
                    }
                }
            }
            for (ptrdiff_t i = 0; i < A.rows(); ++i) {
                for (ptrdiff_t j = A.outerIndexPtr()[i], e = A.outerIndexPtr()[i + 1]; j < e; ++j) {
                    A.valuePtr()[j] *= D[i] * D[A.innerIndexPtr()[j]];
                }
                rhs[i] *= D[i];
            }
        }
        Eigen::SparseMatrix<float> Af = A.cast<float>();
        Af.makeCompressed();
        double reTol = 1e-3; //Relative error tolerence
        unsigned int N = A.rows();
        float abstol = reTol * reTol * rhs.norm();
        unsigned int threadsPerBlock = 256;
        unsigned int blocksPerGrid = ceil((double)N / 256.0);

        VectorXf root(N);

        // solve at device side
        gpu_s = clock();
        if (algo == "s") {
            wrapper_PoissonSolverSparse_multiblock<float>(blocksPerGrid, threadsPerBlock,
                rhs, Af, x, root, N, maxIter, abstol, iter_residual, iters);
        }
        else if (algo == "st") {
            wrapper_PoissonSolverSparse_texture_multiblock<float>(blocksPerGrid, threadsPerBlock,
                rhs, Af, x, root, N, maxIter, abstol, iter_residual, iters);
        }
        gpu_e = clock();
        cout << (double)(gpu_e - gpu_s) / CLOCKS_PER_SEC << endl;
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
        }
        else {
            printf("CUDA NO error\n");
        }
        // check
        double err = (Af * root - rhs).norm()/ rhs.norm();
        if (err < 1e-5) {
            cout << "pass with err:" << err << endl;
        }
        else {
            cout << "not pass with err:" << err << endl;
           /* cout << "current solution: \n" << root << endl;*/
        }
        cout << "iteration times " << iters << endl;
        cout << "Residual: " << iter_residual << endl;

    }
}