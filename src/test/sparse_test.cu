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

using namespace std;


//Sparse solver use row major matrix, every row is assigned one thread to do the iteration

int main(int argc, char* argv[]) {
	if(argc!=5){
		cout<<"Usage: ./cudatest path_to_data num_iter precision(0/1) method"<<endl;
       exit(1);
	}
	/*string PATH = "../../data/";*/
    // string PATH = "C:/NYU/gpu/PoissonSolver/data/";
	string PATH = string(argv[1]);
    int maxIter = atoi(argv[2]);
    bool double_precision = atoi(argv[3]); // 0 false, 1 true
    string algo = string(argv[4]);

	if(double_precision){
        cout<<"using double precision.."<<endl;
	    Eigen::SparseMatrix<double> A;
	    readSymetric(A, PATH + "nos6.mtx");
	    A.makeCompressed();
	    VectorXd rhs(A.cols()), x(A.rows()), x_cg(A.rows());
	    rhs.setOnes();
	    x.setZero();
	    double reTol = 1e-8; //Relative error tolerence
	    unsigned int N = A.rows();
	    double abstol = reTol * reTol * rhs.norm();

        VectorXd root(N);

        // solve at device side
        if (algo == "s"){
            wrapper_PoissonSolverSparse_multiblock<double>(blocksPerGrid, threadsPerBlock, 
                rhs, A, x, root, N, maxIter,abstol);
        }
        else if (algo == "st"){
            Eigen::SparseMatrix<float> Af = A.cast<float>();
            wrapper_PoissonSolverSparse_texture_multiblock<double>(blocksPerGrid, threadsPerBlock, 
                rhs, Af, x, root, N, maxIter,abstol);
        }
        else{
            cout << "method not recognised! we support sparse(s), sparse texture(t),.."<<endl;
            exit(1);
        }

        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
        }
        else {
            printf("CUDA NO error\n");
        }
        // check
        double err = (A * root - rhs).norm();
        if (err < 1e-5) {
            cout << "pass with err:" << err << endl;
        }
        else {
            cout << "not pass with err:" << err << endl;
            cout << "current solution: \n" << root << endl;
        }
    }
    else{
        cout<<"using single precision..."<<endl;
        Eigen::SparseMatrix<double> A;
        readSymetric(A, PATH + "nos6.mtx");
        Eigen::SparseMatrix<float> Af = A.cast<float>();
        Af.makeCompressed();
        VectorXf rhs(A.cols()), x(A.rows()), x_cg(A.rows());
        rhs.setOnes();
        x.setZero();
        double reTol = 1e-3; //Relative error tolerence
        unsigned int N = A.rows();
        double abstol = reTol * reTol * rhs.norm();

        VectorXf root(N);

        // solve at device side
        wrapper_PoissonSolverSparse_multiblock<double>(blocksPerGrid, threadsPerBlock, 
            rhs, Af, x, root, N, maxIter,abstol);

        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
        }
        else {
            printf("CUDA NO error\n");
        }
        // check
        double err = (Af * root - rhs).norm();
        if (err < 1e-5) {
            cout << "pass with err:" << err << endl;
        }
        else {
            cout << "not pass with err:" << err << endl;
            cout << "current solution: \n" << root << endl;
        }
    }
}