/*
test main
*/
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
//#include <unistd.h>
#include <string>
#include "../utils.h"
// #include "../solver/gpukernels.cuh"
#include "../solver/densesolver.cuh" //must see 
#include "../solver/texturesolver.cuh"
#include "../solver/sparsesolver.cuh"

using namespace std;
using namespace Eigen;

// bool double_precision = true; //default
// int maxIter=1000;
// string algo="t";

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

    if (double_precision){
        cout<<"using double precision..."<<endl;
        Eigen::MatrixXd A = openData<double>(PATH);
        Eigen::VectorXd rhs(A.cols()), x(A.rows());
        rhs.setOnes();
        x.setOnes();
        double reTol = 1e-8; //Relative error tolerence
        // convert matrix to row-major storage
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Arowmajor = A;
        unsigned int N = A.rows();
        double abstol = reTol * reTol * rhs.norm();
        //setup geometry
        unsigned int threadsPerBlock = N;
        unsigned int blocksPerGrid = 1;

        Eigen::VectorXd root(N);
        if (algo == "d"){
            wrapper_PoissonSolverDense<double>(
                blocksPerGrid, 
                threadsPerBlock, 
                rhs.data(),
                Arowmajor.data(),
                x.data(),
                root,
                abstol,
                N,
                maxIter
            );
        }
        else if (algo == "t"){
            // cast A to float
            MatrixXf Arf = Arowmajor.cast<float>();
            wrapper_PoissonSolverTexture<double>(
                blocksPerGrid,
                threadsPerBlock,
                rhs.data(),
                Arf.data(),
                x.data(),
                root,
                abstol,
                N,
                maxIter
            );
        }
        else{
            cout << "method not recognised! we support dense(d), texture(t),.."<<endl;
            exit(1);
        }
        
        //cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if(error!=cudaSuccess){
            printf("CUDA error: %s\n", cudaGetErrorString(error));
        }else{
            printf("CUDA NO error\n");
        }

        // check
        double err = (A * root - rhs).norm();
        if (err < 1e-5){
            cout << "pass with err:"<< err << endl;
        }
        else {
            cout << "not pass with err:"<< err << endl;
            cout << "current solution: \n"<<root<< endl;
        }
    }
    else{
        cout<<"using single precision..."<<endl;
        Eigen::MatrixXf A = openData<float>(PATH + "test_100.csv");
        Eigen::VectorXf rhs(A.cols()), x(A.rows());
        rhs.setOnes();
        x.setOnes();
        float reTol = 1e-3; //Relative error tolerence
        // convert matrix to row-major storage
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Arowmajor = A;
        unsigned int N = A.rows();
        float abstol = reTol * reTol * rhs.norm();
        unsigned int threadsPerBlock = N;
        unsigned int blocksPerGrid = 1;

        Eigen::VectorXf root(N);
        if(algo=="d"){
        wrapper_PoissonSolverDense<float>(
                blocksPerGrid, 
                threadsPerBlock, 
                rhs.data(),
                Arowmajor.data(),
                x.data(),
                root,
                abstol,
                N,
                maxIter
            ); 
        }
        else if (algo=="t"){
        wrapper_PoissonSolverTexture<float>(
                blocksPerGrid,
                threadsPerBlock,
                rhs.data(),
                Arowmajor.data(),
                x.data(),
                root,
                abstol,
                N,
                maxIter
            );
        }
        else{
            cout << "method not recognised! we support dense(d), texture(t),.."<<endl;
            exit(1);
        }

        //cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if(error!=cudaSuccess){
            printf("CUDA error: %s\n", cudaGetErrorString(error));
        }else{
            printf("CUDA NO error\n");
        }

        // check
        float err = (A * root - rhs).norm();
        if (err < 1e-4){
            cout << "pass with err:"<< err << endl;
        }
        else {
            cout << "not pass with err:"<< err << endl;
            cout << "current solution: \n"<<root<< endl;
        }
    }
}