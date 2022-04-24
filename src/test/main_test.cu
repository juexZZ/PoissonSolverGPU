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
#include <unistd.h>
#include <string>
#include "../utils.h"
// #include "../solver/gpukernels.cuh"
#include "../solver/densesolver.cuh" //must see 
#include "../solver/texturesolver.cuh"

using namespace std;

bool double_precision = true; //default
int maxIter;
string algo;

void arg_parser(int argc, char** argv){
    int c;
    while((c=getopt(argc, argv, "i::a::p")) != -1){
        switch (c)
        {
        case 'i':
            maxIter = stoi(string(optarg));
            break;
        case 'a':
            algo = string(optarg);
            break;
        case 'p':
            double_precision = stoi(string(optarg));
        case '?':
            if (optopt == 'i' || optopt == 'a' || optopt == 'p')
                fprintf (stderr, "Option -%c requires an argument.\n", optopt);
            else if (isprint (optopt))
                fprintf (stderr, "Unknown option `-%c'.\n", optopt);
            else
                fprintf (stderr,
                        "Unknown option character `\\x%x'.\n",
                        optopt);
            exit(1);
        default:
            abort();
        }
    }
}

int main(int argc, char* argv[]) {
	if(argc!=4){
		cout<<"Usage: ./cudatest path_to_data -i num_iter -p precision(0/1) -a method"<<endl;
        exit(1);
	}
	/*string PATH = "../../data/";*/
    // string PATH = "C:/NYU/gpu/PoissonSolver/data/";
    arg_parser()
	string PATH = string(argv[1]);
    // double_precision = atoi(argv[3]); // 0 false, 1 true

    if (double_precision){
        cout<<"using double precision..."<<endl;
        Eigen::MatrixXd A = openData<double>(PATH + "test_100.csv");
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

        Eigen::VectorXd root;
        if (algo == "d"){
            root = wrapper_PoissonSolverDense<double>(
                blocksPerGrid, 
                threadsPerBlock, 
                rhs.data(),
                Arowmajor.data(),
                x.data(),
                abstol,
                N,
                maxIter
            );
        }
        else if (algo == "t"){
            // cast A to float
            EigenXf Arf = Arowmajor.cast<float>();
            root = wrapper_PoissonSolverTexture<double>(
                blocksPerGrid,
                threadsPerBlock,
                rhs.data(),
                Arf.data(),
                x.data(),
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
        int maxIter = atoi(argv[1]);
        // convert matrix to row-major storage
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Arowmajor = A;
        unsigned int N = A.rows();
        float abstol = reTol * reTol * rhs.norm();
        unsigned int threadsPerBlock = N;
        unsigned int blocksPerGrid = 1;

        Eigen::VectorXf root;
        if(algo=="d"){
            root = wrapper_PoissonSolverDense<float>(
                blocksPerGrid, 
                threadsPerBlock, 
                rhs.data(),
                Arowmajor.data(),
                x.data(),
                abstol,
                N,
                maxIter
            ); 
        }
        else if (algo=="t"){
            root = wrapper_PoissonSolverTexture<float>(
                blocksPerGrid,
                threadsPerBlock,
                rhs.data(),
                Arf.data(),
                x.data(),
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