# PoissonSolverGPU

A GPU Solver for Poisson Problem 2D

## Compile

Eigen library is the prerequisite. To use, download eigen release to your machine and include that path during compilation. See example below.

Double precision is only supported in computing compability >= 6.0, so make sure the machine satisfies.

First you need:

`cd ./src/test`

To make sure the project can compile, load cuda version 11+:

`module load cuda-11.4`

To compile the dense version:

`nvcc --std c++11 -I /PATH/TO/eigen-3.3.9 -o dense dense.cu -arch=sm_60 `

To compile the sparse version:

`nvcc --std c++11 -I /PATH/TO/eigen-3.3.9 -o sparse sparse.cu -arch=sm_60 -lcublas -lcusparse `

## Run

To run the dense solver, run the following command (arguments are explained below):

`./dense PATH/TO/DATA NUM_ITER PRECISION ALGO`

To run the sparse solver, run the following command:

`./sparse PATH/TO/DATA NUM_ITER PRECISION ALGO SCALE`

The arguments:

* PATH/TO/DATA: path to the data file
* NUM_ITER: number of maximum iterations, should be an integer
* PRECISION: whether to use single or double precision. 0 for single precision, 1 for double precision
* ALGO: the string specifying which algorithm to use.
  * For the dense solver, two are supported: "d" for dense, "dt" for dense with texture memory.
  * For the sparse solver, the supported listed below:"s" for sparse, "st" for sparse with texture, "sc" for the cuSparse
* SCALE: wether to apply scale trick: 0 for not apply, 1 for apply.

For example:

`./dense ../../data/test_100.csv 30000 1 d`

`./sparse ../../data/nos6.mtx 30000 1 s 1`
