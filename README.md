# PoissonSolverGPU

A GPU Solver for Poisson Problem 2D

### Test CUDA

Eigen library is the prerequisite. To use, download eigen release to your machine and include that path during compilation. See example below.

Double precision is only supported in computing compability >= 6.0, so make sure the machine satisfies. For example, this runs successfully on cims cuda2.

You should compile with this command:

`nvcc --std c++11 -I /home/jz4725/eigen-3.3.9 -o cudatest cudatest.cu -arch=sm_60 `

Then hit run:

`./cudatest num_iters`
