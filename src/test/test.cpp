#include <catch2/catch.hpp>
#include "../utils.h"
#include "../solver/cg.h"
using namespace std;
using namespace Eigen;
TEST_CASE("LoadMarket") {
	const std::string path = "E:/NYU/GPU_Poisson/PoissonSolver/data";
	Eigen::SparseMatrix<double> A;
	const bool ok = loadMarket(path + "/poisson3Db.mtx",A);
	REQUIRE(ok);	
}
TEST_CASE("DynamicMatrix") {
	MatrixXd a;
	a.resize(100, 100);
	a.setOnes();
	cout << a.rows() << endl;
	cout << a.cols() << endl;
	/*REQUIRE(a.rows() == 100);*/
}
//
//TEST_CASE("CreateRandom") {
//	int n = 100;
//	MatrixXd a,b;
//	denseSPD(n, a);
//	string PATH ="E:/NYU/GPU_Poisson/PoissonSolver/data/";
//	saveData(PATH+"r1" + to_string(n) + ".csv", a);
//	b=openData(PATH+"r1" + to_string(n) + ".csv");
//	REQUIRE((a-b).norm()<1e-8);
//}

TEST_CASE("CGSequential") {
	string PATH = "E:/NYU/GPU_Poisson/PoissonSolver/data/";
	MatrixXd A = openData(PATH + "test_100.csv");
	VectorXd b(A.cols()),x(A.rows()),x_cg(A.rows());
	b.setOnes();
	x.setZero();
	cgDenseSolver testSolver(A,b,x);
	x=testSolver.solve();
	REQUIRE((A * x - b).norm() < 1e-5);
	// fill A and b
	ConjugateGradient<MatrixXd> cg;
	cg.compute(A);
	x_cg = cg.solve(b);
	REQUIRE((A * x_cg - b).norm() < 1e-5);
}

