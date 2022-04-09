#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;
using namespace Eigen;

class BoundaryCondition {
public:
	bool is_Dirichlet = true; //True for Dirichlet condition and false for cauchy condition. 
	int location = 0; // 0=left side, 1=up side, 2=right side, 3=down side, 4=front side, 5=back side
	MatrixXd _value;
	MatrixXd _xDiff;
	MatrixXd _yDiff;
	MatrixXd _zDiff;
};

class Problem {
public:
	int xlength=0; 
	int ylength=0;
	int zlength=0;
	double dx=1, dy=1, dz=1;
	bool is_2d = true;
	vector<BoundaryCondition> bc;
	Problem() {};
	Problem(int x, int y,int z, bool _is_2d) {
		xlength = x;
		ylength = y;
		zlength = z;
		is_2d = _is_2d;
	}
	//Check condition if it is capable
	//Every boundary should have a conition,if not return -1
	//If all conditions are cauchy condition, then return 2, in this case we should fix the (0,0) to 0
	//If it is all fine, then return 1
	int conditionCheck() {
		if (is_2d)
		{
			vector<bool> _Bd{ false,false,false,false };
			bool exist_Dirichlet=false;
			for (size_t i = 0; i < bc.size(); i++)
			{
				if (bc[i].location<_Bd.size())
				{
					_Bd[bc[i].location] = true;
					if (bc[i].is_Dirichlet == true)
					{
						exist_Dirichlet = true;
					}
				}
			}
			bool is_capable = true;
			for (size_t i = 0; i <_Bd.size(); i++)
			{
				is_capable = _Bd[i] && is_capable;
			}
			if (is_capable)
			{
				if (exist_Dirichlet)
				{
					return 1;
				}
				else
				{
					return 2;
				}
			}
			else
			{
				return -1;
			}

		}
		else
		{
			vector<bool> _Bd{ false,false,false,false,false,false };
			bool exist_Dirichlet = false;
			for (size_t i = 0; i < bc.size(); i++)
			{
				if (bc[i].location < _Bd.size())
				{
					_Bd[bc[i].location] = true;
					if (bc[i].is_Dirichlet == true)
					{
						exist_Dirichlet = true;
					}
				}
			}
			bool is_capable = true;
			for (size_t i = 0; i < _Bd.size(); i++)
			{
				is_capable = _Bd[i] && is_capable;
			}
			if (is_capable)
			{
				if (exist_Dirichlet)
				{
					return 1;
				}
				else
				{
					return 2;
				}
			}
			else
			{
				return -1;
			}
		}
	}
};

void preConditioner(Problem pro,MatrixXd &output,VectorXd &rhs) {
	//The poisson preconditioner
	int conCheck = pro.conditionCheck();
	if (conCheck==-1)
	{
		cout << "Boundary condition is not capable " << endl;
		exit(0);
	}
	if (pro.is_2d)
	{
		MatrixXd D(pro.xlength - 2, pro.xlength - 2);
		D.setZero();
		for (size_t i = 0; i < pro.xlength-2; i++)
		{
			D(i, i + 1) = -1;
		}
		D = D + D.transpose();
		D = D + 4 * MatrixXd::Identity(pro.xlength-2,pro.xlength-2);
		output.resize((pro.xlength - 2)*(pro.ylength-2), (pro.xlength - 2)* (pro.ylength - 2));
		output.setZero();
		for (size_t i = 0; i < pro.ylength-2; i++)
			for (size_t j = 0; j < pro.ylength-2; j++)
		{	
			//The matrix is (m-2)*n-2

		}
	}
	else 
	{
		output.resize((pro.xlength - 2) * (pro.ylength - 2)*(pro.zlength-2), (pro.xlength - 2) * (pro.ylength - 2)*(pro.zlength-2));

	}
	
}

void conjugateGradient(Problem pro, VectorXd result) {
	//Solving problems AX=B
	MatrixXd A;
	VectorXd B,X;
	for (size_t i = 0; i < pro.bc.size(); i++)
	{
		//To do, set rhs
		//If it is the Dirichlet bc, then set the corresbonding rhs to the _value,do not need to update matrix output
		//If it is the Cauchy bc, then set the corresbonding rhs to the dx/dy/dz*close boundary term
	}

}
