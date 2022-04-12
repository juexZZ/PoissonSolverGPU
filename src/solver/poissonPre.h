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

class PoissonProblem {
public:
	int xlength=0; 
	int ylength=0;
	int zlength=0;
	double dx=1, dy=1, dz=1;
	bool is_2d = true;
	vector<BoundaryCondition> bc;
	PoissonProblem() {};
	PoissonProblem(int x, int y,int z, bool _is_2d) {
		xlength = x;
		ylength = y;
		zlength = z;
		is_2d = _is_2d;
	}
	//Check condition if it is capable
	//Every boundary should have a conition,if not return -1
	//If all conditions are cauchy condition, then return 2, in this case we should fix the (0,0)/(0,0,0) to 0
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
		//3D problem
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

void preConditioner(PoissonProblem pro,MatrixXd &output,VectorXd &rhs) {
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
		for (size_t i = 0; i < pro.ylength-3; i++)
		{	
			//The matrix is (m-2)*(n-2)
				output.block(i*(pro.xlength-2), i * (pro.xlength - 2), pro.xlength - 2, pro.xlength - 2)=D;
				output.block((i+1) * (pro.xlength - 2), i * (pro.xlength - 2), pro.xlength - 2, pro.xlength - 2) = -MatrixXd::Identity(pro.xlength - 2, pro.xlength - 2);
				output.block(i * (pro.xlength - 2), (i+1) * (pro.xlength - 2), pro.xlength - 2, pro.xlength - 2) = -MatrixXd::Identity(pro.xlength - 2, pro.xlength - 2);
		}
		output.block((pro.ylength - 2) * (pro.xlength - 2), (pro.ylength - 2) * (pro.xlength - 2), pro.xlength - 2, pro.xlength - 2) = D;
		MatrixXd temp((pro.xlength - 2) * (pro.ylength - 2), (pro.xlength - 2) * (pro.ylength - 2));
		for (size_t i = 0; i < pro.xlength - 2; i++)
		{
			bool xub = (i < pro.xlength - 3);
			bool xlb = (i > 1);
			for (size_t j = 0; j < pro.ylength - 2; j++)
			{
				temp(i * j, i * j) = 4;				
				bool yub = (j < pro.ylength - 3);
				bool ylb = (j > 1);
				if (ylb)
				{
					temp(i * j, i * (j - 1)) = -1;
				}
				if (yub)
				{
					temp(i * j, i * (j + 1)) = -1;
				}
				if (xub)
				{
					temp(i * j, (i + 1) * j) = -1;
				}
				if (xlb)
				{
					temp(i * j, (i - 1) * j) = -1;
				}	
			}
		}
		assert(output == temp);

	}
	else 
	{
		output.resize((pro.xlength - 2) * (pro.ylength - 2)*(pro.zlength-2), (pro.xlength - 2) * (pro.ylength - 2)*(pro.zlength-2));
		for (size_t i = 0; i < pro.xlength - 2; i+ pro.xlength - 3)
		{
			bool xub = (i < pro.xlength - 3);
			bool xlb = (i > 1);
			for (size_t j = 0; j < pro.ylength - 2; j+pro.ylength - 3)
			{
				bool yub = (j < pro.ylength - 3);
				bool ylb = (j > 1);
				for (size_t k = 0; k < pro.zlength-2; k+ pro.zlength-3)
				{
					bool zub = (k < pro.zlength - 3);
					bool zlb = (k > 1);
					output(i * j * k,i*j*k) = 6;
					if (ylb)
					{
						output(i * j*k, i * (j - 1)*k) = -1;
					}
					if (yub)
					{
						output(i * j*k, i * (j + 1)*k) = -1;
					}
					if (xub)
					{
						output(i * j*k, (i + 1) * j*k) = -1;
					}
					if (xlb)
					{
						output(i * j*k, (i - 1) * j*k) = -1;
					}
					if (zub)
					{
						output(i * j * k, i * j * (k+1)) = -1;
					}
					if (zlb)
					{
						output(i * j * k, i * j * (k-1)) = -1;
					}
				}
			}
		}

		for (size_t i = 1; i < pro.xlength - 3; i++)
		{
			for (size_t j = 1; j < pro.ylength - 3; j++)
			{
				for (size_t k = 1; k < pro.zlength-3; k++)
				{
					output(i * j * k, i * j * k) = 6;
					output(i * j * k, i * (j - 1) * k) = -1;
					output(i * j * k, i * (j + 1) * k) = -1;
					output(i * j * k, (i + 1) * j * k) = -1;
					output(i * j * k, (i - 1) * j * k) = -1;
					output(i * j * k, i * j * (k + 1)) = -1;
					output(i * j * k, i * j * (k - 1)) = -1;
				}
			}
		}
	}

	
}

void conjugateGradient(PoissonProblem pro, VectorXd result) {
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
