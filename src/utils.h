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


bool loadMarket(string fileName,  SparseMatrix<double>&mat) {
	ifstream fin(fileName);
	long int M, N, L;
	if (!fin.is_open())
	{
		return false;
	}
	while (fin.peek()=='%')
	{
		fin.ignore(2048, '\n');
	}
	fin >> M >> N >> L;
	mat.resize(M,N);
	mat.reserve(2*L);
	vector<Eigen::Triplet<double>>triple;
	for (size_t i = 0; i < L; i++)
	{
		int m, n;
		double data;
		fin >> m >> n >> data;
		triple.push_back(Triplet<double>(m - 1, n - 1, data));
		triple.push_back(Triplet<double>(n - 1, m - 1, data));
	}
	fin.close();
	mat.setFromTriplets(triple.begin(), triple.end());
	return true;
}

void denseSPD(int n, MatrixXd& result) {
	result.resize(n, n);
	MatrixXd Q, D;
	Q.resize(n, n);
	D.resize(n, n);
	D.setZero();
	Q.setRandom();
	for (size_t i = 0; i < n; i++)
	{
		//The diagonal is a random number between 0 to 10
		D(i,i) = 10.0*double(rand())/ double(RAND_MAX);
		cout << D(i, i) << endl;
	}
	MatrixXd temp= Q.transpose() * D * Q;
	result = Q.transpose() * D * Q;
}


void saveData(string fileName, MatrixXd  matrix)
{
	//https://eigen.tuxfamily.org/dox/structEigen_1_1IOFormat.html
	const static IOFormat CSVFormat(FullPrecision, DontAlignCols, ", ", "\n");

	ofstream file(fileName);
	if (file.is_open())
	{
		file << matrix.format(CSVFormat);
		file.close();
	}
}

MatrixXd openData(string fileToOpen)
{

	// the inspiration for creating this function was drawn from here (I did NOT copy and paste the code)
	// https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix

	// the input is the file: "fileToOpen.csv":
	// a,b,c
	// d,e,f
	// This function converts input file data into the Eigen matrix format



	// the matrix entries are stored in this variable row-wise. For example if we have the matrix:
	// M=[a b c 
	//    d e f]
	// the entries are stored as matrixEntries=[a,b,c,d,e,f], that is the variable "matrixEntries" is a row vector
	// later on, this vector is mapped into the Eigen matrix format
	vector<double> matrixEntries;

	// in this object we store the data from the matrix
	ifstream matrixDataFile(fileToOpen);

	// this variable is used to store the row of the matrix that contains commas 
	string matrixRowString;

	// this variable is used to store the matrix entry;
	string matrixEntry;

	// this variable is used to track the number of rows
	int matrixRowNumber = 0;


	while (getline(matrixDataFile, matrixRowString)) // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
	{
		stringstream matrixRowStringStream(matrixRowString); //convert matrixRowString that is a string to a stream variable.

		while (getline(matrixRowStringStream, matrixEntry, ',')) // here we read pieces of the stream matrixRowStringStream until every comma, and store the resulting character into the matrixEntry
		{
			matrixEntries.push_back(stod(matrixEntry));   //here we convert the string to double and fill in the row vector storing all the matrix entries
		}
		matrixRowNumber++; //update the column numbers
	}

	// here we convet the vector variable into the matrix and return the resulting object, 
	// note that matrixEntries.data() is the pointer to the first memory location at which the entries of the vector matrixEntries are stored;
	return Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);

}


