//
//  Copied from IOfn.cpp
/*******************************************************
* Copyright (C) 2020 {Derek Driggs} <{dtd24@cam.ac.uk}>
*
* All rights reserved.
*******************************************************/
#include "stdafx_dd.h"
using namespace Eigen;
using namespace std;
#include <fstream>
#include <string>

using namespace std;
using namespace Eigen;

#define MAXBUFSIZE  ((int) 1e6)
extern int DATATYPE; // ==1 is the expected, ==0 is the binary. #include <iostream>


// Read in matrix from file into MatrixXd
MatrixXd readMatrix(const char *filename)
{
  int cols = 0, rows = 0;

  double buff[MAXBUFSIZE];


  // Read numbers from file into buffer.
  ifstream infile;
  infile.open(filename);

  while (! infile.eof())
    {
      string line;
      getline(infile, line);
      int temp_cols = 0;
      stringstream stream(line);
      while(! stream.eof())
          {stream >> buff[cols*rows+temp_cols++];
        }

      if (temp_cols == 0)
          continue;

      if (cols == 0)
          cols = temp_cols;

      rows++;
      }

  infile.close();

  rows--;

  cout << "\n";

  // Populate matrix with numbers.
  MatrixXd result(rows,cols);
  for (int i = 0; i < rows; i++){
      for (int j = 0; j < cols; j++){
          result(i,j) = buff[ cols*i+j ];
          cout << result(i,j) << "   ";
        }
      cout << "\n";
    }

  return result;
};


///////////////////////////////////////////
// copy and paste the following function to use it for reading
// sparse matrix binary and weighted case for reading G_XA, G_XB and G_XC
Eigen::SparseMatrix<double> read_G_sparse(char *file_name, char *G_name, int N1, int N2) // input: file name, adjacent matrix name, NA/NB/NC, output: sparse matrix
{
	printf("reading %s\n", G_name); fflush(stdout);
	Eigen::SparseMatrix<double> G_mat(N1, N2); // NX \times (NA or NB or NC)
	G_mat.makeCompressed();
	vector<Triplet<double> > triplets_sparse;
	double r_idx, c_idx; // row and column indices - matlab style
	double val;

	FILE* file_ptr = fopen(file_name, "r"); // opening G_name
	if (file_ptr == NULL) // exception handling if reading G_name fails
	{
		printf("reading adjacency submatrix failed\n"); fflush(stdout);
		exit(1);
	}
	while (!feof(file_ptr)) // reading G_name
	{
		fscanf(file_ptr, "%lf", &r_idx); // first read in row then col
		fscanf(file_ptr, "%lf", &c_idx);
		fscanf(file_ptr, "%lf", &val);
		if (DATATYPE == 1)
		{
			triplets_sparse.push_back(Triplet<double>(r_idx - 1, c_idx - 1, val));
		}
		else
			triplets_sparse.push_back(Triplet<double>(r_idx , c_idx , val));
	}
	fclose(file_ptr);
	G_mat.setFromTriplets(triplets_sparse.begin(), triplets_sparse.end());
	G_mat.prune(TOLERANCE);
	return G_mat;
}


int write_pi(char *filename, SparseMatrix<double> mat)
{
	fstream f(filename, ios::out);
	for (long k = 0; k<mat.outerSize(); ++k)
	for (SparseMatrix<double>::InnerIterator it(mat, k); it; ++it)
	{
		f << it.row() + 1 << "\t" << it.col() + 1 << "\t" << it.value() << endl;
	}
	f.close();
	return 0;
}

int write_alpha(char * filename, VectorXd vec){
	fstream f(filename, ios::out);
	for (int i = 0; i < vec.size(); i++){
		f << vec(i) << endl;
	}
	f.close();
	return 0;
}

int write_beta(char * filename, MatrixXd mat){
	fstream f(filename, ios::out);
	f << mat << endl;
	f.close();
	return 0;
}



void furongprintVector(double value[], long len, char *character) // print the elements of an array
{
	for (long i = 0; i<len; i++)
	{
		printf("%s=%.10f\n", character, value[i]);
	}
}
