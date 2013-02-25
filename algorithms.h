#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cuda.h>
#include <time.h>   
#include "helper_functions.h"  // helper for shared functions common to CUDA SDK samples
#include "helper_cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.cuh"
//#include "cusparse.h"
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>


#define _USE_MATH_DEFINES
#include <math.h>
using namespace std;

#ifndef algorithms_h
#define algorithms_h

//solves sparse system (d_val)*x=d_r 
double* CGsolve(double *d_val, int* d_col, int* d_row, double* d_r, int nz, int N);

//prints CSR matrix
void printCSRMatrix(double *csr_values, int *csr_col, int *csr_row, int N);

//creates triangulation
int* createTriangulation(double *coordinatesX, double *coordinatesY, int degree, int elementsX, int elementsY, double sizeX, double sizeY);

//determines border nodes
int* determineBorders(int elementsX, int elementsY, int degree);


//print double matrix
void printMatrix(double* A, int n, int m);

//prints int matrix
void printMatrix_int(int* A, int n, int m);

#endif