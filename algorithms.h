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


double* CGsolve(double *d_val, int* d_col, int* d_row, double* d_r, int nz, int N);

void printCSRMatrix(double *csr_values, int *csr_col, int *csr_row, int N);

void quickSort(double *arr,int *index_i, int *index_j, int left, int right);

void SortCOO(double *coo_values, int *coo_row, int *coo_col,int PointsPerElement, int ElementCount);

int reduceCOO(double *coo_values, int *coo_row, int *coo_col,int PointsPerElement, int ElementCount,unsigned long int length);

double func(double x, double y);

//works as intended (correct for n=1 , n=2 tested only for specific cases n>2 "seems" okay)
int* createTriangulation(double *coordinatesX, double *coordinatesY, int degree, int elementsX, int elementsY, double sizeX, double sizeY);

int* determineBorders(int elementsX, int elementsY, int degree);

double* assembleLoadVector(double a, double b, int degree, int *elements, int elementsX, int elementsY, double *nodes_x, double *nodes_y, int pointCount);

//works as intended maybe modify to overload and print int and doubles in one routine
void printMatrix(double* A, int n, int m);

//works as intended maybe modify to overload and print int and doubles in one routine
void printMatrix_int(int* A, int n, int m);

#endif