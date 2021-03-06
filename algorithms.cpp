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



int checkcublasStatus ( cublasStatus_t status, const char *msg ) 
{

    if ( status != CUBLAS_STATUS_SUCCESS ){
        fprintf (stderr, "!!!! CUBLAS %s ERROR \n", msg);
        return 1;
    }
    return 0;
}


int checkCusparseStatus ( cusparseStatus_t status, const char *msg )
{
    if ( status != CUSPARSE_STATUS_SUCCESS ){
        fprintf (stderr, "!!!! CUSPARSE %s ERROR \n", msg);
        return 1;
    }
    return 0;
}

double* CGsolve(double *d_val, int* d_col, int* d_row, double* d_r, int nz, int N)
{
	
	const int max_iter =10000;
	int k;
	double tol=1e-16;
	double *d_x, *d_p, *d_Ax;
	double a, b, na, r0, r1,dot;
	double *x =new double[N];

	for (int i = 0; i < N; i++){      
		x[i]=0;
	}

	double time1=0.0, tstart;

	/* Get handle to the CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);
    if ( checkcublasStatus (cublasStatus, "!!!! CUBLAS initialization error\n") )
		printf("EXIT_FAILURE");

    /* Get handle to the CUSPARSE context */
    cusparseHandle_t cusparseHandle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);
    if ( checkCusparseStatus (cusparseStatus, "!!!! CUSPARSE initialization error\n") ) 
		printf("EXIT_FAILURE");

    cusparseMatDescr_t descr = 0;
    cusparseStatus = cusparseCreateMatDescr(&descr); 
    if ( checkCusparseStatus (cusparseStatus, "!!!! CUSPARSE cusparseCreateMatDescr error\n") )
		printf("EXIT_FAILURE");

    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
    
    cudaMalloc((void**)&d_x, N*sizeof(double));     
    cudaMalloc((void**)&d_p, N*sizeof(double));
    cudaMalloc((void**)&d_Ax, N*sizeof(double));

    cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);

    double alpha = 1.0;
    double alpham1 = -1.0;
    double beta = 0.0;
	r0 = 0.;

    cusparseStatus= cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_x, &beta, d_Ax);	
	cublasDaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);	
    cublasStatus = cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
		
    k = 1;
	tstart = clock(); 
    while (r1 > tol*tol && k <= max_iter){
		if (k > 1){
			b = r1 / r0;
            cublasStatus = cublasDscal(cublasHandle, N, &b, d_p, 1);
            cublasStatus = cublasDaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);
        }
		else{
			cublasStatus = cublasDcopy(cublasHandle, N, d_r, 1, d_p, 1);
        }
        cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_p, &beta, d_Ax);

        cublasStatus = cublasDdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);
		a = r1 / dot;
		
        cublasStatus = cublasDaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);		
		na = -a;
        
		cublasStatus = cublasDaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);
        r0 = r1;
		
        cublasStatus = cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
        cudaThreadSynchronize();
        
        k++;
    }
	printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
	k--;
	cudaThreadSynchronize();
	time1 += clock() - tstart;
	time1 = (1000*time1)/CLOCKS_PER_SEC;
	printf("CG Zeit pro iteration:  %e ms. \n",time1/k);
    cudaMemcpy(x, d_x, N*sizeof(double), cudaMemcpyDeviceToHost);
 
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);

    cudaFree(d_col);
    cudaFree(d_row);
    cudaFree(d_val);
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ax);

	return x;
	}

void printCSRMatrix(double *csr_values, int *csr_col, int *csr_row, int N)
{

		int k=0;
		int m, n;
		m=N;
		n=N;
		cout.precision(3);
		string Filename="matrix.txt";
		
		ofstream File(Filename);

		for(int j=0;j<m;j++){			
			for(int i=0;i<n;i++){
				if((i==csr_col[k]) &&(csr_row[j]<=k)){
					cout << csr_values[k] << "|";
					if ((csr_values[k]==1) || (csr_values[k]==0))
						File << csr_values[k];
					else
						File << "2";
					k++;
				}
				else{
					cout << "0" << "|";
					File << "0";
				}
			}
			cout << endl;
			File << "|||\n";
		}
	File.close();
}

int* createTriangulation(double *coordinatesX, double *coordinatesY, int degree, int elementsX, int elementsY, double sizeX, double sizeY)
{
	int ElementCount=elementsX*elementsY;
	int PointsPerElement=(degree+1)*(degree+1);
	int VertexPoints=(elementsX+1)*(elementsY+1);

	int *elements= new int[ElementCount*PointsPerElement];
	
	for(int k=0; k<ElementCount;k++){
		//vertex nodes
		for(int i=0; i<2;i++)
			for(int j=0; j<2;j++){
				elements[k*PointsPerElement+i+j*(degree+1)]=k+k/elementsX+i+j*(elementsX+1);
				coordinatesX[k*PointsPerElement+i+j*(degree+1)]=k*sizeX/elementsX+i*sizeX/elementsX;
				coordinatesY[k*PointsPerElement+i+j*(degree+1)]=k*sizeY/elementsY+j*sizeY/elementsY;
			}

		//center nodes
		for(int i=0; i<degree-1;i++)
			for(int j=0; j<degree-1;j++){
				elements[k*PointsPerElement+i+2+(j+2)*(degree+1)]=VertexPoints+1+(degree-1)*elementsX
							+degree*k
								+(k/elementsX)*((degree*elementsX+1)*(degree-2)+(degree-1)*elementsX+1)
									+i+(degree*elementsX+1)*j;
				coordinatesX[k*PointsPerElement+i+2+(j+2)*(degree+1)]=0;//to be implemented
				coordinatesY[k*PointsPerElement+i+2+(j+2)*(degree+1)]=0;//to be implemented
			}

		//side nodes
		for(int i=0; i<degree-1;i++)
			for(int j=0; j<2;j++){
				elements[k*PointsPerElement+i+2+j*(degree+1)]=VertexPoints+k*(degree-1)
																+(k/elementsX)*((degree*elementsX+1)*(degree-1))
																+i+j*((degree*elementsX+1)*(degree-1)+(degree-1)*elementsX);
				
				coordinatesX[k*PointsPerElement+i+2+j*(degree+1)]=0;//to be implemented
				coordinatesY[k*PointsPerElement+i+2+j*(degree+1)]=0;//to be implemented
			}
			for(int i=0; i<2;i++)
				for(int j=0; j<degree-1;j++){
					elements[k*PointsPerElement+i+(j+2)*(degree+1)]=VertexPoints
																	+(degree-1)*elementsX+k*(degree)+(k/elementsX)*((degree*elementsX+1)*(degree-2)+(degree-1)*elementsX+1)
																	+i*degree+j*(degree*elementsX+1);

					coordinatesX[k*PointsPerElement+i+(j+2)*(degree+1)]=0;//to be implemented
					coordinatesY[k*PointsPerElement+i+(j+2)*(degree+1)]=0;//to be implemented
				}
	}
	return elements;
}

int* determineBorders(int elementsX, int elementsY, int degree)
{
	int pointCount=(degree+1+(elementsX-1)*degree)*(degree+1+(elementsY-1)*degree);
	
	int *boundaryNodes= new int[pointCount];

	for(int i=0; i<pointCount;i++)
		boundaryNodes[i]=0;

	/*			2		*/
	/*	-------------	*/
	/*	|			|	*/
	/* 	|			|	*/
	/*4 |			| 3	*/
	/*	|			|	*/
	/*	-------------	*/
	/*			1		*/



	//left border (3)
		//Vertex Nodes
			//i*(elementsX+1) (0<=i<=elementsY)
			for(int i=0; i<=elementsY;i++)
				boundaryNodes[i*(elementsX+1)]=3;
			
		//Side Nodes
			//(elementsX+1)*(elementsY+1) +elementsX*(degree-1)+i*(elementsX*(degree-1)+ elementsX+1)+ i/(degree-1)*(elementsX*(degree-1)) (0<=i<=(degree-1)*elementsY)
		if(degree>1){
			for(int i=0;i<=(degree-1)*elementsY;i++)
				boundaryNodes[(elementsX+1)*(elementsY+1) +elementsX*(degree-1)+i*(elementsX*(degree-1)+ elementsX+1)+ i/(degree-1)*(elementsX*(degree-1))]=3;			
		}
	//right border (4)
		//Vertex Nodes
			//i*(elementsX+1)+elementsX (0<=i<=elementsY)
			for(int i=0;i<=elementsY;i++)
				boundaryNodes[i*(elementsX+1)+elementsX]=4;
			
	//Side Nodes
			if(degree>1){
			//(elementsX+1)*(elementsY+1) +elementsX*(degree-1)+elementsX*degree+i*(elementsX*(degree-1)+ elementsX+1)+ i/(degree-1)*(elementsX*(degree-1)) (0<=i<=(degree-1)*elementsY)
				for(int i=0;i<(degree-1)*elementsY;i++)
					boundaryNodes[(elementsX+1)*(elementsY+1) +elementsX*(degree-1)+elementsX*degree+i*(elementsX*(degree-1)+ elementsX+1)+ i/(degree-1)*(elementsX*(degree-1))]=4;
			}

				//lower border (1)
		//Vertex Nodes
			//0 ... elementsX
			for(int i=0;i<=elementsX;i++)
				boundaryNodes[i]=1;
	
		//Side Nodes
			if(degree>1){
			//(elementsX+1)*(elementsY+1) ...(elementsX+1)*(elementsY+1) +elementsX*(degree-1)-1
				for(int i=(elementsX+1)*(elementsY+1);i<=(elementsX+1)*(elementsY+1) +elementsX*(degree-1)-1;i++)
					boundaryNodes[i]=1;
			}
	//upper border (2)
		//Vertex Nodes
			//(elementsX+1)*elementsY...(elementsX+1)*(elementsY+1)-1
			for(int i=(elementsX+1)*elementsY;i<=(elementsX+1)*(elementsY+1)-1;i++)
				boundaryNodes[i]=2;
		//Side Nodes
			if(degree>1){
				//pointCount -elementsX*(degree-1)...pointCount-1
			for(int i=pointCount -elementsX*(degree-1);i<=pointCount-1;i++)
				boundaryNodes[i]=2;
			}
			return boundaryNodes;
}

void printMatrix(double* A, int n, int m)
{
		
	string Filename="vektor.txt";		
	ofstream File(Filename);
	cout.precision(5);
	for(int j=0;j<m;j++){			
			for(int i=0;i<n;i++){			 
				cout << A[i+j*m]<< "|";
				File << A[i+j*m] << "\n";
			}
			cout << endl;			
		}		
}

void printMatrix_int(int* A, int n, int m)
{
	cout.precision(2);
	for(int j=0;j<m;j++){			
			for(int i=0;i<n;i++){			 
				cout << A[i+j*m]<< "|";
			}
			cout << endl;			
		}
		
}