#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cuda.h>
#include "helper_functions.h"  // helper for shared functions common to CUDA SDK samples
#include "helper_cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.cuh"
//#include "cusparse.h"
#include <cusparse_v2.h>
#include <cublas_v2.h>

#define _USE_MATH_DEFINES
#include <math.h>
using namespace std;

int checkcublasStatus ( cublasStatus_t status, const char *msg ) 
{
    if ( status != CUBLAS_STATUS_SUCCESS ) {
        fprintf (stderr, "!!!! CUBLAS %s ERROR \n", msg);
        return 1;
    }
    return 0;
}

/* checkCusparseStatus: concise method for verifying cusparse return status */
int checkCusparseStatus ( cusparseStatus_t status, const char *msg )
{
    if ( status != CUSPARSE_STATUS_SUCCESS ) {
        fprintf (stderr, "!!!! CUSPARSE %s ERROR \n", msg);
        return 1;
    }
    return 0;
}

double* CGsolve(double *d_val, int* d_col, int* d_row, double* d_r, int nz, int N){
	
	const int max_iter =10000;
	int k;
	double tol=1e-15;
	double *d_x, *d_p, *d_Ax;
	double a, b, na, r0, r1,dot;
	double *x =new double[N];

	 for (int i = 0; i < N; i++) {
      
        x[i] =0;
    }

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
    if ( checkCusparseStatus (cusparseStatus, "!!!! CUSPARSE initialization error\n") ) printf("EXIT_FAILURE");

    cusparseMatDescr_t descr = 0;
    cusparseStatus = cusparseCreateMatDescr(&descr); 
    if ( checkCusparseStatus (cusparseStatus, "!!!! CUSPARSE cusparseCreateMatDescr error\n") ) printf("EXIT_FAILURE");

    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
    
    checkCudaErrors( cudaMalloc((void**)&d_x, N*sizeof(double)) );  
    //checkCudaErrors( cudaMalloc((void**)&d_r, N*sizeof(double)) );
    checkCudaErrors( cudaMalloc((void**)&d_p, N*sizeof(double)) );
    checkCudaErrors( cudaMalloc((void**)&d_Ax, N*sizeof(double)) );



    cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);


    double alpha = 1.0;
    double alpham1 = -1.0;
    double beta = 0.0;
	r0 = 0.;

    cusparseStatus= cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_x, &beta, d_Ax);
	
	cublasDaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);

	
    cublasStatus = cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
	
	
    k = 1;
    while (r1 > tol*tol && k <= max_iter) {
        if (k > 1) {
            b = r1 / r0;
            cublasStatus = cublasDscal(cublasHandle, N, &b, d_p, 1);
            cublasStatus = cublasDaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);
        } else {
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
        printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }

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

void printCSRMatrix(double *csr_values, int *csr_col, int *csr_row, int N){
		int k=0;
		int m, n;
		m=N;
		n=N;
		cout.precision(3);
	for(int j=0;j<m;j++){			
			for(int i=0;i<n;i++){
				if((i==csr_col[k]) &&(csr_row[j]<=k)){
					cout << csr_values[k] << "|";
					k++;
				}

				else
					cout << "0" << "|";
			}
			cout << endl;			
		}

	}


void quickSort(double *arr,int *index_i, int *index_j, int left, int right) {
      int i = left, j = right;
      double tmp;
	  int itemp, jtemp;
      int pivot = index_i[(left + right) / 2];

 
      /* partition */
      while (i <= j) {
            while (index_i[i] < pivot)
                  i++;
            while (index_i[j] > pivot)
                  j--;
            if (i <= j) {
                  tmp = arr[i];
				  itemp = index_i[i];
				  jtemp = index_j[i];

				  index_i[i]=index_i[j];
				  index_j[i]=index_j[j];
                  arr[i] = arr[j];
				  index_i[j]=itemp;
				  index_j[j]=jtemp;
                  arr[j] = tmp;
                  i++;
                  j--;
            }
      };
 
      /* recursion */
      if (left < j)
            quickSort(arr,index_i, index_j, left, j);
      if (i < right)
            quickSort(arr,index_i, index_j, i, right);
}


void SortCOO(double *coo_values, int *coo_row, int *coo_col,int PointsPerElement, int ElementCount)
{
	quickSort(coo_values,coo_row,coo_col,0,PointsPerElement*PointsPerElement*ElementCount-1);
		int start=0;
		int end;
		int j=0;
		for(int i=0;i<ElementCount*PointsPerElement;i++)
		{
			while(coo_row[j]==i)
			{
				j++;
			}
			end=j-1;
			quickSort(coo_values,coo_col,coo_row,start,end);
			start=end+1;
		}

}
//works as intended


int reduceCOO(double *coo_values, int *coo_row, int *coo_col,int PointsPerElement, int ElementCount)
{
	int zeroEntries=0;
	int j;
	for(unsigned long int i=0; i<PointsPerElement*PointsPerElement*ElementCount;i++)
	{
		j=1;
		while((coo_row[i]==coo_row[i+j]) && (coo_col[i]==coo_col[i+j]) && (i+j<PointsPerElement*PointsPerElement*ElementCount))
		{
			coo_values[i]+=coo_values[i+j];
			coo_values[i+j]=0;
			zeroEntries++;
			j++;
		}
			
		i+=j-1;		
	}

	
	//put zero entries at the end
	
	for(unsigned long int i=0; i<PointsPerElement*PointsPerElement*ElementCount;i++)
	{
		if(coo_values[i]==0)
		{
			j=0;
			while((coo_values[i+j]==0) && (i+j<PointsPerElement*PointsPerElement*ElementCount))
				j++;

			if(i+j<PointsPerElement*PointsPerElement*ElementCount)
			{
				coo_values[i]=coo_values[i+j];
				coo_values[i+j]=0;
				coo_row[i]=coo_row[i+j];
				coo_col[i]=coo_col[i+j];
			}
			//look for next nz entry j
			//copy entry to i, set j to zero

		}
		
	}
	return zeroEntries;
}

void convertCOOtoCSR(double *coo_values, int *coo_row, int *coo_col,int PointsPerElement, int ElementCount,int zeroEntries)
{
	double* CSR_values = new double[ElementCount*PointsPerElement*PointsPerElement-zeroEntries];
	int* CSR_index_col = new int[ElementCount*PointsPerElement*PointsPerElement-zeroEntries];
	int* CSR_pointer_row = new int[ElementCount*PointsPerElement];

	int k=0;
	int i=0;
	int j=0;
	while((i<ElementCount*PointsPerElement*PointsPerElement-zeroEntries)&& (j<ElementCount*PointsPerElement*PointsPerElement))
	{
		if (coo_values[j]!=0)
		{
			CSR_values[i]=coo_values[j];
			CSR_index_col[i]=coo_col[j];
			
			if(coo_row[j]==k)
			{
				CSR_pointer_row[k]=coo_row[j];
				k++;
			}
			i++;
			j++;
		}
		else
			j++;

	}
}

double func(double x, double y)
{
	return -2*M_PI*M_PI*cos(2*M_PI*x)*sin(M_PI*y)*sin(M_PI*y)-2*M_PI*M_PI*cos(2*M_PI*y)*sin(M_PI*x)*sin(M_PI*x);
}

//works as intended (correct for n=1 , n=2 tested only for specific cases n>2 "seems" okay)
int* createTriangulation(double *coordinatesX, double *coordinatesY, int degree, int elementsX, int elementsY, double sizeX, double sizeY)
{
	int ElementCount=elementsX*elementsY;
	int PointsPerElement=(degree+1)*(degree+1);
	int VertexPoints=(elementsX+1)*(elementsY+1);

	int *elements= new int[ElementCount*PointsPerElement];
	
	for(int k=0; k<ElementCount;k++)
	{
		//vertex nodes
		for(int i=0; i<2;i++)
			for(int j=0; j<2;j++)
			{
				elements[k*PointsPerElement+i+j*(degree+1)]=k+k/elementsX+i+j*(elementsX+1);
				coordinatesX[k*PointsPerElement+i+j*(degree+1)]=k*sizeX/elementsX+i*sizeX/elementsX;
				coordinatesY[k*PointsPerElement+i+j*(degree+1)]=k*sizeY/elementsY+j*sizeY/elementsY;
			}

		//center nodes
		for(int i=0; i<degree-1;i++)
			for(int j=0; j<degree-1;j++)
			{
				elements[k*PointsPerElement+i+2+(j+2)*(degree+1)]=VertexPoints+1+(degree-1)*elementsX
							+degree*k
								+(k/elementsX)*((degree*elementsX+1)*(degree-2)+(degree-1)*elementsX+1)
									+i+(degree*elementsX+1)*j;
				coordinatesX[k*PointsPerElement+i+2+(j+2)*(degree+1)]=0;//to be implemented
				coordinatesY[k*PointsPerElement+i+2+(j+2)*(degree+1)]=0;//to be implemented
			}

		//side nodes
							for(int i=0; i<degree-1;i++)
								for(int j=0; j<2;j++)
								{
										elements[k*PointsPerElement+i+2+j*(degree+1)]=VertexPoints+k*(degree-1)
											+(k/elementsX)*((degree*elementsX+1)*(degree-1))
												+i+j*((degree*elementsX+1)*(degree-1)+(degree-1)*elementsX);

										coordinatesX[k*PointsPerElement+i+2+j*(degree+1)]=0;//to be implemented
										coordinatesY[k*PointsPerElement+i+2+j*(degree+1)]=0;//to be implemented
								}
						for(int i=0; i<2;i++)
								for(int j=0; j<degree-1;j++)
								{
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
//has yet to be tested
double* assembleLoadVector(double a, double b, int degree, int *elements, int elementsX, int elementsY, double *nodes_x, double *nodes_y, int pointCount)
{
	int ElementCount=(elementsX+1)*(elementsY+1);
	int PointsPerElement=(degree+1)*(degree+1);
	int m;
	double xc,yc;	
	double *load_sub;
	load_sub=new double[PointsPerElement];
	double *load= new double[pointCount];
	for(int k=0;k<ElementCount;k++)
	{

		//get x,y cordinate from the element k of point P_0
		//cout << "Element k=" << k << endl;
		for(int i=0;i<PointsPerElement;i++)
		{
			xc=nodes_x[elements[PointsPerElement*k+i]];
			yc=nodes_y[elements[PointsPerElement*k+i]];
			//cout <<"i= " <<i << " global: " << elements[4*k+i]<< "  x: " << xc << "  y: "<< yc << endl;
			load_sub[i]=a*b/((degree+1)*(degree+1))*func(xc,yc);			
		}
			for(int i=0;i<4;i++){
				m=elements[i+k*PointsPerElement];
				load[m]+=load_sub[i];
			}
		
	}
	return load;
}



//works as intended maybe modify to overload and print int and doubles in one routine
void printMatrix(double* A, int n, int m)
{
	cout.precision(5);
	for(int j=0;j<m;j++){			
			for(int i=0;i<n;i++){			 
				cout << A[i+j*m]<< "|";
			}
			cout << endl;			
		}
		
}

//works as intended maybe modify to overload and print int and doubles in one routine
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

void testMatrixSym(double* A, int n, int m)
{
	
	for(int j=0;j<m;j++)
	{			
		for(int i=0;i<n;i++)
		{			 
			if (A[i+j*n]!=A[j+i*n])
			{
				cout << "Nicht symmetrisch" << endl;
				return;
			}
		
		}
		
	}
	cout <<"Ist symmetrisch"<< endl;		
}


 
int main()
{
	/*Simulation Variables*/
	int degree=1;	
	int elementsX=10;
	int elementsY=10;
	double sizeX=1.0;
	double sizeY=1.0;
	

	/*variables necessary for computation*/
	int ElementCount=elementsX*elementsY;
	int PointsPerElement=(degree+1)*(degree+1);
	int *elements=NULL;
	int *boundaryNodes=NULL;
	int pointCount=(degree+1+(elementsX-1)*degree)*(degree+1+(elementsY-1)*degree)-1;
	
	cusparseHandle_t handle=0;
	cusparseStatus_t status;
	double *LoadVector= new double[pointCount+1];

	/*allocation of necessary host memory*/
	double *coordinatesX= new double[ElementCount*PointsPerElement];
	double *coordinatesY= new double[ElementCount*PointsPerElement];	

	/*allocation of necessary device memory*/
	double	*coo_values_device;
	int		*coo_row_device;
	int		*coo_col_device;
	int		*elements_device;
	double	*M_device;
	double	*M_m_device;


	dim3 dimGrid(1+(elementsX*elementsY)/256,1,1);
	dim3 dimBlock(256,1,1);
	dim3 dimGridM(1,1,1);
	dim3 dimBlockM(degree+1,degree+1,1);
	dim3 dimBlockM_m(degree,degree,1);
	
	cudaMalloc((void**)&coo_values_device, ElementCount*PointsPerElement*PointsPerElement*sizeof(double));
	cudaMalloc((void**)&M_device, (degree+1)*(degree+1)*sizeof(double));
	cudaMalloc((void**)&M_m_device, degree*degree*sizeof(double));
	cudaMalloc((void**)&elements_device, ElementCount*PointsPerElement*sizeof(int));
	cudaMalloc((void**)&coo_row_device, ElementCount*PointsPerElement*PointsPerElement*sizeof(int));
	cudaMalloc((void**)&coo_col_device, ElementCount*PointsPerElement*PointsPerElement*sizeof(int));
	
	
	/*create triangulation for the simulation*/
	elements=createTriangulation(coordinatesX,coordinatesY,degree,elementsX,elementsY,sizeX,sizeY);
	boundaryNodes=determineBorders(elementsX, elementsY, degree);
	printf("boundary  nodes \n");
	printMatrix_int(boundaryNodes,8,8);
	/*copy necessarry memory to device*/	
	cudaMemcpy(elements_device,elements, ElementCount*PointsPerElement*sizeof(int), cudaMemcpyHostToDevice);


	
	/*assemble system matrix*/
	double a=sizeX/elementsX;
	double b=sizeY/elementsY;
	BernBinomCoeff<<<dimGridM, dimBlockM>>>(M_device, degree);
	BernBinomCoeff<<<dimGridM, dimBlockM_m>>>(M_m_device, degree-1);
	

	ass_A_exact<<<dimGrid, dimBlock>>>(a,b,coo_row_device, coo_col_device, coo_values_device,degree, elements_device, M_device, M_m_device,elementsX, elementsY);

	/* convert coo output into crs format*/
	//copy COO dataset from device to host
	double *coo_values = new double[ElementCount*PointsPerElement*PointsPerElement];
	int *coo_row = new int[ElementCount*PointsPerElement*PointsPerElement];
	int *coo_col = new int[ElementCount*PointsPerElement*PointsPerElement];

	cudaMemcpy(coo_values,coo_values_device, ElementCount*PointsPerElement*PointsPerElement*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(coo_row,coo_row_device, ElementCount*PointsPerElement*PointsPerElement*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(coo_col,coo_col_device, ElementCount*PointsPerElement*PointsPerElement*sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(coo_values_device);
	cudaFree(coo_row_device);
	cudaFree(coo_col_device);

	

	SortCOO(coo_values, coo_row, coo_col,PointsPerElement,ElementCount);
	int zeroEntries=reduceCOO(coo_values,coo_row,coo_col,PointsPerElement,ElementCount);

	for(int i=0; i<ElementCount*PointsPerElement*PointsPerElement;i++)
		printf("i: %i, col: %i\n",i, coo_col[i]);
	//allocating necessary sparse dataset memory

	int *csrRowPtr=0;
	double *csr_data_device;
	int *csr_col_device;
	
	cudaMalloc((void**)&csr_col_device, (ElementCount*PointsPerElement*PointsPerElement-zeroEntries)*sizeof(int));
	cudaMalloc((void**)&csrRowPtr, (pointCount+2)*sizeof(csrRowPtr[0]));
	cudaMalloc((void**)&csr_data_device, (ElementCount*PointsPerElement*PointsPerElement-zeroEntries)*sizeof(double));
	cudaMalloc((void**)&coo_row_device, (ElementCount*PointsPerElement*PointsPerElement-zeroEntries)*sizeof(int));
	
	//copy data back to device memory
	cudaMemcpy(csr_data_device,coo_values, (ElementCount*PointsPerElement*PointsPerElement-zeroEntries)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(coo_row_device,coo_row, (ElementCount*PointsPerElement*PointsPerElement-zeroEntries)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(csr_col_device,coo_col, (ElementCount*PointsPerElement*PointsPerElement-zeroEntries)*sizeof(int), cudaMemcpyHostToDevice);

	/*initialize cusparse library*/
	status=cusparseCreate(&handle);
	if(status!=CUSPARSE_STATUS_SUCCESS)
		cout << "CUSPARSE Library initialization failed" << endl;

	//convert to CSR format
	status=cusparseXcoo2csr(handle, coo_row_device,ElementCount*PointsPerElement*PointsPerElement-zeroEntries,pointCount+1,csrRowPtr,CUSPARSE_INDEX_BASE_ZERO);
	if(status!=CUSPARSE_STATUS_SUCCESS)
		cout << "Conversion from COO to CSR format failed" << endl;
	// csr_data_device , csrRowPtr, csr_col_device
	/*assemble load vector*/
	//assuming for the time beeing f=0

	 

	dim3 dimGridL(1+((pointCount+1)*(pointCount+1))/256,1,1);
	dim3 dimBlockL(256,1,1);
	double* LoadVector_device;

	cudaMalloc((void**)&LoadVector_device,(pointCount+1)*sizeof(double));
	fillArray<<<dimGridL,dimBlockL>>>(LoadVector_device, pointCount+1, 0.0);
	

	
	/*apply dirichlet boundary conditions*/
	int *boundaryNodes_device;
	int nz=ElementCount*PointsPerElement*PointsPerElement-zeroEntries;


	cudaMalloc((void**)&boundaryNodes_device,(pointCount+1)*sizeof(int));
	cudaMemcpy(boundaryNodes_device,boundaryNodes,(pointCount+1)*sizeof(int),cudaMemcpyHostToDevice);

	applyDirichlet<<<dimGridL,dimBlockL>>>(LoadVector_device, csr_data_device,csr_col_device,csrRowPtr,boundaryNodes_device, ElementCount*PointsPerElement*PointsPerElement-zeroEntries, elementsX, elementsY, degree, 2.0);
	
	int *row_index = new int[pointCount+2];
	int *col_index = new int[nz];
	
	cudaMemcpy(coo_values,csr_data_device,(nz)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(row_index,csrRowPtr,(pointCount+2)*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(col_index,csr_col_device,(nz)*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(LoadVector,LoadVector_device,(pointCount+1)*sizeof(double), cudaMemcpyDeviceToHost);
	printf("nz: %i\n",nz);
	printf("pointCount: %i\n",pointCount+1);
	printf("values:\n");
	//printMatrix(coo_values,nz,1);
	//printCSRMatrix(coo_values,col_index,row_index,pointCount+1);
	//printf("row:\n");
	//printMatrix_int(row_index,pointCount+2,1);
	//printf("col:\n");
	//printMatrix_int(col_index,nz,1);
	printf("LoadVector:\n");
	//printMatrix(LoadVector,pointCount+1,1);
	//find upper boundary nodes, set them to v
	//find lower,left, right boundary nodes, set them to 0

	
	
	
	/*solve system of equations*/
	double *x=CGsolve(csr_data_device,csr_col_device,csrRowPtr, LoadVector_device,nz,pointCount+1);

	/*write solution into file*/


	


	

	/*for(int i=0; i<ElementCount*PointsPerElement*PointsPerElement; i++)
		cout << coo_row[i] << "  |  " << coo_col[i] << "  |  " << coo_values[i] << endl;
	
	
	
	/*free memory*/

	cudaFree(elements_device);
	cudaFree(M_device);
	cudaFree(M_m_device);
	free(coo_values);


	/*
	//for(int i=1;i<16;i++)
	//runBernsteinSecondDegree(i,i);
	//runBernsteinSecondDegree(40	,40);
	//elements=createTriangulation(degree,elementsX,elementsY,1.0,1.0);
	//__global__ void ass_A_exact(a,b, coordinates *coo_index, double*coo_value,int degree, double *elements, double *M, double *M_m, degree);
	for(int k=0;k<elementsX*elementsY;k++){
		printMatrix(&coordinatesX[k*(degree+1)*(degree+1)],degree+1,degree+1);
		cout << endl;
	}*/
	
	

	stringstream fnAssembly;

	string Filename="";

	fnAssembly << "output" << elementsX <<"_" << elementsY<< ".txt";
	fnAssembly >> Filename;
	ofstream File(Filename);
	File << "{";
	for(int j=0;j<elementsY+1;j++)
	{
		File << "{";
		for(int i=0;i<elementsX;i++)		
			File <<x[i+j*(elementsX+1)] << "," ;
		File <<x[elementsX+j*(elementsX+1)];
		File << "}";
		if(j!=elementsY)
			File << ",";		
	}
	File << "}";
	File.close();
	

	double test;
	cin >>test;
    return 0;
}


