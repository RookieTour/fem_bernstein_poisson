#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.cuh"
#include "cusparse.h"

#define _USE_MATH_DEFINES
#include <math.h>
using namespace std;



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

	/*			1		*/
	/*	-------------	*/
	/*	|			|	*/
	/* 	|			|	*/
	/*2 |			| 3	*/
	/*	|			|	*/
	/*	-------------	*/
	/*			0		*/

	//lower border (0)
		//Vertex Nodes
			//0 ... elementsX
		//Side Nodes
			//(elementsX+1)*(elementsY+1) ...(elementsX+1)*(elementsY+1) +elementsX*(degree-1)-1
		
	//upper border (1)
		//Vertex Nodes
			//(elementsX+1)*elementsY...(elementsX+1)*(elementsY+1)-1
		//Side Nodes
			//pointCount -elementsX*(degree-1)...pointCount-1

	//left border (2)
		//Vertex Nodes
			//i*(elementsX+1) (0<=i<=elementsY)
		//Side Nodes
			//(elementsX+1)*(elementsY+1) +elementsX*(degree-1)+i*(elementsX*(degree-1)+ elementsX+1)+ i/(degree-1)*(elementsX*(degree-1)) (0<=i<=(degree-1)*elementsY)

	//right border (3)
		//Vertex Nodes
			//i*(elementsX+1)+elementsX (0<=i<=elementsY)
	//Side Nodes
			//(elementsX+1)*(elementsY+1) +elementsX*(degree-1)+elementsX*degree+i*(elementsX*(degree-1)+ elementsX+1)+ i/(degree-1)*(elementsX*(degree-1)) (0<=i<=(degree-1)*elementsY)

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

//has yet to be tested
void CGmethodSolve(double* A, double*b,int degree, int elementsX, int elementsY, double *x)
{

	int node_count=(elementsX+1+(degree-1)*elementsX)*(elementsY+1+(degree-1)*elementsY);
	double* Ax =new double[node_count];
	double* Ap =new double[node_count];
	double* r =new double[node_count];
	double* p =new double[node_count];
	double sum, norm, alpha, beta, eps;
	int iter_max=node_count;

	eps=1.0e-15;

		//initialize starting vector
		for(int i=0;i<node_count;i++)
			x[i]=0.0;

	for(int j=0;j<node_count;j++){
		sum=0;
		for(int i=0;i<node_count;i++)
			sum+=A[i+j*node_count]*x[i];
		Ax[j]=sum;
	}

	for(int i=0; i<node_count;i++){
		r[i]=b[i]-Ax[i];
		p[i]=r[i];
	}

	for(int k=0;k<iter_max;k++){

		for(int j=0;j<node_count;j++){
			sum=0;
		for(int i=0;i<node_count;i++)
			sum+=A[i+j*node_count]*p[i];
		Ap[j]=sum;
	}
		sum=0;
		for(int i=0; i<node_count;i++)
			sum+=p[i]*Ap[i];
		norm=sum;

		sum=0;
		for(int i=0; i<node_count;i++)
			sum+=r[i]*p[i];

		alpha= sum/norm;
		for(int i=0; i<node_count;i++){
			x[i]=x[i]+alpha*p[i];
			r[i]=r[i]-alpha*Ap[i];
		}

		sum=0;
		for(int i=0; i<node_count;i++)
			sum+=r[i]*Ap[i];
		beta=sum/norm;

		for(int i=0; i<node_count;i++)
			p[i]=r[i]-beta*p[i];
		if(norm<eps)
			k=iter_max;
		//print Norm to console every iteration
		cout << k << " Norm: " << norm << endl;
	}
}

//has yet to be tested
void applyDirichlet(double* A, int degree, int elementsX, int elementsY,double g, double *load, double *nodes_x, double *nodes_y, double size_x, double size_y)
{
		int node_count=(elementsX+1+(degree-1)*elementsX)*(elementsY+1+(degree-1)*elementsY);

		for(int i=0;i<node_count;i++)
		if(		(nodes_x[i]==0) ||(nodes_x[i]==size_x) || (nodes_y[i]==0) || (nodes_y[i]==size_y))
		{
			for(int j=0;j<node_count; j++)
			{
				load[j]-=g*A[i+j*node_count];
				A[i+j*node_count]=0;
				A[j+i*node_count]=0;
			}
			A[i+i*node_count]=1;
			load[i]=g;
		}
		
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
	int elementsX=4;
	int elementsY=2;
	double sizeX=4.0;
	double sizeY=2.0;
	

	/*variables necessary for computation*/
	int ElementCount=elementsX*elementsY;
	int PointsPerElement=(degree+1)*(degree+1);
	int*elements=NULL;
	int pointCount=(degree+1+(elementsX-1)*degree)*(degree+1+(elementsY-1)*degree);
	cusparseHandle_t handle=0;
	cusparseStatus_t status;
	double *LoadVector=NULL;

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

	dim3 dimGrid(1,1,1);
	dim3 dimBlock(elementsX*elementsY,1,1);
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
	
	/*copy necessarry memory to device*/	
	cudaMemcpy(elements_device,elements, ElementCount*PointsPerElement*sizeof(int), cudaMemcpyHostToDevice);



	/*assemble system matrix*/
	double a=sizeX/elementsX;
	double b=sizeY/elementsY;
	BernBinomCoeff<<<dimGrid, dimBlockM>>>(M_device, degree);

	BernBinomCoeff<<<dimGrid, dimBlockM_m>>>(M_m_device, degree-1);
	

	ass_A_exact<<<dimGrid, dimBlock>>>(a,b,coo_row_device, coo_col_device, coo_values_device,degree, elements_device, M_device, M_m_device);

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

	//allocating necessary sparse dataset memory

	int *csrRowPtr=0;
	double *csr_data_device;
	cudaMalloc((void**)&csrRowPtr, (pointCount+1)*sizeof(csrRowPtr[0]));
	cudaMalloc((void**)&csr_data_device, (ElementCount*PointsPerElement*PointsPerElement-zeroEntries)*sizeof(double));
	cudaMalloc((void**)&coo_row_device, (ElementCount*PointsPerElement*PointsPerElement-zeroEntries)*sizeof(int));
	
	//copy data back to device memory
	cudaMemcpy(csr_data_device,coo_values, (ElementCount*PointsPerElement*PointsPerElement-zeroEntries)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(coo_row_device,coo_row, (ElementCount*PointsPerElement*PointsPerElement-zeroEntries)*sizeof(int), cudaMemcpyHostToDevice);

	/*initialize cusparse library*/
	status=cusparseCreate(&handle);
	if(status!=CUSPARSE_STATUS_SUCCESS)
		cout << "CUSPARSE Library initialization failed" << endl;

	//convert to CSR format
	status=cusparseXcoo2csr(handle, coo_row_device,ElementCount*PointsPerElement*PointsPerElement-zeroEntries,pointCount,csrRowPtr,CUSPARSE_INDEX_BASE_ZERO);
	if(status!=CUSPARSE_STATUS_SUCCESS)
		cout << "Conversion from COO to CSR format failed" << endl;

	/*assemble load vector*/
	//assuming for the time beeing f=0
	dim3 dimGridL(pointCount/256,1,1);
	dim3 dimBlockL(256,1,1);
	double* LoadVector_device;
	cudaMalloc((void**)&LoadVector_device,pointCount*sizeof(double));
	fillArray<<<dimGridL,dimBlockL>>>(LoadVector_device, pointCount, 0.0);
	//LoadVector = assembleLoadVector(a,b,degree, elements, elementsX, elementsY, coordinatesX, coordinatesY,pointCount);

	/*apply dirichlet boundary conditions*/

	//find upper boundary nodes, set them to v
	//find lower,left, right boundary nodes, set them to 0

	/*solve system of equations*/

	/*write solution into file*/


	
	

	

	for(int i=0; i<ElementCount*PointsPerElement*PointsPerElement; i++)
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
	double test;
	cin >>test;
    return 0;
}


