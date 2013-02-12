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
#include "algorithms.h"

#define _USE_MATH_DEFINES
#include <math.h>
using namespace std;

struct is_zero
{
    __host__ __device__
    bool operator()(double x)
    {
	if( x!=0)
        	return 0;
	else 
		return 1;

    }
};


struct is_neg 
{
    __host__ __device__
    bool operator()(int x)
    {
	if( x>=0)
        	return 0;
	else 
		return 1;

    }
};


void CUDAreduction(double* data,long long int* index, unsigned long int length, unsigned long int pointCount, unsigned int &nz, int *cols, int *rows, double *crs_data)
{
	
	
	
	dim3 dimGrid(1+(length)/512,1,1);
	dim3 dimBlock(512,1,1);

	thrust::device_ptr<double> vals_new(crs_data);
	thrust::device_ptr<long long int> keys(index);
	thrust::device_ptr<double>new_end;
	thrust::device_ptr<double> vals(data);
	
	long long int *index_new;
	cudaMalloc((void**)&index_new, length*sizeof(long long int));

	thrust::device_ptr<long long int> keys_new(index_new);
	thrust::sort_by_key(&keys[0], &keys[length], &vals[0]);
	cudaFree(index_new);


	//reduces consequtive vals with equal index and copys reduced data and index into new array
	//new_end=thrust::reduce_by_key(&keys[0], &keys[length], &vals[0], &keys_new[0], vals_new).second;

	//nz=new_end-&vals_new[0];
	//printf("length: %i\n",nz);

	//cudaFree(data);
	//cudaFree(index);
	
	//split<<<dimGrid, dimBlock>>>(index_new,cols, rows,pointCount, length);

	split<<<dimGrid, dimBlock>>>(index,cols, rows,pointCount, length);
	

}

int main()
{
	/*Simulation Variables*/
	int degree=1;	
	int elementsX=256;
	int elementsY=256;
	double sizeX=1.0;
	double sizeY=1.0;

	/*variables necessary for computation*/
	unsigned long int ElementCount=elementsX*elementsY;
	unsigned int lol=0;
	lol--;
	
	printf("Element Count: %u\n", lol);
	unsigned long int PointsPerElement=(degree+1)*(degree+1);
	int *elements=NULL;
	int *boundaryNodes=NULL;
	unsigned long int pointCount=(degree+1+(elementsX-1)*degree)*(degree+1+(elementsY-1)*degree)-1;
	double a=sizeX/elementsX;
	double b=sizeY/elementsY;
	unsigned int nz=ElementCount*PointsPerElement*PointsPerElement;	

	
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
	long long int *index;
	double	*M_device;
	double	*M_m_device;
	int		*boundaryNodes_device;
	double	*LoadVector_device;
	double	*crs_data;
	
	dim3 dimGrid(1+(elementsX*elementsY)/512,1,1);
	dim3 dimBlock(512,1,1);
	dim3 dimGridM(1,1,1);
	dim3 dimBlockM(degree+1,degree+1,1);
	dim3 dimBlockM_m(degree,degree,1);
	
	cudaMalloc((void**)&coo_values_device, ElementCount*PointsPerElement*PointsPerElement*sizeof(double));
	cudaMalloc((void**)&M_device, (degree+1)*(degree+1)*sizeof(double));
	cudaMalloc((void**)&M_m_device, degree*degree*sizeof(double));
	cudaMalloc((void**)&elements_device, ElementCount*PointsPerElement*sizeof(int));
	cudaMalloc((void**)&coo_row_device, ElementCount*PointsPerElement*PointsPerElement*sizeof(int));
	cudaMalloc((void**)&coo_col_device, ElementCount*PointsPerElement*PointsPerElement*sizeof(int));
	cudaMalloc((void**)&index, ElementCount*PointsPerElement*PointsPerElement*sizeof(long long int));
	cudaMalloc((void**)&crs_data, ElementCount*PointsPerElement*PointsPerElement*sizeof(double));

	
	//create triangulation for the simulation
	printf("create triangulation...");
	elements=createTriangulation(coordinatesX,coordinatesY,degree,elementsX,elementsY,sizeX,sizeY);
	printf("done\n");

	//determine boundary nodes
	printf("determine boundary nodes...");
	boundaryNodes=determineBorders(elementsX, elementsY, degree);	
	printf("done\n");

	//copy necessarry data to device	
	printf("copy triangulation to device...");
	cudaMemcpy(elements_device,elements, ElementCount*PointsPerElement*sizeof(int), cudaMemcpyHostToDevice);
	printf("done\n");	
	
	//precompute binomial coeffitients
	printf("compute binomial coeffitients...");
	BernBinomCoeff<<<dimGridM, dimBlockM>>>(M_device, degree);
	BernBinomCoeff<<<dimGridM, dimBlockM_m>>>(M_m_device, degree-1);
	printf("done\n");

	//compute system matrix
	printf("compute system matrix...");	
	ass_A_exact<<<dimGrid, dimBlock>>>(a,b,coo_row_device, coo_col_device, coo_values_device,degree,index, elements_device, M_device, M_m_device,elementsX, elementsY);
	printf("done\n");
	
	//convert coo output into crs format
	//sort the COO output in parallel, reduction is not workin yet
	CUDAreduction(coo_values_device,index,ElementCount*PointsPerElement*PointsPerElement, pointCount+1, nz, coo_col_device, coo_row_device, crs_data);

	//cudaMemcpy(csr_data_device,coo_values_device, (nz)*sizeof(double), cudaMemcpyDeviceToDevice);
	//cudaMemcpy(csr_col_device,coo_col_device, (nz)*sizeof(int), cudaMemcpyDeviceToDevice);


	//copy data back to host, to reduce on CPU
	double *coo_values = new double[ElementCount*PointsPerElement*PointsPerElement];
	int *coo_row = new int[ElementCount*PointsPerElement*PointsPerElement];
	int *coo_col = new int[ElementCount*PointsPerElement*PointsPerElement];
	long long int *indec = new long long int[ElementCount*PointsPerElement*PointsPerElement];


	cudaMemcpy(coo_values,coo_values_device	, ElementCount*PointsPerElement*PointsPerElement*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(coo_row,coo_row_device		, ElementCount*PointsPerElement*PointsPerElement*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(coo_col,coo_col_device		, ElementCount*PointsPerElement*PointsPerElement*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(indec,index					, ElementCount*PointsPerElement*PointsPerElement*sizeof(long long int), cudaMemcpyDeviceToHost);		




	



	
	//reduce values
	
	printf("reduce values...");
	int zeroEntries=reduceCOO(coo_values,coo_row,coo_col,PointsPerElement,ElementCount, nz);
	printf("done\n");

	nz=ElementCount*PointsPerElement*PointsPerElement-zeroEntries;	

	ofstream file("output normal.txt");

/*	for(int i=0;i<nz;i++)
		file << "val: " << coo_values[i] << "\n";

	file << "\n";

	for(int i=0;i<nz;i++)
		file << "col: " << coo_col[i] << "\n";
	
	file << "\n";

	for(int i=0;i<nz;i++)
		file << "row: " << coo_row[i] << "\n";

	for(int i=0;i<nz-1;i++)
		if(indec[i]>indec[i+1])
			printf("error!!!!!\n");*/


	file <<"nz: " << nz ;

	file.close();
	


	double *csr_data_device;
	int *csr_col_device;

	cudaFree(coo_values_device);
	cudaFree(coo_col_device);

	cudaMalloc((void**)&csr_col_device, (nz)*sizeof(int));
	cudaMalloc((void**)&csr_data_device, (nz)*sizeof(double));
	cudaMalloc((void**)&coo_row_device, (nz)*sizeof(int));
	

	//copy data back to device memory	
	cudaMemcpy(csr_data_device,coo_values, (nz)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(coo_row_device,coo_row, (nz)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(csr_col_device,coo_col, (nz)*sizeof(int), cudaMemcpyHostToDevice);
	
	int *csrRowPtr=0;		
	cudaMalloc((void**)&csrRowPtr, (pointCount+2)*sizeof(csrRowPtr[0]));	
	
	
	//initialize cusparse library
	printf("convert to crs format...");
	status=cusparseCreate(&handle);
	if(status!=CUSPARSE_STATUS_SUCCESS)
		cout << "CUSPARSE Library initialization failed" << endl;

	//convert to CSR format
	status=cusparseXcoo2csr(handle, coo_row_device,nz,pointCount+1,csrRowPtr,CUSPARSE_INDEX_BASE_ZERO);
	if(status!=CUSPARSE_STATUS_SUCCESS)
		cout << "Conversion from COO to CSR format failed" << endl;

	printf("done\n");
	
	int *row_index = new int[pointCount+2];
	int *col_index = new int[nz];
	cudaMemcpy(coo_values,csr_data_device,(nz)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(row_index,csrRowPtr,(pointCount+2)*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(col_index,csr_col_device,(nz)*sizeof(int), cudaMemcpyDeviceToHost);
	
	//assemble load vector
	//assuming for the time beeing f=0	

	dim3 dimGridL(1+((pointCount+1))/512,1,1);
	dim3 dimBlockL(512,1,1);
	
	printf("fill load vector...");

	cudaMalloc((void**)&LoadVector_device,(pointCount+1)*sizeof(double));
	fillArray<<<dimGridL,dimBlockL>>>(LoadVector_device, pointCount+1, 0.0);
	printf("done\n");
	
	
	dim3 dimGridK(1+(nz)/512,1,1);
	dim3 dimBlockK(512,1,1);
	cudaMalloc((void**)&boundaryNodes_device,(pointCount+1)*sizeof(int));
	cudaMemcpy(boundaryNodes_device,boundaryNodes,(pointCount+1)*sizeof(int),cudaMemcpyHostToDevice);

	

	
	//apply dirichlet boundary conditions /modify to matrix vector product 
	printf("apply dirichlet...");	
	applyDirichlet<<<dimGridK,dimBlockK>>>(LoadVector_device, csr_data_device,csr_col_device,csrRowPtr,boundaryNodes_device, nz, elementsX, elementsY, degree, 1.0);
	vectorDirichlet<<<dimGridK,dimBlockK>>>(LoadVector_device, csr_data_device,csr_col_device,csrRowPtr,boundaryNodes_device, nz, elementsX, elementsY, degree, 1.0);
	printf("done\n");
	//get info on device memory
	size_t freee;  
    size_t total;  
    cuMemGetInfo(&freee, &total);  
  
    cout << "free memory: " << freee / 1024 / 1024 << "mb, total memory: " << total / 1024 / 1024 << "mb" << endl;  
	
	//solve system of equations
	printf("solve equation...\n");
	double *x=CGsolve(csr_data_device,csr_col_device,csrRowPtr, LoadVector_device,nz,pointCount+1);
		printf("done\n");
	




	
	//free memory

	cudaFree(elements_device);
	cudaFree(M_device);
	cudaFree(M_m_device);
	free(coo_values);


	//write solution into file
	if(degree==1)
	{
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
	}
	else{

		stringstream fnAssembly;

		string Filename="";

		fnAssembly << "output_" << degree << "_" << elementsX <<"_" << elementsY<< ".txt";
		fnAssembly >> Filename;
		ofstream File(Filename);

		int vertexptr=0;
		int sideptr=(elementsX+1)*(elementsY+1);
		File << "{";
		
		
		for(int vlines=0; vlines < elementsY+1; vlines++){
			File << "{";
			for(int vperline=0; vperline<elementsX+1;vperline++){
				File << x[vertexptr++];
				if(vperline< elementsX){
					File << ",";			
					for(int sperline=0;sperline <degree-1;sperline++){
					File << x[sideptr++];
					File << ",";
					}
				}
			}
			File << "}";
			if(vlines <elementsY){
				File <<",";
				
			}
			if(vlines<elementsY){
			for(int sline=0; sline <degree-1;sline++){
				File << "{";
				for(int sperline=0; sperline<1+degree*elementsX;sperline++){
					File << x[sideptr++];
					if(sperline<degree*elementsX)
						File << ",";
				}
				File <<"},";
				
			}
			}
		}
		File << "}";
		File.close();
		
	}
		
	double test;
	cin >>test;
    return 0;
}


