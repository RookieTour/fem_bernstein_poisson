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
	printf("sort values...");
	thrust::sort_by_key(&keys[0], &keys[length], &vals[0]);
	printf("ok...");



	//reduces consequtive vals with equal index and copys reduced data and index into new array
	printf("reducing multiple entries...");
	new_end=thrust::reduce_by_key(&keys[0], &keys[length], &vals[0], &keys_new[0], vals_new).second;
	printf("ok...");
	nz=new_end-&vals_new[0];
	

	cudaFree(data);
	cudaFree(index);
	printf("split entries...");
	split<<<dimGrid, dimBlock>>>(index_new,cols, rows,pointCount, length);
	cudaFree(index_new);
	printf("ok...");	
}


void loadreduction(double *LoadList, double *LoadVector, int length, int *index)
{
	
	
	thrust::device_ptr<double> vals_new(LoadVector);
	thrust::device_ptr<int> keys(index);
	thrust::device_ptr<double>new_end;
	thrust::device_ptr<double> vals(LoadList);
	
	int *index_new;
	cudaMalloc((void**)&index_new, length*sizeof(int));

	thrust::device_ptr<int> keys_new(index_new);
	printf("sort values...");
	thrust::sort_by_key(&keys[0], &keys[length], &vals[0]);
	printf("ok...");



	//reduces consequtive vals with equal index and copys reduced data and index into new array
	printf("reducing multiple entries...");
	new_end=thrust::reduce_by_key(&keys[0], &keys[length], &vals[0], &keys_new[0], vals_new).second;
	printf("ok...");
	
	int nz=new_end-&vals_new[0];
	printf("length: %i\n",nz);

	cudaFree(LoadList);
	cudaFree(index);
	
	

}


int main()
{
	//*simulation variables*/
	int degree=2;	
	int elementsX=100;
	int elementsY=100;
	double sizeX=1.0;
	double sizeY=1.0;
	double f=1.0;

	/*variables necessary for computation*/
	unsigned long int ElementCount=elementsX*elementsY;

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
	int *LoadIndex_device;
	double *LoadVectorList;
	int *csrRowPtr=0;		

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
	cudaMalloc((void**)&LoadVectorList,ElementCount*PointsPerElement*sizeof(double));
	cudaMalloc((void**)&LoadIndex_device,ElementCount*PointsPerElement*sizeof(int));
	cudaMalloc((void**)&LoadVector_device,(pointCount+1)*sizeof(double));
	cudaMalloc((void**)&boundaryNodes_device,(pointCount+1)*sizeof(int));
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
	
	double time2=0.0, tstart;
	tstart = clock(); 
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
	//sort COO output in parallel, and reduce
	printf("reduce values...");	
	CUDAreduction(coo_values_device,index,ElementCount*PointsPerElement*PointsPerElement, pointCount+1, nz, coo_col_device, coo_row_device, crs_data);
	printf("done\n");
		
	
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
	cudaThreadSynchronize();
	time2 += clock() - tstart;

	time2 = (1000*time2)/CLOCKS_PER_SEC;

	printf("Konstruktions Zeit:  %e ms. \n",time2);
	
	//assemble load vector
	//assuming f constant

	dim3 dimGridL(1+((pointCount+1))/512,1,1);
	dim3 dimBlockL(512,1,1);
	
	printf("fill load vector...");	

	loadVector<<<dimGrid,dimBlock>>>(LoadVectorList, LoadIndex_device,elements_device,  a,b, degree, f, ElementCount);
	loadreduction(LoadVectorList,LoadVector_device,ElementCount*PointsPerElement,LoadIndex_device);
	printf("done\n");

	
	dim3 dimGridK(1+(nz)/512,1,1);
	dim3 dimBlockK(512,1,1);

	cudaMemcpy(boundaryNodes_device,boundaryNodes,(pointCount+1)*sizeof(int),cudaMemcpyHostToDevice);

	

	
	//apply dirichlet boundary conditions /modify to matrix vector product 
	printf("apply dirichlet...");	
	tstart = clock(); 
	applyDirichlet<<<dimGridK,dimBlockK>>>(LoadVector_device, crs_data,coo_col_device,csrRowPtr,boundaryNodes_device, nz, elementsX, elementsY, degree, 1.0);
	vectorDirichlet<<<dimGridK,dimBlockK>>>(LoadVector_device, crs_data,coo_col_device,csrRowPtr,boundaryNodes_device, nz, elementsX, elementsY, degree, 1.0);
	
	cudaThreadSynchronize();
	time2=0.0;
	time2 += clock() - tstart;

	time2 = (1000*time2)/CLOCKS_PER_SEC;

	printf("Dirichlet Zeit:  %e ms. \n",time2);
	printf("done\n");
	//get info on device memory
	size_t freee;  
    size_t total;  
    cuMemGetInfo(&freee, &total);  
  
    cout << "free memory: " << freee / 1024 / 1024 << "mb, total memory: " << total / 1024 / 1024 << "mb" << endl;  
	
	//solve system of equations
	printf("solve equation...\n");
	double *x=CGsolve(crs_data,coo_col_device,csrRowPtr, LoadVector_device,nz,pointCount+1);
		printf("done\n");

	
	//free memory

	cudaFree(elements_device);
	cudaFree(M_device);
	cudaFree(M_m_device);

	cudaFree(coo_values_device);
	cudaFree(coo_row_device);
	cudaFree(coo_col_device);
	cudaFree(index);
	cudaFree(boundaryNodes_device);
	cudaFree(LoadVector_device);
	cudaFree(crs_data);
	cudaFree(LoadIndex_device);
	cudaFree(LoadVectorList);
	cudaFree(csrRowPtr);	
	


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


