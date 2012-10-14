#include<cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void ass_A(double* A_device, int* g_mapping_device, int* c_mapping, double a, double b, int n, int Nx, int Ny, int color, int shiftx, int shifty)
{
	//gaussian quadrature points
	double q1=0.7886751346;
	double q0=0.2113248654;

	int iscolor;

	//matrix for derivative of the form function [dX*[ξ,η,1]^t]_i=dx φ(ξ,η)_i and [dY*[ξ,η,1]^t]_i=dy φ(ξ,η)_i
	/*next step precumpute dericative matrices for Bernstein polynomial k<=3 and sum over for loop in kernel*/

	double dX[12]={0 , 1 , -1 ,
			       0 ,-1 ,  1 ,
			       0 , 1 ,  0,
			       0 ,-1 ,  0};

	double dY[12]={1, 0,-1,
			      -1, 0, 0,
			       1, 0, 0,
			      -1, 0, 1};

	//allocate memory for element stiffness matrix on shared memory -> fast access
	__shared__ double A_elem[16];

	//global element indexes
	int i= blockIdx.x;
	int j= blockIdx.y;

	int iswithinbounds=((i<Nx)&&(j<Ny));

	//local element matrix indexes
	int i_sub=threadIdx.x;
	int j_sub=threadIdx.y;

	

	
	iscolor=(color==c_mapping[i+Nx*j]);
	

	
	if(iscolor){
	//computing the summands for 2x2 gaussian quadrature
	double B00=(dX[i_sub*3]*q0+dX[1+i_sub*3]*q0+dX[2+i_sub*3])*(dX[j_sub*3]*q0+dX[1+j_sub*3]*q0+dX[2+j_sub*3])/(a*a)
			  +(dY[i_sub*3]*q0+dY[1+i_sub*3]*q0+dY[2+i_sub*3])*(dY[j_sub*3]*q0+dY[1+j_sub*3]*q0+dY[2+j_sub*3])/(b*b);

	double B01=(dX[i_sub*3]*q0+dX[1+i_sub*3]*q1+dX[2+i_sub*3])*(dX[j_sub*3]*q0+dX[1+j_sub*3]*q1+dX[2+j_sub*3])/(a*a)
			  + (dY[i_sub*3]*q0+dY[1+i_sub*3]*q1+dY[2+i_sub*3])*(dY[j_sub*3]*q0+dY[1+j_sub*3]*q1+dY[2+j_sub*3])/(b*b);

	double B10=(dX[i_sub*3]*q1+dX[1+i_sub*3]*q0+dX[2+i_sub*3])*(dX[j_sub*3]*q1+dX[1+j_sub*3]*q0+dX[2+j_sub*3])/(a*a)
			  + (dY[i_sub*3]*q1+dY[1+i_sub*3]*q0+dY[2+i_sub*3])*(dY[j_sub*3]*q1+dY[1+j_sub*3]*q0+dY[2+j_sub*3])/(b*b);

	double B11=(dX[i_sub*3]*q1+dX[1+i_sub*3]*q1+dX[2+i_sub*3])*(dX[j_sub*3]*q1+dX[1+j_sub*3]*q1+dX[2+j_sub*3])/(a*a)
			  + (dY[i_sub*3]*q1+dY[1+i_sub*3]*q1+dY[2+i_sub*3])*(dY[j_sub*3]*q1+dY[1+j_sub*3]*q1+dY[2+j_sub*3])/(b*b);

	A_elem[i_sub+j_sub*4]=a*b*(B00+B01+B10+B11)/4;

	}
	__syncthreads();

	if(iscolor){
	int k=g_mapping_device[i_sub+(i+j*Nx)*4];
	int l=g_mapping_device[j_sub+(i+j*Nx)*4];

	//write in respective entries of global matrix
	A_device[l+k*(Nx+1)*(Ny+1)]+=A_elem[i_sub+j_sub*4];
	}

}


__global__ void VectorAdd(double* X, double* Y, double* Z, double alpha, int n)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;

	if (i < n) 
		Z[i]=X[i]+alpha*Y[i];
}
