#include<cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct coordinates{         /* deklariert den Strukturtyp person */
    int i;
	int j;

  } ;
__global__ void BernBinomCoeff(double *M, int n)
{
	 int i= threadIdx.x;
	 int j= threadIdx.y;

	unsigned int top_0=1;
	unsigned int top_1=1;
	unsigned int bottom=1;
	unsigned int n_save=n;
	//guarantees that every step in the solution is smaller than the final solution thus avoiding overflow
	for (int d=1; d <= i; d++)
	{
		top_0*= n_save--;
        top_0 /= d;
    }

	n_save=n;
	
	for (int d=1; d <= j; d++)
	{
		top_1*= n_save--;
        top_1 /= d;
    }
	n_save=2*n;

	for (int d=1; d <= i+j; d++)
	{
		bottom*= n_save--;
        bottom /= d;
    }
	
	
	M[i+j*(n+1)]=(double)(top_0*top_1)/bottom;

	
}

__global__ void ass_A(double* A_device, int* g_mapping_device, int* c_mapping, double a, double b, int n, int Nx, int Ny, int color, int shiftx, int shifty)
{
//gaussian quadrature points
double q1=0.7886751346;
double q0=0.2113248654;

int iscolor;

//matrix for derivative of the form function [dX*[ξ,η,1]^t]_i=dx φ(ξ,η)_i and [dY*[ξ,η,1]^t]_i=dy φ(ξ,η)_i
/*next step precumpute dericative matrices for Bernstein polynomial k<=3 and sum over for loop in kernel*/

double dX[12]={0 , 1 , -1 ,
0 ,-1 , 1 ,
0 , 1 , 0,
0 ,-1 , 0};

double dY[12]={1, 0,-1,
-1, 0, 0,
1, 0, 0,
-1, 0, 1};

//allocate memory for element stiffness matrix on shared memory -> fast access
__shared__ double A_elem[16];

//global element indexes
int i= blockIdx.x;
int j= blockIdx.y;



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

}
__syncthreads();

if(iscolor){
int k=g_mapping_device[i_sub+(i+j*Nx)*4];
int l=g_mapping_device[j_sub+(i+j*Nx)*4];

//write in respective entries of global matrix
A_device[l+k*(Nx+1)*(Ny+1)]+=A_elem[i_sub+j_sub*4];
}

}

__global__ void ass_A_exact(double a, double b, coordinates *coo_index, double*coo_value,int degree, int *elements, double *M, double *M_m)
{
	double *B;
	B=(double*)malloc((degree+1)*(degree+1)*(degree+1)*(degree+1)*sizeof(double));
	int i_glob;
	int j_glob;
	int shift;
	double sum=0;
	int element=threadIdx.x;
	int n=degree;
	


	 for (int i=0; i<=n;i++)
		for(int j=0; j<=n;j++)
			for (int k=0; k<=n;k++)
				for(int l=0; l<=n;l++)
				{
					if(n>1)
					{
						sum=M_m[i+(n-1)*j];

						if(i>0)
							sum-=M_m[i-1+(n-1)*j];
						if(j>0)
							sum-=M_m[i+(n-1)*(j-1)];
						if((j>0) && (i>0))
							sum+=M_m[i-1+(n-1)*(j-1)];
					}	
					B[i+j+n*n*(k+l)]=M[i+j*n]*b/a*(sum);		
					sum=0;
					if(n>1)
					{
						sum=M_m[k+(n-1)*l];
				
						if(k>0)
							sum-=M_m[k-1+(n-1)*l];
						if(l>0)
							sum-=M_m[k+(n-1)*(l-1)];
						if((k>0) && (l>0))
							sum+=M_m[k-1+(n-1)*(l-1)];
					}
				
					B[i+j+n*n*(k+l)]+=M[k+l*n]*a/b*(sum);
					B[i+j+n*n*(k+l)]*=n*n/(4*n*n-1);
					//start dumping values into coo list
						shift=i+j+n*n*(k+l);
						
						i_glob=elements[k*(n+1)*(n+1)+i+j];
						j_glob=elements[k*(n+1)*(n+1)+k+l];
						
						coo_index[element*(n+1)*(n+1)+shift].i=i_glob;
						coo_index[element*(n+1)*(n+1)+shift].j=j_glob;
						coo_value[element*(n+1)*(n+1)+shift]=B[i+j+n*n*(k+l)];
				}

		
	free(B);
}