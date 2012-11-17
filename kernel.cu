#include<cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"




__global__ void compute_bernstein(double *bernstein_q0, double *dbernstein_q0, double *bernstein_q1, double *dbernstein_q1, int n)	
{
	double q1=0.7886751346;
	double q0=0.2113248654;
	int j;
	//evaluates bernstein polynomials for 2x2 Gaußian quadrature at poins q1=0.7886751346 and q0=0.2113248654;

	if (threadIdx.x==0)
	{		
		if (n==0)	
		{
			bernstein_q0[0] = 1.0;
		}
		else if (0<n)
		{
			bernstein_q0[0]=1.0-q0;
			bernstein_q0[1]=q0;
 
			for (int i=2;i<=n;i++)
			{
				bernstein_q0[i]=q0*bernstein_q0[i-1];
				for (j=i-1;1<=j;j--)
				{
					bernstein_q0[j]=q0*bernstein_q0[j-1]+(1.0-q0)*bernstein_q0[j];
				}
				bernstein_q0[0]=(1.0-q0)*bernstein_q0[0];
			}
		}
	}

	if (threadIdx.x==1)
	{		
		if (n==0)	
		{
			bernstein_q1[0] = 1.0;
		}
		else if (0<n)
		{
			bernstein_q1[0]=1.0-q1;
			bernstein_q1[1]=q1;
 
			for (int i=2;i<=n;i++)
			{
				bernstein_q1[i]=q1*bernstein_q1[i-1];
				for (j=i-1;1<=j;j--)
				{
					bernstein_q1[j]=q1*bernstein_q1[j-1]+(1.0-q1)*bernstein_q1[j];
				}
				bernstein_q0[0]=(1.0-q1)*bernstein_q1[0];
			}
		}
	}

	int m=n-1;

	if (threadIdx.x==2)
	{		
		if (m==0)	
		{
			dbernstein_q0[0] = 1.0;
		}
		else if (0<m)
		{
			dbernstein_q0[0]=1.0-q0;
			dbernstein_q0[1]=q0;
 
			for (int i=2;i<=m;i++)
			{
				dbernstein_q0[i]=q0*dbernstein_q0[i-1];
				for (j=i-1;1<=j;j--)
				{
					dbernstein_q0[j]=q0*dbernstein_q0[j-1]+(1.0-q0)*dbernstein_q0[j];
				}
				dbernstein_q0[0]=(1.0-q0)*dbernstein_q0[0];
			}
		}

		dbernstein_q0[n]=n*dbernstein_q0[n-1];
		for(int i=m;i>=2;i--)
		{
			dbernstein_q0[i]=n*(dbernstein_q0[i-1]-dbernstein_q0[i]);
		}
		dbernstein_q0[0]=-n*dbernstein_q0[0];
	}


	if (threadIdx.x==3)
	{		
		if (m==0)	
		{
			dbernstein_q1[0] = 1.0;
		}
		else if (0<m)
		{
			dbernstein_q1[0]=1.0-q1;
			dbernstein_q1[1]=q1;
 
			for (int i=2;i<=m;i++)
			{
				dbernstein_q1[i]=q1*dbernstein_q1[i-1];
				for (j=i-1;1<=j;j--)
				{
					dbernstein_q1[j]=q1*dbernstein_q1[j-1]+(1.0-q1)*dbernstein_q1[j];
				}
				dbernstein_q1[0]=(1.0-q1)*dbernstein_q1[0];
			}
		}

		dbernstein_q1[n]=n*dbernstein_q1[n-1];
		for(int i=m;i>=2;i--)
		{
			dbernstein_q1[i]=n*(dbernstein_q1[i-1]-dbernstein_q1[i]);
		}
		dbernstein_q1[0]=-n*dbernstein_q1[0];
	}

	

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
/*//computing the summands for 2x2 gaussian quadrature
double B00=(dX[i_sub*3]*q0+dX[1+i_sub*3]*q0+dX[2+i_sub*3])*(dX[j_sub*3]*q0+dX[1+j_sub*3]*q0+dX[2+j_sub*3])/(a*a)
+(dY[i_sub*3]*q0+dY[1+i_sub*3]*q0+dY[2+i_sub*3])*(dY[j_sub*3]*q0+dY[1+j_sub*3]*q0+dY[2+j_sub*3])/(b*b);

double B01=(dX[i_sub*3]*q0+dX[1+i_sub*3]*q1+dX[2+i_sub*3])*(dX[j_sub*3]*q0+dX[1+j_sub*3]*q1+dX[2+j_sub*3])/(a*a)
+ (dY[i_sub*3]*q0+dY[1+i_sub*3]*q1+dY[2+i_sub*3])*(dY[j_sub*3]*q0+dY[1+j_sub*3]*q1+dY[2+j_sub*3])/(b*b);

double B10=(dX[i_sub*3]*q1+dX[1+i_sub*3]*q0+dX[2+i_sub*3])*(dX[j_sub*3]*q1+dX[1+j_sub*3]*q0+dX[2+j_sub*3])/(a*a)
+ (dY[i_sub*3]*q1+dY[1+i_sub*3]*q0+dY[2+i_sub*3])*(dY[j_sub*3]*q1+dY[1+j_sub*3]*q0+dY[2+j_sub*3])/(b*b);

double B11=(dX[i_sub*3]*q1+dX[1+i_sub*3]*q1+dX[2+i_sub*3])*(dX[j_sub*3]*q1+dX[1+j_sub*3]*q1+dX[2+j_sub*3])/(a*a)
+ (dY[i_sub*3]*q1+dY[1+i_sub*3]*q1+dY[2+i_sub*3])*(dY[j_sub*3]*q1+dY[1+j_sub*3]*q1+dY[2+j_sub*3])/(b*b);
*/
		if(i_sub==j_sub)
			A_elem[i_sub+4*j_sub]=(a*a+b*b)/(3*a*b);
		
		if((i_sub==0 && j_sub==1) || (i_sub==1 && j_sub==0) ||(i_sub==2 && j_sub==3) || (i_sub==3 && j_sub==2))
				A_elem[i_sub+4*j_sub]=(a*a-2*b*b)/(6*a*b);
		if((i_sub==0 && j_sub==2) || (i_sub==2 && j_sub==0) ||(i_sub==3 && j_sub==1) || (i_sub==1 && j_sub==3))
					A_elem[i_sub+4*j_sub]=-(a*a+b*b)/(6*a*b);
		if((i_sub==2 && j_sub==1) || (i_sub==1 && j_sub==2) ||(i_sub==0 && j_sub==3) || (i_sub==3 && j_sub==0))
							A_elem[i_sub+4*j_sub]=(b*b-2*a*a)/(6*a*b);
	
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


__global__ void ass_A_exact(double* A_device, int* g_mapping_device, int* c_mapping, double a, double b, int n, int Nx, int Ny, int color, int shiftx, int shifty)
{
	__shared__ double B[16];
int iscolor;

	//global element indexes
int i= blockIdx.x;
int j= blockIdx.y;

int iswithinbounds=((i<Nx)&&(j<Ny));

//local element matrix indexes
int i_sub=threadIdx.x;
int j_sub=threadIdx.y;




iscolor=(color==c_mapping[i+Nx*j]);



if(iscolor){

	if(n==1)
	{
		if(i_sub==j_sub)
			B[i_sub+4*j_sub]=(a*a+b*b)/(3*a*b);		
		if((i_sub==0 && j_sub==1) || (i_sub==1 && j_sub==0) ||(i_sub==2 && j_sub==3) || (i_sub==3 && j_sub==2))
			B[i_sub+4*j_sub]=(a*a-2*b*b)/(6*a*b);
		if((i_sub==0 && j_sub==2) || (i_sub==2 && j_sub==0) ||(i_sub==3 && j_sub==1) || (i_sub==1 && j_sub==3))
			B[i_sub+4*j_sub]=-(a*a+b*b)/(6*a*b);
		if((i_sub==2 && j_sub==1) || (i_sub==1 && j_sub==2) ||(i_sub==0 && j_sub==3) || (i_sub==3 && j_sub==0))
			B[i_sub+4*j_sub]=(b*b-2*a*a)/(6*a*b);
	}
	else
	{
		
		
		if((i_sub==0) && (j_sub==0))
			B[i_sub+4*j_sub]=(a*a+b*b)*n*n/(a*b*(4*n*n-1)); //OK MATHEMATICA
		if((i_sub==1) && (j_sub==1))
			//B[i_sub+4*j_sub]=n*n*((n-1)/(2*n-3)*b*b+n/(2*n-1)*a*a)/(a*b*(8*n*n-2));
			B[i_sub+4*j_sub]=n*n*(a*a*n*(2*n-3)+b*b*(1-3*n+2*n*n))/(a*b*(1-2*n)*(1-2*n)*(2*n-3)*(1+2*n));
		if((i_sub==2) && (j_sub==2))
			B[i_sub+4*j_sub]=(a*a+b*b)*n*n*n*(n-1)/(a*b*(1-2*n)*(1-2*n)*(1+2*n));
		if((i_sub==3) && (j_sub==3))
			B[i_sub+4*j_sub]=n*n*(b*b*(2*n-3)*n+a*a*(1-3*n+2*n*n))/(a*b*(1-2*n)*(1-2*n)*(-3-4*n+4*n*n));

		if(((i_sub==0) && (j_sub==1)) || ((i_sub==1) && (j_sub==0)) )
			B[i_sub+4*j_sub]=(a*a-b*b)*n*n/(a*b*(8*n*n-2)); //OK MATHEMATICA
		if(((i_sub==0) && (j_sub==2)) || ((i_sub==2) && (j_sub==0)) )
			B[i_sub+4*j_sub]=-n*n*(a*a+b*b)/(a*b*(16*n*n-4));
		if(((i_sub==0) && (j_sub==3)) || ((i_sub==3) && (j_sub==0)) )
			B[i_sub+4*j_sub]=-(a*a-b*b)*n*n/(a*b*(8*n*n-2));

		if(((i_sub==1) && (j_sub==3)) || ((i_sub==3) && (j_sub==1)) )
			B[i_sub+4*j_sub]=-n*n*(a*a+b*b)/(a*b*(16*n*n-4));
		if(((i_sub==1) && (j_sub==2)) || ((i_sub==2) && (j_sub==1)) )
			B[i_sub+4*j_sub]=n*n*((3-2*n)*n*a*a+(1-3*n+2*n*n)*b*b)/(2*a*b*(1-2*n)*(1-2*n)*(-3-4*n+4*n*n));

		if(((i_sub==2) && (j_sub==3)) || ((i_sub==3) && (j_sub==2)) )
			B[i_sub+4*j_sub]=((1-3*n+2*n*n)*a*a+(3-2*n)*n*b*b)*n*n/(2*a*b*(1-2*n)*(1-2*n)*(-3-4*n+4*n*n));

		
	}
}
__syncthreads();

if(iscolor){
int k=g_mapping_device[i_sub+(i+j*Nx)*4];
int l=g_mapping_device[j_sub+(i+j*Nx)*4];

//write in respective entries of global matrix
A_device[l+k*(Nx+1)*(Ny+1)]+=B[i_sub+j_sub*4];
}

}