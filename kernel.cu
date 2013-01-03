#include<cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void fillArray(double* array, int size, double value)
{ 
	int i=threadIdx.x+blockIdx.x*blockDim.x;

	if(i<size)
		array[i]=value;

}

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

__global__ void ass_A_exact(double a, double b,int *coo_row_device,int *coo_col_device, double*coo_value,int degree, int *elements, double *M, double *M_m)
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
					sum=0;
					shift=i+j*(degree+1)+(degree+1)*(degree+1)*(k+l*(degree+1));
					
					if((i<n) && (k<n))
						sum+=M_m[i+n*k];
					
						if((i>0) && (i-1<n) && (k<n))
							sum-=M_m[i-1+n*k];
						if((k>0)&& (i<n) && (k-1<n))
							sum-=M_m[i+n*(k-1)];
						if((k>0) && (i>0) && (i-1<n)&& (k-1<n))
							sum+=M_m[i-1+n*(k-1)];
					

					B[shift]=M[j+l*(n+1)]*b/a*sum;
					sum=0;
					if((j<n) && (l<n))
						sum=M_m[j+n*l];
				
						if((j>0) && (j-1<n) && (l<n))
							sum-=M_m[j-1+n*l];
						if((l>0)&& (j<n) && (l-1<n))
							sum-=M_m[j+n*(l-1)];
						if((l>0) && (j>0) && (j-1<n)&& (l-1<n))
							sum+=M_m[j-1+n*(l-1)];
				
				
					B[shift]+=M[i+k*(n+1)]*a/b*(sum);
			
					B[shift]*=(double)(n*n)/(4*n*n-1);
					
					//start dumping values into coo list
			
						//wrong mapping!!!
						
				}
				for(int i=0; i<(n+1)*(n+1);i++)
				{
					for(int j=0; j<(n+1)*(n+1);j++)
					{
						i_glob=elements[element*(n+1)*(n+1)+i];
						j_glob=elements[element*(n+1)*(n+1)+j];
						
						coo_row_device[element*(n+1)*(n+1)*(n+1)*(n+1)+i+j*(n+1)*(n+1)]=i_glob;
						coo_col_device[element*(n+1)*(n+1)*(n+1)*(n+1)+i+j*(n+1)*(n+1)]=j_glob;
						coo_value[element*(n+1)*(n+1)*(n+1)*(n+1)+i+j*(n+1)*(n+1)]=B[+i+j*(n+1)*(n+1)];
					}
				}

		
	free(B);
}