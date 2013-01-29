#include<cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <iostream>
#include <sstream>

__global__ void fillArray(double* array, int size, double value)
{ 
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	
	if(i<size){		
		array[i]=value;
	}
	
		

}


__global__ void applyDirichlet(double* load, double* csr_matrix, int* csr_col_device,int* csrRowPtr,int *isboundaryNode, int entries, int elementsX, int elementsY, int degree, double boundaryValue)
{ 
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	int row=0;
	int col;
	int ucindex;
	int pointCount=(degree+1+(elementsX-1)*degree)*(degree+1+(elementsY-1)*degree)-1;
	double sum=0;
	double bound[4];
	bound[0]=1;
	bound[1]=0;
	bound[2]=0;
	bound[3]=0;
	//if (i>400)
		//printf("i_device: %i\n",i);
	if(i<pointCount+1){
		int start=csrRowPtr[i];
		int end =csrRowPtr[i+1];

		for(int j=start;j<end;j++)
		{	
			if(isboundaryNode[csr_col_device[j]]!=0)
			{					
				
				sum+=bound[isboundaryNode[csr_col_device[j]]-1]*csr_matrix[j];
			}
				
		}
		load[i]-=sum;

	}
	__syncthreads();		
	if(i<entries)
	{
		
		col=csr_col_device[i]; //spalte der großen matrix aber
		//printf("i: %i, isborder: %i  col: %i \n ", i,isboundaryNode[col],col);
		
		while((row<=pointCount) && (csrRowPtr[row]<=i))
		{		
			row++;
		}
		row--;
		
		
		//__syncthreads();		
			
		if((isboundaryNode[col]!=0)||(isboundaryNode[row]!=0)){			
			if(col!=row){				
				csr_matrix[i]=0;

			}
		}

		__syncthreads();		
		if((isboundaryNode[col]!=0)||(isboundaryNode[row]!=0)){
				if(col==row){
				//printf(" col : %i\n", col);
				csr_matrix[i]=1;	
				//load[col]=bound[isboundaryNode[col]-1];
				//printf(" Dirichlet auf : %i \n", col);
			}
			
			
		}

	// für jeden index i,jh in csr schaue ob i=j und i boundaryNode[i]=1 dann csr[i][j]=1 sonst wenn i!=j und i in boundary node dann csr[i][]=0
	}

		
}


__global__ void vectorDirichlet(double* load, double* csr_matrix, int* csr_col_device,int* csrRowPtr,int *isboundaryNode, int entries, int elementsX, int elementsY, int degree, double boundaryValue)
{ 
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	int row=0;
	int col;
	int ucindex;
	int pointCount=(degree+1+(elementsX-1)*degree)*(degree+1+(elementsY-1)*degree)-1;
	double sum=0;
	double bound[4];
	bound[0]=1;
	bound[1]=0;
	bound[2]=0;
	bound[3]=0;
		
	if(i<entries)
	{
		
		col=csr_col_device[i]; //spalte der großen matrix aber

		
		while((row<=pointCount) && (csrRowPtr[row]<=i))
		{		
			row++;
		}
		row--;
		
	
		if((isboundaryNode[col]!=0)||(isboundaryNode[row]!=0)){
				if(col==row){
					
				load[col]=bound[isboundaryNode[col]-1];
			
			}
			
			
		}

	// für jeden index i,jh in csr schaue ob i=j und i boundaryNode[i]=1 dann csr[i][j]=1 sonst wenn i!=j und i in boundary node dann csr[i][]=0
	}

		
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

__global__ void ass_A_exact(double a, double b,int *coo_row_device,int *coo_col_device, double*coo_value,int degree, int *elements, double *M, double *M_m, int elementsX, int elementsY)
{
	unsigned long int pointCount=(degree+1+(elementsX-1)*degree)*(degree+1+(elementsY-1)*degree);
	double *B;
	B=(double*)malloc((degree+1)*(degree+1)*(degree+1)*(degree+1)*sizeof(double));
	int i_glob;
	int j_glob;
	int shift;
	double sum=0;
	int element=threadIdx.x+blockIdx.x*blockDim.x;
	int n=degree;
	

	if(element<elementsX*elementsY)
	{
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

					//if(((i+j+k==0) && (l<2)) || ((i+j+l==0) && (j<2)))
						B[shift]*=(double)(n*n)/(4*n*n-1);
					//else
						//B[shift]*=(double)(n*n)/(degree*(4*n*n-1));
					//start dumping values into coo list
			
						
						
				}
				for(int i=0; i<(n+1)*(n+1);i++)
				{
					for(int j=0; j<(n+1)*(n+1);j++)
					{
						i_glob=elements[element*(n+1)*(n+1)+i];
						j_glob=elements[element*(n+1)*(n+1)+j];
						
						coo_row_device[element*(n+1)*(n+1)*(n+1)*(n+1)+i+j*(n+1)*(n+1)]=i_glob;
						coo_col_device[element*(n+1)*(n+1)*(n+1)*(n+1)+i+j*(n+1)*(n+1)]=j_glob;
						
						//coo_row_device[element*(n+1)*(n+1)*(n+1)*(n+1)+i+j*(n+1)*(n+1)]=i_glob*pointCount+j_glob;
						coo_value[element*(n+1)*(n+1)*(n+1)*(n+1)+i+j*(n+1)*(n+1)]=B[i+j*(n+1)*(n+1)];
					}
				}

	}	
	free(B);
	
		
}

__global__ void reduce(double* data, int* index, int length)
{
	
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	int j=0;
	if((index[i]==index[i+1]) && (index[i]!=index[i-1]))
		while(index[i]==index[i+j+1])
		{
			data[i]+=data[i+j+1];
			data[i+j+1]=0;
			index[i+j+1]=-1;
			j++;
		}



}

__global__ void split(int *index, int*cols, int*rows,int pointCount,int length)
{

	int i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i<length)
	{
		rows[i]=index[i]/pointCount;
		
		
	}
	__syncthreads();
	if(i<length)
	{
		cols[i]=index[i]%pointCount;
		
	}
}