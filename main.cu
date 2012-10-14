#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.cuh"


using namespace std;



__global__ void square_array(float *a, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx<N) a[idx] = a[idx] * a[idx];
}

class nodes {
	double x,y;	
public:
	double getx ();
	double gety ();

};

double nodes::getx(){
	return x;
}
double nodes::gety(){
	return y;
}



//evauates bernstein polynomial of n-th degree and returns the value of B^n_i(x) forall i may be used in future iterations for polynome degree >1
double* eval_bernstein(int n, double x){
	double *bernstein;
	int j;

	bernstein = new double[n+1];

	if (n==0)
	{
		bernstein[0] = 1.0;
	}
	else if (0<n)
	{
		bernstein[0]=1.0-x;
		bernstein[1]=x;
 
		for (int i=2;i<=n;i++)
		{
			bernstein[i]=x*bernstein[i-1];
			for (j=i-1;1<=j;j--)
			{
				bernstein[j]=x*bernstein[j-1]+(1.0-x)*bernstein[j];
			}
			bernstein[0]=(1.0-x)*bernstein[0];
		}
	}
	return bernstein;
}

int main()
{
	//solving //-Δu=f with f=const and u=g=const on borders
    
	
	//value of f
	double f=1.0;

	//value of dirichlet border condition
	double g=0.0;

	//size of the simulated domain
	double size_x=1.0,
	       size_y=1.0;
	
	int N_x=89, //element count x
		N_y=89; //element count y

	//total element count
	int elem_count=N_x*N_y;

	//total count of nodes
	int node_count=(N_x+1)*(N_y+1);

	//Finite Element System Matrix A
	double* A =new double[node_count*node_count];

	//right-hand side of b of the FEM equations (ignoring boundary conditions)
	double* b= new double[node_count];

	//colour mapping
	int* c_mapping= new int[elem_count];

	//degree of Bernstein polynom for quadrilateral elements n<=3 for 2x2 Gaussian quadrature
	int n=1;

	//initializing A and b
	for(int i=0;i<node_count;i++)
	{
		for(int j=0;j<node_count;j++)
		{
			 A[i+j*node_count] = 0.0;
		}
	}

	for(int i=0;i<node_count;i++)	
		b[i]=0.0;

	//create rectangular grid with quadrilateral elements
	double* nodes_x=NULL;
	double* nodes_y=NULL;
	int* elements=NULL;

	nodes_x= new double[node_count];
	nodes_y= new double[node_count];
	elements= new int[elem_count*4];

	for(int j=0;j<N_y+1;j++){
		for(int i=0;i<N_x+1;i++){
			nodes_x[j*(N_x+1)+i]=(j%2)*size_x+pow(-1.0,j)*(double)size_x*i/N_x;
			nodes_y[j*(N_x+1)+i]=(double)size_y*j/N_y;
		}
	}
	//example of ordering for a N_x=4 , N_y=2
	//(10)--(11)--(12)--(13)--(14)
	//  |     |     |     |     |
	//  |  7  |  6  |  5  |  4  |
	//  |     |     |     |     |
	// (9)---(8)---(7)---(6)---(5)
	//  |     |     |     |     |
	//  |  0  |  1  |  2  |  3  |
	//  |     |     |     |     |
	// (0)---(1)---(2)---(3)---(4)
	//
	for(int j=0;j<N_y;j++){
		for(int i=0;i<N_x;i++){
			elements[0+(i+j*N_x)*4]=i+j*(N_x+1)+j%2;
			elements[1+(i+j*N_x)*4]=elements[0+(i+j*N_x)*4]+pow(-1.0, j+2);
			elements[3+(i+j*N_x)*4]=-(i+1)+(j+2)*(N_x+1)-(j%2);
			elements[2+(i+j*N_x)*4]=elements[3+(i+j*N_x)*4]+pow(-1.0, j+1);
			
		}
	}
	
	//allocate necessary device pointers and memory

	//each element is given a Block of 4x4 threads to compute the
	//local stiffness matrix and its contribution to the global stiffness matrix
	

	//blocks per Grid
	dim3 dimGrid(N_x,N_y,1);

	//threads per Block
	dim3 dimBlock(4,4,1);
	

	double *A_device;
	double *b_device;
	int* g_mapping_device;
	int* c_mapping_device;

	//allocate respective memory for the global stiffness matrix on the GPU , the solution vector and the gobal node mapping

	cudaMalloc((void**)&A_device, node_count*node_count*sizeof(double));
	cudaMalloc((void**)&b_device,			 node_count*sizeof(double));
	cudaMalloc((void**)&g_mapping_device,     elem_count*4*sizeof(int));	
	cudaMalloc((void**)&c_mapping_device,       elem_count*sizeof(int));

	//compute colouring for 2d equidistant element nodes 
	/* next step: use greedy algorithm to compute colouring of more general triangulations*/
	
	for(int i=0; i<N_x; i++)
		for(int j=0; j<N_y; j++)
			c_mapping[i+j*N_x]=(i%2)+2*(j%2);

	

	//copy precomputed values to GPU e.g. zero values

	cudaMemcpy(A_device,A,sizeof(double)*node_count*node_count,cudaMemcpyHostToDevice);
	cudaMemcpy(g_mapping_device,elements,elem_count*4*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(c_mapping_device,c_mapping,elem_count*sizeof(int),cudaMemcpyHostToDevice);
	//get the size of the respektive element (in this cas element 0) assuming every element has the same size
	//otherwise the element sizes have to be copied to the gpu and x_ass_A hass to be launched including the sizes for every element it evaluates
	
	double el_x, el_y;//

	el_x=nodes_x[elements[1]]-nodes_x[elements[0]];
	el_y=nodes_y[elements[3]]-nodes_x[elements[0]];
	
	
	//construct A on GPU ussing mesh colouring to prevent race condition
	for(int color=0;color<4;color++)
		//for(int i=0;i<N_x/16.0;i++)
			//for(int j=0;j<N_y/16.0;j++)
		ass_A<<<dimGrid, dimBlock>>>(A_device, g_mapping_device, c_mapping_device, el_x,el_y, n, N_x,N_y, color,0,0);
		
	//copy assembled stiffness matrix back to host
	cudaMemcpy(A, A_device, sizeof(double)*node_count*node_count, cudaMemcpyDeviceToHost);
	
	//compute load vector using 1 point gaussian quadrature 
	/*has yet to be implemented in cuda*/
	
	double b_sub[4];
	for(int i=0;i<4;i++)
		b_sub[i]=0.25*f*el_x*el_y;
	int m;
	for(int k=0;k<elem_count;k++){
		for(int i=0;i<4;i++){
			m=elements[i+k*4];
			b[m]+=b_sub[i];

		}
	}
	

	
	
	
	//free device memory
	cudaFree(A_device);
	cudaFree(g_mapping_device);
	cudaFree(b_device);
    cudaFree(c_mapping_device);
		
	//apply dirichlet boundary conditions	
	
	for(int i=0;i<node_count;i++)
		if( (nodes_x[i]==0) ||(nodes_x[i]==size_x) || (nodes_y[i]==0) || (nodes_y[i]==size_y))
		{
			for(int j=0;j<node_count; j++)
			{
				b[j]-=g*A[i+j*node_count];
				A[i+j*node_count]=0;
				A[j+i*node_count]=0;
			}
			A[i+i*node_count]=1;
			b[i]=g;
		}
			
	//solve Ax=b with CG method
	/*next step apply cuda matrix vector product kernels to accelerate CG method*/

	//allocating necessary memory for CG-method
		
	double* x =new double[node_count];
	double* Ax =new double[node_count];
	double* Ap =new double[node_count];
	double* r =new double[node_count];
	double* p =new double[node_count];
	double sum, norm, alpha, beta, eps;
	int iter_max=(N_x+1)*(N_y+1);

	eps=1.0e-15;
		//initialize starting vector
		for(int i=0;i<node_count;i++)
			x[i]=1.0;

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

	//matrix output
	/*
	cout.precision(3);
	for(int j=0;j<node_count;j++){			
			for(int i=0;i<node_count;i++){			 
				cout << A[i+j*node_count]<< "|";
			}
			cout << endl;			
		}
	*/

	//create file output of the solution vector
	
	ofstream File("output.txt");
	for(int j=0;j<N_y+1;j++){
	File << "{";
	for(int i=0;i<N_x;i++)		
			File <<x[i+j*(N_x+1)] << "," ;
	File <<x[N_x+j*(N_x+1)];

	File << "},";
}
	File.close();

	//CODE
	double test;
	cin >> test ;
	free(A);
	free(b);
	free(nodes_x);
	free(nodes_y);
	free(elements);
	free(x);
	free(Ax);
	free(Ap);
	free(r);
	free(p);
	free(c_mapping);
    return 0;
}
