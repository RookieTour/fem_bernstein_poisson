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

#define _USE_MATH_DEFINES
#include <math.h>
using namespace std;

//works as intended
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
				coordinatesX[k*PointsPerElement+i+j*(degree+1)]=k*sizeX/elementsX+i*sizeX/elementsX;//to be implemented
				coordinatesY[k*PointsPerElement+i+j*(degree+1)]=k*sizeY/elementsY+j*sizeY/elementsY;//to be implemented
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

//has yet to be tested
void assembleLoadVector(int degree, int *elements, int elementsX, int elementsY, double *load, double *nodes_x, double *nodes_y)
{
	int ElementCount=(elementsX+1)*(elementsY+1);
	int PointsPerElement=(degree+1)*(degree+1);
	int m;
	double xc,yc;
	double a,b;
	double *load_sub;
	load_sub=new double[PointsPerElement];

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


void runBernsteinSecondDegree(int n_x, int n_y){
	//solving //-Δu=f with f=const and u=g=const on borders
    
	
	//value of f
	//double f=1.0;

	//value of dirichlet border condition
	double g=0.0;

	//size of the simulated domain
	double size_x=1,
	       size_y=1;
	
	int N_x=n_x, //element count x
		N_y=n_y; //element count y

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
	int n=2;

	

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
	
	//stores values of bernsteinpolynomials of degree n
	double *bernstein_q0;
	double *bernstein_q1;
	double *dbernstein_q0;
	double *dbernstein_q1;
	double *A_device;
	double *b_device;
	int* g_mapping_device;
	int* c_mapping_device;

	unsigned int freem, t;
	cudaMemGetInfo(&freem, &t);
	cout << "Free Memory:" << freem << endl; 

	//allocate respective memory for the global stiffness matrix on the GPU , the solution vector and the gobal node mapping
	if ( cudaSuccess != cudaMalloc((void**)&A_device, node_count*node_count*sizeof(double)) )
	    printf( "Error not enough memory!\n" );
	cudaMalloc((void**)&A_device, node_count*node_count*sizeof(double));
	cudaMalloc((void**)&b_device,			 node_count*sizeof(double));
	cudaMalloc((void**)&g_mapping_device,     elem_count*4*sizeof(int));	
	cudaMalloc((void**)&c_mapping_device,       elem_count*sizeof(int));
	cudaMalloc((void**)&bernstein_q0,			 (n+1)*sizeof(double));
	cudaMalloc((void**)&bernstein_q1,			 (n+1)*sizeof(double));
	cudaMalloc((void**)&dbernstein_q0,			 (n+1)*sizeof(double));
	cudaMalloc((void**)&dbernstein_q1,			 (n+1)*sizeof(double));
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
	
	//precompute values of bernstein polynomials and its derivate
	//compute_bernstein<<<1, 4>>>(bernstein_q0,dbernstein_q0, bernstein_q1, dbernstein_q1, n);
	//construct A on GPU ussing mesh colouring to prevent race condition
	for(int color=0;color<4;color++)
		//for(int i=0;i<N_x/16.0;i++)
			//for(int j=0;j<N_y/16.0;j++)
//		ass_A_exact<<<dimGrid, dimBlock>>>(A_device, g_mapping_device, c_mapping_device, el_x,el_y, n, N_x,N_y, color,0,0);
		
	//copy assembled stiffness matrix back to host
	cudaMemcpy(A, A_device, sizeof(double)*node_count*node_count, cudaMemcpyDeviceToHost);
	
	//compute load vector using 1 point gaussian quadrature 
	/*has yet to be implemented in cuda*/
	
	double b_sub[4];
	double xc,yc;
	int m;
	for(int k=0;k<elem_count;k++){
		//get x,y cordinate from the element k of point P_0
		//cout << "Element k=" << k << endl;
		for(int i=0;i<4;i++)
		{
			xc=nodes_x[elements[4*k+i]];
			yc=nodes_y[elements[4*k+i]];
			//cout <<"i= " <<i << " global: " << elements[4*k+i]<< "  x: " << xc << "  y: "<< yc << endl;
			//b_sub[i]=el_x*el_y/((n+1)*(n+1))*func(xc,yc);
			b_sub[i]=el_x*el_y/(4)*func(xc,yc);
			//b_sub[i]=el_x*el_y/((n+1)*(n+1))*f;
			
		}
			for(int i=0;i<4;i++){
				m=elements[i+k*4];
				b[m]+=b_sub[i];
			}
		
	}

	for(int i=0;i<node_count;i++	)
		cout << b[i] << endl;

	cout <<"----------------------" << endl;

	//old
	/*
	cout << n << endl;
	for(int i=0;i<4;i++)
		//b_sub[i]=0.25*f*el_x*el_y;
		b_sub[i]=1.0/((n+1)*(n+1))*f*el_x*el_y;
	int m;
	for(int k=0;k<elem_count;k++){
		for(int i=0;i<4;i++){
			m=elementss[i+k*4];
			b[m]+=b_sub[i];

		}
	}*/
	

	
	
	
	//free device memory
	cudaFree(A_device);
	cudaFree(g_mapping_device);
	cudaFree(b_device);
    cudaFree(c_mapping_device);
		
	//apply dirichlet boundary conditions	
	
	for(int i=0;i<node_count;i++)
		if(		(nodes_x[i]==0) ||(nodes_x[i]==size_x) || (nodes_y[i]==0) || (nodes_y[i]==size_y))
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
		

		
	for(int i=0;i<node_count;i++	)
		cout << b[i] << endl;

	cout <<"----------------------" << endl;
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
	
	//matrix output
	
	/*cout.precision(3);
	for(int j=0;j<node_count;j++){			
			for(int i=0;i<node_count;i++){			 
				cout << A[i+j*node_count]<< "|";
			}
			cout << endl;			
		}
		*/
	ofstream Filematrix("matrix.txt");
	Filematrix << "{";
	for(int j=0;j<node_count;j++)
	{
		Filematrix << "{";
		for(int i=0;i<node_count-1;i++)		
			Filematrix <<A[i+j*(node_count)] << "," ;
		Filematrix <<A[node_count-1+j*(node_count)];
		Filematrix << "}";
		if(j!=node_count-1)
			Filematrix << ",";		
	}
	Filematrix << "}";
	Filematrix.close();

	
	//create file output of the solution vector
	stringstream fnAssembly;

	string Filename="";
	
	fnAssembly << "output" << N_x <<"_" << N_y<< ".txt";
	fnAssembly >> Filename;
	ofstream File(Filename);
	File << "{";
	for(int j=0;j<N_y+1;j++)
	{
		File << "{";
		for(int i=0;i<N_x;i++)		
			File <<x[i+j*(N_x+1)] << "," ;
		File <<x[N_x+j*(N_x+1)];
		File << "}";
		if(j!=N_y)
			File << ",";		
	}
	File << "}";
	File.close();
	
	//CODE
	double test;
	//cin >> test ;
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

	/*allocation of necessary host memory*/
	double *coordinatesX= new double[ElementCount*PointsPerElement];
	double *coordinatesY= new double[ElementCount*PointsPerElement];	

	/*allocation of necessary device memory*/
	double	*coo_values_device;
	coordinates		*coo_index_device;
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
	cudaMalloc((void**)&coo_index_device, ElementCount*PointsPerElement*sizeof(int)*2);

	
	/*create triangulation for the simulation*/
	elements=createTriangulation(coordinatesX,coordinatesY,degree,elementsX,elementsY,sizeX,sizeY);
	
	/*copy necessarry memory to device*/	
	cudaMemcpy(elements_device,elements, ElementCount*PointsPerElement*sizeof(int), cudaMemcpyHostToDevice);



	/*assemble system matrix*/
	double a=sizeX/elementsX;
	double b=sizeY/elementsY;
	BernBinomCoeff<<<dimGrid, dimBlockM>>>(M_device, degree);

	BernBinomCoeff<<<dimGrid, dimBlockM_m>>>(M_m_device, degree-1);
	

	ass_A_exact<<<dimGrid, dimBlock>>>(a,b,coo_index_device,coo_values_device,degree, elements_device, M_device, M_m_device);

	/*assemble load vector*/

	/*apply dirichlet boundary conditions*/

	/* convert coo output into crs format*/

	/*solve system of equations*/

	/*write solution into file*/

	double *coo_values = new double[ElementCount*PointsPerElement*PointsPerElement];
	cudaMemcpy(coo_values,coo_values_device, ElementCount*PointsPerElement*PointsPerElement*sizeof(double), cudaMemcpyDeviceToHost);
	printMatrix(coo_values, (degree+1)*( degree+1),(degree+1)*( degree+1));

	testMatrixSym(coo_values,(degree+1)*(degree+1),(degree+1)*(degree+1));

	/*free memory*/
	cudaFree(coo_values_device);
	cudaFree(coo_index_device);
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


