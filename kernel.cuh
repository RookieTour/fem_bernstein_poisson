#ifndef kernel_cuh
#define kernel_cuh

struct coordinates;

//computes the product of bernstein coeffitients
__global__ void BernBinomCoeff(double *M, int n);

//computes list of elemental stiffnes matrices
__global__ void ass_A_exact(double a, double b, int *coo_row_device,int *coo_col_device, double*coo_value,int degree,long long int* index, int *elements, double *M, double *M_m, int elementsX, int elementsY);


//fill array with size <size> with value <value>
__global__ void fillArray(double* array, int size, double value);

//applies Dirichlet to matrix
__global__ void applyDirichlet(double* load, double* csr_matrix, int* csr_col_device,int* csrRowPtr,int *isboundaryNode, unsigned int entries, int elementsX, int elementsY, int degree, double boundaryValue);

//applies dirichlet to RHS
__global__ void vectorDirichlet(double* load, double* csr_matrix, int* csr_col_device,int* csrRowPtr,int *isboundaryNode,unsigned int entries, int elementsX, int elementsY, int degree, double boundaryValue);

//splits index to (i,j) pair
__global__ void split(long long int *index, int*cols, int*rows,unsigned long int pointCount,unsigned long int length);

//computes load vector
__global__ void loadVector(double* loadList, int* index,int *elements, double a, double b, int degree, double func, int ElementCount);
#endif
