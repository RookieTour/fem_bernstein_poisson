#ifndef kernel_cuh
#define kernel_cuh

struct coordinates;
	
__global__ void BernBinomCoeff(double *M, int n);
__global__ void ass_A_exact(double a, double b, int *coo_row_device,int *coo_col_device, double*coo_value,int degree, int *elements, double *M, double *M_m, int elementsX, int elementsY);
__global__ void fillArray(double* array, int size, double value);
__global__ void applyDirichlet(double* load, double* csr_matrix, int* csr_col_device,int* csrRowPtr,int *isboundaryNode, int entries, int elementsX, int elementsY, int degree, double boundaryValue);
__global__ void vectorDirichlet(double* load, double* csr_matrix, int* csr_col_device,int* csrRowPtr,int *isboundaryNode, int entries, int elementsX, int elementsY, int degree, double boundaryValue);
__global__ void reduce(double* data, int* index, int length);
__global__ void split(int *index, int*cols, int*rows,int pointCount, int length);
#endif
