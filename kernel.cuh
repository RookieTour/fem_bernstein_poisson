#ifndef kernel_cuh
#define kernel_cuh

struct coordinates;
	
__global__ void BernBinomCoeff(double *M, int n);
__global__ void ass_A_exact(double a, double b, coordinates *coo_index, double*coo_value,int degree, int *elements, double *M, double *M_m);

#endif
