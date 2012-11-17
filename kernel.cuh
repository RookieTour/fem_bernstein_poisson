#ifndef kernel_cuh
#define kernel_cuh

__global__ void compute_bernstein(double *bernstein_q0, double *dbernstein_q0, double *bernstein_q1, double *dbernstein_q1, int n);
__global__ void ass_A(double* A_device, int* g_mapping_device,int* c_mapping,  double a, double b, int n, int Nx, int Ny,int color,int shiftx, int shifty);
__global__ void ass_A_exact(double* A_device, int* g_mapping_device, int* c_mapping, double a, double b, int n, int Nx, int Ny, int color, int shiftx, int shifty);
__global__ void VectorAdd(double* X, double* Y, double* Z, double alpha, int n);
#endif
