#ifndef kernel_cuh
#define kernel_cuh
__global__ void ass_A(double* A_device, int* g_mapping_device,int* c_mapping,  double a, double b, int n, int Nx, int Ny,int color,int shiftx, int shifty);
__global__ void VectorAdd(double* X, double* Y, double* Z, double alpha, int n);
#endif
