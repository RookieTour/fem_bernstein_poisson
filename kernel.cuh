#ifndef kernel_cuh
#define kernel_cuh
__global__ void red_ass_A(double* A_device, int* g_mapping_device, double a, double b, int n, int Nx, int Ny);
__global__ void pink_ass_A(double* A_device, int* g_mapping_device, double a, double b, int n, int Nx, int Ny);
__global__ void blue_ass_A(double* A_device, int* g_mapping_device, double a, double b, int n, int Nx, int Ny);
__global__ void yellow_ass_A(double* A_device, int* g_mapping_device, double a, double b, int n, int Nx, int Ny);
#endif
