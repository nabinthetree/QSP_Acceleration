



#include "finite_field_arithmetic.h"



__global__
void fadd(int n,i64* d_out, i64* d_a, i64* d_b)
{


	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
		d_out[i] = d_a[i] + d_b[i];

	return;

}

__global__
void fsub(int n, i64* d_out, i64* d_a, i64* d_b)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
		d_out[i] = d_a[i] - d_b[i];

	return;
}



