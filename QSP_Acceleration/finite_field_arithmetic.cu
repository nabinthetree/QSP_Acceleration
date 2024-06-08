



#include "finite_field_arithmetic.h"
#include <iostream>

__global__
void fadd(int n, i64* d_out, i64* d_a, i64* d_b)
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

void Finite_Field_255_19::carry25519(i64* elem)
{
	int i;
	i64 carry;

	for (i = 0; i < 16; ++i)
	{
		carry = elem[i] >> 16;
		elem[i] -= carry << 16;

		if (i < 15)
		{
			elem[i + 1] += carry;
			
		}
		else
		{
			elem[0] += 38 * carry;
		}
	}
}

Finite_Field_255_19::~Finite_Field_255_19()
{
	delete[] this->field_element;
}

Finite_Field_255_19::Finite_Field_255_19(const u8* in)
{
	this->N = 16;
	this->field_element = (i64*)malloc(this->N * sizeof(i64));
	this->unpack25519(this->field_element, in);
}

Finite_Field_255_19::Finite_Field_255_19()
{
	this->field_element = (i64*)malloc(16 * sizeof(i64));
	this->N = 16;
	for (unsigned int i = 0; i < 16; ++i)
	{
		this->field_element[i] = 0;
	}
}
void Finite_Field_255_19::get_data_to_device(i64* source,i64*& d_elem)
{
	this->allocate_gpu_space(d_elem);
	this->copyHostDevice(source, d_elem, cudaMemcpyHostToDevice);

}

void Finite_Field_255_19::copyHostDevice(i64* elem, i64* d_elem, cudaMemcpyKind flag)
{
	cudaMemcpy(d_elem, elem, 16*sizeof(i64),flag);
}
void Finite_Field_255_19::allocate_gpu_space(i64* &d_elem)
{
	cudaMalloc(&d_elem, 16 * sizeof(i64));
}

void Finite_Field_255_19::deallocate_gpu_space(i64*& d_elem)
{
	cudaFree(d_elem);
}
Finite_Field_255_19::Finite_Field_255_19(const Finite_Field_255_19& toCopy)
{
	this->N = toCopy.N;
	this->field_element = (i64*)malloc(this->N * sizeof(i64));
	for (unsigned int i = 0; i < 16; ++i)
	{
		this->field_element[i] = toCopy.field_element[i];
	}
}

Finite_Field_255_19 Finite_Field_255_19::operator+(Finite_Field_255_19 const& op2)
{
	int N = 16;
	i64* d_op1;
	i64* d_op2;
	i64* d_out;
	Finite_Field_255_19 out;
	this->get_data_to_device(this->field_element, d_op1);
	this->get_data_to_device(op2.field_element, d_op2);
	this->allocate_gpu_space(d_out);

	fadd << <(this->N + 255) / 256, 256 >> > (N, d_out, d_op1, d_op2);

	this->copyHostDevice(out.field_element, d_out, cudaMemcpyDeviceToHost);
	cudaFree(d_op1);
	cudaFree(d_op2);
	cudaFree(d_out);
	return out;
}


Finite_Field_255_19 Finite_Field_255_19::operator*(Finite_Field_255_19 const& op2)
{
	

	Finite_Field_255_19 product;
	i64 product_31[31];
	i64 i, j;

	for (i = 0; i < 16; ++i)
	{
		for (j = 0; j < 16; ++j)
		{
			product_31[i + j] = this->field_element[i] * op2.field_element[j];
		}
	}
	for (i = 0; i < 15; ++i)
	{
		product_31[i] += 38 * product_31[i + 16];
	}
	for (i = 0; i < 16; ++i)
	{
		product.field_element[i] = product_31[i];
	}

	this->carry25519(product.field_element);
	this->carry25519(product.field_element);
	this->carry25519(product.field_element);

	return product;
}


void Finite_Field_255_19::unpack25519(i64* out, const u8* in)
{
	int i;
	for (i = 0; i < 16; ++i)
	{
		out[i] = in[2 * i] + ((i64)in[2*i + 1] << 8);
		out[15] &= 0x7fff;
	}
}