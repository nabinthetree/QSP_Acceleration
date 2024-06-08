

#ifndef FINITE_H
#define FINITE_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdio.h>
#include <iostream>

typedef unsigned char u8;
typedef long long i64;
typedef i64 field_elem[16];

class Finite_Field_255_19
{
public:

	Finite_Field_255_19 operator+(Finite_Field_255_19 const& op2);
	Finite_Field_255_19 operator-(Finite_Field_255_19 const& op2);
	Finite_Field_255_19 operator*(Finite_Field_255_19 const& op2);
	
	Finite_Field_255_19();
	Finite_Field_255_19(const Finite_Field_255_19& toCopy);
	Finite_Field_255_19(const u8* in);
	~Finite_Field_255_19();

	i64* const GetValue() { return field_element; }



private:
	i64* field_element;
	int N;
	void get_data_to_device(i64* source,i64* &d_elem);
	void allocate_gpu_space(i64* &d_elem);
	void deallocate_gpu_space(i64* &d_elem);
	void copyHostDevice(i64* elem, i64* d_elem, cudaMemcpyKind flag);
	void unpack25519(i64* out, const u8* in);
	void carry25519(i64* elem);
};

__global__
void fadd(int n,i64* out, i64* a, i64* b); /* the sumb will be stored in out */

__global__
void fsub(int n, i64* out, i64* a, i64* b); /* the sumb will be stored in out */



#endif