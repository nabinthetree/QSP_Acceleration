

#ifndef FINITE_H
#define FINITE_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdio.h>
#include <iostream>

typedef unsigned char u8;
typedef long long i64;
typedef i64 field_elem[16];

__global__
void fadd(int n,i64* out, i64* a, i64* b); /* the sumb will be stored in out */

__global__
void fsub(int n, i64* out, i64* a, i64* b); /* the sumb will be stored in out */



#endif