

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "test_finite_field.h"
#include "finite_field_arithmetic.h"

#include <iostream>
#include <assert.h>



void Test_fAdd(void)
{
    int N = 16;
    i64* a = (i64*)malloc(16 * sizeof(i64));
    i64* b = (i64*)malloc(16 * sizeof(i64));
    i64* sol = (i64*)malloc(16 * sizeof(i64));
    i64* out = (i64*)malloc(16 * sizeof(i64));

    i64* d_a;
    i64* d_b;
    i64* d_out;

    cudaMalloc(&d_a, 16 * sizeof(i64));
    cudaMalloc(&d_b, 16 * sizeof(i64));
    cudaMalloc(&d_out, 16 * sizeof(i64));

    for (int i = 0; i < N; ++i)
    {
        a[i] = (i64)std::pow(2, 12);
        b[i] = (i64)std::pow(2, 14);
        sol[i] = (i64)std::pow(2, 12) + (i64)std::pow(2, 14);
    }

    cudaMemcpy(d_a, a, N * sizeof(i64), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(i64), cudaMemcpyHostToDevice);

    // Perform the addition on the 16 elements
    fadd << <(N + 255) / 256, 256 >> > (N, d_out, d_a, d_b);

    cudaMemcpy(out, d_out, N * sizeof(i64), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i)
    {
        assert(out[i] == sol[i]);

    }
    printf("Assertion passed, elements added correctly\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    free(a);
    free(b);
    free(sol);
    free(out);

}




void Test_fSub()
{
    int N = 16;
    i64* a = (i64*)malloc(16 * sizeof(i64));
    i64* b = (i64*)malloc(16 * sizeof(i64));
    i64* sol = (i64*)malloc(16 * sizeof(i64));
    i64* out = (i64*)malloc(16 * sizeof(i64));

    i64* d_a;
    i64* d_b;
    i64* d_out;

    cudaMalloc(&d_a, 16 * sizeof(i64));
    cudaMalloc(&d_b, 16 * sizeof(i64));
    cudaMalloc(&d_out, 16 * sizeof(i64));

    for (int i = 0; i < N; ++i)
    {
        a[i] = (i64)std::pow(2, 12);
        b[i] = (i64)std::pow(2, 14);
        sol[i] = (i64)std::pow(2, 12) - (i64)std::pow(2, 14);
    }

    cudaMemcpy(d_a, a, N * sizeof(i64), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(i64), cudaMemcpyHostToDevice);

    // Perform the addition on the 16 elements
    fsub << <(N + 255) / 256, 256 >> > (N, d_out, d_a, d_b);

    cudaMemcpy(out, d_out, N * sizeof(i64), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i)
    {
        assert(out[i] == sol[i]);

    }
    printf("Assertion passed, elements subtracted correctly\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    free(a);
    free(b);
    free(sol);
    free(out);

}

void Test_Multiply()
{
    const u8 in1[32] = { 35 };
    const u8 in2[32] = { 128 };
    Finite_Field_255_19 a(in1);
    Finite_Field_255_19 b(in2);

    i64 sol = 35 * 128;

    Finite_Field_255_19 c = a * b;
    assert(a.GetValue()[0] == 35);
    assert(b.GetValue()[0] == 128);
    i64* const val = c.GetValue();

    assert(val[0] == sol);
    printf("Assertion passed, elements multiplied correctly\n");

}