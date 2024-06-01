
#include "cuda_runtime.h"
#include "device_launch_parameters.h"



#include <stdio.h>
#include <iostream>
#include <assert.h>

#include "finite_field_arithmetic.h"
#include "test_finite_field.h"


int main(void)
{
    
    Test_fAdd();
    Test_fSub();
   
    return 0;
}
