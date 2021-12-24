#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  
  __pp_vec_float x;
  __pp_vec_float result;

  __pp_vec_int y;
  __pp_vec_int count;

  __pp_vec_float max = _pp_vset_float(9.999999f);
  __pp_vec_float one_f = _pp_vset_float(1.f);

  __pp_vec_int one_t = _pp_vset_int(1);
  __pp_vec_int zero = _pp_vset_int (0);

  __pp_mask maskAll, maskIsZero, maskIsNotZero, maskIsHuge, maskIsGtZero;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    if ( N % VECTOR_WIDTH != 0 && i == (N / VECTOR_WIDTH)*VECTOR_WIDTH )  
    {
      int remain = N % VECTOR_WIDTH;
      maskAll = _pp_init_ones(remain);
    }
    else 
    {
      maskAll = _pp_init_ones();
    }

    maskIsZero = _pp_init_ones(0);
    maskIsNotZero = _pp_init_ones(0);
    maskIsHuge = _pp_init_ones(0);
    maskIsGtZero = _pp_init_ones(0);

    _pp_vload_float(x, values+i, maskAll); // x = values[i]
  
    _pp_vload_int(y, exponents+i, maskAll); // y = exponents[i]
  
    _pp_veq_int(maskIsZero, y, zero, maskAll); // if (y == 0) {
  
    _pp_vstore_float(output+i, one_f, maskIsZero); // output[i] = 1.f

    //maskIsNotZero = maskAll and not(maskIsZero) 
    maskIsNotZero = _pp_mask_not(maskIsZero); 

    maskIsNotZero = _pp_mask_and(maskIsNotZero, maskAll); // } else {

    _pp_vmove_float(result, x, maskIsNotZero); // result = x
  
    _pp_vsub_int(count, y, one_t, maskIsNotZero); //count = y -1
  
    _pp_vgt_int(maskIsGtZero, count, zero, maskIsNotZero); //count > 0?

    while ( _pp_cntbits(maskIsGtZero) > 0 )  // while( count > 0) {
    {
      _pp_vmult_float(result, result, x, maskIsGtZero);  // result *= x

      _pp_vsub_int(count, count, one_t, maskIsGtZero);  // count--

      _pp_vgt_float(maskIsHuge, result, max, maskIsGtZero);  // if (result > 9.999999f){

      _pp_vmove_float(result, max, maskIsHuge);  // result = 9.999999f

      _pp_vgt_int(maskIsGtZero, count, zero, maskIsGtZero); // count > 0?
    }
    _pp_vstore_float(output+i, result, maskIsNotZero);  // } output[i] = result

  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  float sum = 0;
  float *sum_tmp = (float *)malloc(VECTOR_WIDTH * sizeof(float));

  __pp_mask maskAll;
  __pp_mask maskToAdd;
  maskAll = _pp_init_ones();
  maskToAdd = _pp_init_ones(VECTOR_WIDTH/2);

  __pp_vec_float val;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    _pp_vload_float(val, values+i, maskAll);  //val = values[i]

    _pp_hadd_float(val, val); //hadd(val)

    _pp_interleave_float(val, val); //interleave(val)

    _pp_vstore_float(sum_tmp, val, maskAll);

    for(int i=0; i<VECTOR_WIDTH/2; i++)
    {
      sum += sum_tmp[i];
    }
  }

  return sum;
}




