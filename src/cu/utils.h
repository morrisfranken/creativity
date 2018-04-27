/*
 * cudaUtils.h
 *
 *  Created on: 28 Aug 2016
 *      Author: morris
 */

#ifndef CUDAUTILS_H_
#define CUDAUTILS_H_

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);

#endif /* CUDAUTILS_H_ */
