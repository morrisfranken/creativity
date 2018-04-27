/*
 * cudaUtils.cpp
 *
 *  Created on: 28 Aug 2016
 *      Author: morris
 */

#include <iostream>
#include "utils.h"

using namespace std;

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}