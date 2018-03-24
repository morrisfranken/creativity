/*
 * cudaUtils.cpp
 *
 *  Created on: 28 Aug 2016
 *      Author: morris
 */

#include <iostream>
#include "cudaUtils.h"

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

cuSharedPointer::~cuSharedPointer() {
	if (--(*uses) <= 0 && data != NULL) {
		CUDA_CHECK_RETURN(cudaFree(data));
		delete uses;
	}
}
cuSharedPointer::cuSharedPointer() : data(NULL), uses(new short(1)) {}
cuSharedPointer::cuSharedPointer(const cuSharedPointer& a) : data(a.data), uses(a.uses) { (*uses)++; }

cuSharedPointer& cuSharedPointer::operator=(const cuSharedPointer& in) {
	if (this != &in) { // prevent self assignment
		(*in.uses)++;
		if (--(*uses) <= 0 && data != NULL) {
			CUDA_CHECK_RETURN(cudaFree(data));
			delete uses;
		}
		uses = in.uses;
		data = in.data;
	}

	return *this;
}

unsigned char *cuSharedPointer::p() const {
	return data;
}

