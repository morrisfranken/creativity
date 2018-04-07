/*
 * Image.cpp
 *
 *  Created on: 28 Aug 2016
 *      Author: morris
 */

#include <iostream>
#include <opencv2/core/matx.hpp>
#include <boost/timer/timer.hpp>

#include "cuVector.h"

using namespace std;

namespace cu {
    template<class T>
    Vector<T>::~Vector() {

    }

    template<class T>
    Vector<T>::Vector() {

    }

    template<class T>
    Vector<T>::Vector(const std::vector<T> &v) {
        size = v.size();
        CUDA_CHECK_RETURN(cudaMalloc(&data, size * sizeof(T)));
        upload(v);
    }

    template<class T>
    void Vector<T>::upload(const std::vector<T> &v) {
        CUDA_CHECK_RETURN(cudaMemcpy(data, v.data(), size * sizeof(T), cudaMemcpyHostToDevice));
    }

    template<class T>
    Vector<T>::Vector(const std::size_t size) : size(size) {
        CUDA_CHECK_RETURN(cudaMalloc(&data, size * sizeof(T)));
    }

    template<class T>
    std::vector<T> Vector<T>::download() const {
        std::vector<T> res(size);
        CUDA_CHECK_RETURN(cudaMemcpy(&res[0], data, size * sizeof(T), cudaMemcpyDeviceToHost));
        return res;
    }

    template class Vector<cv::Vec3b>;
}