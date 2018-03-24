/*
 * GpuImage.cpp
 *
 *  Created on: 28 Aug 2016
 *      Author: morris
 */

#include <iostream>
#include <memory>
#include <opencv2/core.hpp>

#include "GpuImage.h"
#include "cudaUtils.h"

using namespace std;
using namespace cv;

GpuImage::GpuImage() {
}

GpuImage::GpuImage(const Mat &img) {
	upload(img);
}

GpuImage::GpuImage(const size_t height, const size_t width, const int type) {
	malloc(height, width, type);
}

GpuImage::~GpuImage() {
}

void GpuImage::malloc(size_t height, size_t width, int type) {
	assert(data == NULL);
	cv::Mat dummy(1,1, type);
	this->width = width;
	this->height = height;
	this->channels = dummy.channels();
	this->type = type;

	const int row_size = dummy.step * width;
	CUDA_CHECK_RETURN(cudaMallocPitch(&data, &pitch, row_size, height));
}

void GpuImage::upload(const cv::Mat &img) {
	assert(data == NULL); // already uploaded
	width = img.cols;
	height = img.rows;
	channels = img.channels();
	type = img.type();

	const int row_size = img.step;
	CUDA_CHECK_RETURN(cudaMallocPitch(&data, &pitch, row_size, height));
	CUDA_CHECK_RETURN(cudaMemcpy2D(data, pitch, img.data, row_size, row_size, height, cudaMemcpyHostToDevice));
}

cv::Mat GpuImage::download() {
	Mat res = Mat(height, width, type);
	const int row_size = res.step;
	CUDA_CHECK_RETURN(cudaMemcpy2D(res.data, row_size, data, pitch, row_size, height, cudaMemcpyDeviceToHost));
	return res;
}
