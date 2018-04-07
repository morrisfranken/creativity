/*
 * Image.cpp
 *
 *  Created on: 28 Aug 2016
 *      Author: morris
 */

#include <iostream>
#include <memory>
#include <opencv2/core.hpp>

#include "cuImage.h"
#include "cudaUtils.h"

using namespace std;
using namespace cv;

namespace cu {
	__global__ void
	f32torgb(float *in, unsigned char *res, const int pitch_in, const int pitch_res, const int width, const int height, const float multiplier) {
		const int row = blockIdx.y * blockDim.y + threadIdx.y;
		const int col = blockIdx.x * blockDim.x + threadIdx.x;

		if (row < height & col < width) {    // make sure the thread is within the image
			const int idx_res = row * pitch_res + col * 3;
			const int idx_in = row * pitch_in + col * 3;

			for (int chan = 0; chan < 3; chan++)
				res[idx_res + chan] = static_cast<unsigned char>(in[idx_in + chan] * multiplier);
		}
	}

	Image::Image() {
	}

	Image::Image(const Mat &img) {
		upload(img);
	}

	Image::Image(const int height, const int width, const int type, const bool init_zero/*=false*/) {
		this->malloc(height, width, type, init_zero);
	}

	Image::~Image() {
	}

	void Image::malloc(int height, int width, int type, bool init_zero/*=false*/) {
		assert(data == NULL);
		cv::Mat dummy(1, 1, type);
		this->width = width;
		this->height = height;
		this->type = type;

		row_size = dummy.step * width;
		CUDA_CHECK_RETURN(cudaMallocPitch(&data, &pitch, row_size, height));
		if (init_zero)
			this->memset(0);
	}

	void Image::upload(const cv::Mat &img) {
		assert(data == NULL); // already uploaded
		width = img.cols;
		height = img.rows;
		type = img.type();

		row_size = img.step;
		CUDA_CHECK_RETURN(cudaMallocPitch(&data, &pitch, row_size, height));
		CUDA_CHECK_RETURN(cudaMemcpy2D(data, pitch, img.data, row_size, row_size, height, cudaMemcpyHostToDevice));
	}

	cv::Mat Image::download() const {
		Mat res = Mat(height, width, type);
		CUDA_CHECK_RETURN(cudaMemcpy2D(res.data, row_size, data, pitch, row_size, height, cudaMemcpyDeviceToHost));
		return res;
	}

	cv::Mat Image::downloadAsRGB(const float multiplier/*=1.0f*/) {
		assert(type == CV_32FC3);
		if (rgb == nullptr) {
			rgb = std::make_shared<Image>(height, width, CV_8UC3);
		}

		constexpr int BLOCK_SIZE = 32;
		const dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
		const dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

		f32torgb <<< dimGrid, dimBlock >>> ((float *) p(), rgb->p(), pitch / sizeof(float), rgb->pitch, width, height, multiplier);
		CUDA_CHECK_RETURN(cudaPeekAtLastError());

		return rgb->download();
	}

	void Image::memset(const unsigned char value) {
		CUDA_CHECK_RETURN(cudaMemset2D(data, pitch, value, row_size, height));
	}
}