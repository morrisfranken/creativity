/*
 * DeviceImage.h
 *
 *  Created on: 28 Aug 2016
 *      Author: morris
 */

#ifndef DEVICEIMAGE_H_
#define DEVICEIMAGE_H_

#include <memory>
#include <opencv2/core.hpp>

#include "cudaUtils.h"

class GpuImage : public cuSharedPointer {
	void malloc(size_t height, size_t width, int type);
public:
	size_t pitch	= 0;
	size_t width	= 0;
	size_t height	= 0;
	size_t channels	= 0;
	int type		= 0;

	~GpuImage();
	GpuImage();
	GpuImage(const cv::Mat &img);
	GpuImage(const size_t height, const size_t width, const int type);

	void upload(const cv::Mat &img);
	cv::Mat download();
};

#endif /* DEVICEIMAGE_H_ */
