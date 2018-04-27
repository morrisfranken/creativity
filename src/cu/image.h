/*
 * DeviceImage.h
 *
 *  Created on: 28 Aug 2016
 *      Author: morris
 */

#ifndef CUIMAGE_H_
#define CUIMAGE_H_

#include <memory>
#include <opencv2/core.hpp>

#include "shared_pointer.h"

namespace cu {
	class Image : public SharedPointer {
		std::shared_ptr<Image> rgb = nullptr;    // is used for downloadAsRGB, and is kept as cache to avoid re-allocating

		void malloc(int height, int width, int type, bool init_zero = false);

		size_t row_size = 0;
        size_t pitch_ = 0;
	public:
		int width = 0;
		int height = 0;
		int type = 0;

		~Image();
		Image();

		Image(const cv::Mat &img);
		Image(int height, int width, int type, bool init_zero = false);
		void upload(const cv::Mat &img);
		cv::Mat download() const;
		cv::Mat downloadAsRGB(float multiplier = 1.0f);
		void memset(unsigned char value);

	    template <class T = unsigned char> size_t pitch() const {
            return pitch_ / sizeof(T);
        }
	};
}

#endif /* CUIMAGE_H_ */
