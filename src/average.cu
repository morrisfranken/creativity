/*
 * average.cpp
 *
 *  Created on: 24 Mar 2018
 *      Author: morris
 */
#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <boost/timer/timer.hpp>

#include "average.h"
#include "cuUtils/cuImage.h"
#include "utils/utils.h"

using namespace std;
using namespace cv;

__global__ void scale_add(unsigned char *in, float *res, const int pitch_in, const int pitch_res, const int width_res, const int height_res, const float f_width, const float f_height) {
	const int row  = blockIdx.y * blockDim.y + threadIdx.y;
	const int col  = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < height_res & col < width_res) {	// make sure the thread is within the image
		const int idx_res = row * pitch_res + col * 3;
		const int idx_in = int(f_height * row) * pitch_in + int(f_width * col) * 3;

		for (int chan = 0; chan < 3; chan++)
			res[idx_res + chan] += (float)in[idx_in + chan];
	}
}

void average::run(const std::string &path) {
    const vector<string> files = my_utils::listdir(path);
	cv::Mat total_ = cv::Mat::zeros(1080/2, 1920/2, CV_32FC3);
	cu::Image gputotal(total_);

	constexpr int BLOCK_SIZE = 32;
	const dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
	const dim3 dimGrid((gputotal.width + dimBlock.x - 1) / dimBlock.x, (gputotal.height + dimBlock.y - 1) / dimBlock.y);

    boost::timer::cpu_timer watch;
	#pragma omp parallel for ordered schedule(dynamic)
	for (int i = 0; i < files.size(); i++) {
		const cv::Mat img = cv::imread(files[i], cv::IMREAD_COLOR);

		#pragma omp ordered
		{
			if (i % 100 == 0)
				cout << i << " / " << files.size() << endl;

			cu::Image gpuimg(img);
			const float f_width = gpuimg.width / (float)gputotal.width;
			const float f_height = gpuimg.height / (float)gputotal.height;
			scale_add<<<dimGrid, dimBlock>>>(gpuimg.p(), gputotal.p<float>(), gpuimg.pitch(), gputotal.pitch<float>(), gputotal.width, gputotal.height, f_width, f_height);
            CUDA_CHECK_RETURN(cudaPeekAtLastError());

			if (omp_get_thread_num() == 0) {
				static int show_count = 0;
				show_count++;
                if (show_count%10==0) {
                    const float multiplier = 1.0f / (float)(i+1);
                    Mat res = gputotal.downloadAsRGB(multiplier);
                    imshow("frame", res);
                    if (waitKey(1) == 27)
                        exit(0);
				}
			}
		}
	}
    cout << "done in " << watch.format() << endl;

    const float multiplier = 1.0f / (float)(files.size());
    Mat res = gputotal.downloadAsRGB(multiplier);
	imshow("frame", res);
	imwrite("average.png", res);
	waitKey();
}
