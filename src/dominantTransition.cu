//
// Created by morris on 4/14/18.
//
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <omp.h>
#include <iomanip>

#include "dominantTransition.h"
#include "cuUtils/cuImage.h"
#include "utils/utils.h"

using namespace std;
using namespace cv;

constexpr int BLOCK_SIZE = 32;

__global__ void trans_subtract(const uchar *src, uint *dst, const int pitch_src, const int pitch_dst, const int width, const int height) {
    const int row  = blockIdx.y * blockDim.y + threadIdx.y;
    const int col  = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width & row < height) {
        const int idx_src = row * pitch_src + col * 3;
        const int idx_dst = row * pitch_dst + col * 3;

        dst[idx_dst + 0] -= src[idx_src + 0];
        dst[idx_dst + 1] -= src[idx_src + 1];
        dst[idx_dst + 2] -= src[idx_src + 2];
    }
}

__global__ void trans_add(const uchar *src, uint *dst, const int pitch_src, const int pitch_dst, const int width, const int height) {
    const int row  = blockIdx.y * blockDim.y + threadIdx.y;
    const int col  = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width & row < height) {
        const int idx_src = row * pitch_src + col * 3;
        const int idx_dst = row * pitch_dst + col * 3;

        dst[idx_dst + 0] += src[idx_src + 0];
        dst[idx_dst + 1] += src[idx_src + 1];
        dst[idx_dst + 2] += src[idx_src + 2];
    }
}

__global__ void retrieve_average(const uint *src, uchar *dst, const int pitch_src, const int pitch_dst, const int width, const int height, const int count) {
    const int row  = blockIdx.y * blockDim.y + threadIdx.y;
    const int col  = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width & row < height) {
        const int idx_src = row * pitch_src + col * 3;
        const int idx_dst = row * pitch_dst + col * 3;

        dst[idx_dst + 0] = src[idx_src + 0] / count;
        dst[idx_dst + 1] = src[idx_src + 1] / count;
        dst[idx_dst + 2] = src[idx_src + 2] / count;
    }
}


void dominantTransition::run(const std::string &path) {
    vector<string> files = my_utils::listdir(path);
    sort(files.begin(), files.end());
    constexpr int temporal_size = 100;

    cu::Image total(1080 / 2, 1920 / 2, CV_32SC3, true);
    cu::Image dom_img(total.height, total.width, CV_8UC3);
    vector<shared_ptr<cu::Image>> images(temporal_size);

    const dim3 dimBlock2(BLOCK_SIZE, BLOCK_SIZE, 1);
    const dim3 dimGrid2((total.width + dimBlock2.x - 1) / dimBlock2.x, (total.height + dimBlock2.y - 1) / dimBlock2.y);

#pragma omp parallel for ordered schedule(dynamic)
    for (int i = 0; i < files.size(); i++) {
        const cv::Mat img_ = cv::imread(files[i], cv::IMREAD_COLOR);

#pragma omp ordered
        {
            const int temporal_idx = i % temporal_size;
            if (i % 100 == 0)
                cout << i << " / " << files.size() << endl;

            if (i > temporal_size) {
                cu::Image &prev_img = *images[temporal_idx];
                trans_subtract <<< dimGrid2, dimBlock2 >>> (prev_img.p(), total.p<uint>(), prev_img.pitch(), total.pitch<uint>(), total.width, total.height);
                CUDA_CHECK_RETURN(cudaPeekAtLastError());
            }
            images[temporal_idx] = make_shared<cu::Image>(img_);
            cu::Image &img = *images[temporal_idx];
            trans_add <<< dimGrid2, dimBlock2 >>> (img.p(), total.p<uint>(), img.pitch(), total.pitch<uint>(), total.width, total.height);
            CUDA_CHECK_RETURN(cudaPeekAtLastError());

            retrieve_average <<< dimGrid2, dimBlock2 >>> (total.p<uint>(), dom_img.p(), total.pitch<uint>(), dom_img.pitch(), total.width, total.height, min(i+1, temporal_size));
            CUDA_CHECK_RETURN(cudaPeekAtLastError());

            const Mat frame = dom_img.download(); // videoframes
            stringstream ss;
            ss << "../results/videoframes/" << setfill('0') << setw(5) << i << ".png";
            imwrite(ss.str(), frame);

            // show intermediate results
            if (omp_get_thread_num() == 0) {
                imshow("frame", frame);
                if (waitKey(1) == 27)
                    exit(0);
            }
        }
    }
}
