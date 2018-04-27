//
// Created by morris on 4/19/18.
//

#include <iostream>
#include <vector>
#include <omp.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <boost/timer/timer.hpp>
#include <boost/filesystem/operations.hpp>

#include "eigenArt.h"
#include "utils/utils.h"
#include "cu/image.h"
#include "cu/utils.h"

using namespace std;
using namespace cv;

constexpr int BLOCK_SIZE = 32;
const dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

namespace d {
    __global__ void
    scale_add(unsigned char *in, float *res, const int pitch_in, const int pitch_res, const int width_res, const int height_res, const float f_width, const float f_height) {
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < height_res & col < width_res) {    // make sure the thread is within the image
            const int idx_res = row * pitch_res + col * 3;
            const int idx_in = int(f_height * row) * pitch_in + int(f_width * col) * 3;

            for (int chan = 0; chan < 3; chan++)
                res[idx_res + chan] += (float) in[idx_in + chan];
        }
    }


    __global__ void
    scale_meandiff_add(unsigned char *in, const float *mean, float *res, const int pitch_in, const int pitch_mean, const int pitch_res, const int width_res, const int height_res, const float f_width, const float f_height) {
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < height_res & col < width_res) {    // make sure the thread is within the image
            const int idx_res  = row * pitch_res + col * 3;
            const int idx_mean = row * pitch_mean + col * 3;
            const int idx_in   = int(f_height * row) * pitch_in + int(f_width * col) * 3;

            for (int chan = 0; chan < 3; chan++)
                const float diff = (float) in[idx_in + chan] - mean[idx_mean + chan]
                res[idx_res + chan] += diff * diff;
        }
    }

    __global__ void
    f32torgb_gray(float *in, unsigned char *res, const int pitch_in, const int pitch_res, const int width, const int height, const float multiplier) {
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < height & col < width) {    // make sure the thread is within the image
            const int idx_res = row * pitch_res + col * 3;
            const int idx_in = row * pitch_in + col * 3;

            for (int chan = 0; chan < 3; chan++)
                res[idx_res + chan] = static_cast<unsigned char>(in[idx_in + chan + 128] * multiplier);
        }
    }

    __global__ void
    addf(float *a, float *b, const int pitch_a, const int pitch_b, const int width_res, const int height_res) {
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < height_res & col < width_res) {    // make sure the thread is within the image
            const int idx_a = row * pitch_a + col * 3;
            const int idx_b = row * pitch_b + col * 3;

            a[idx_a + 0] += b[idx_b + 0];
            a[idx_a + 1] += b[idx_b + 1];
            a[idx_a + 2] += b[idx_b + 2];
        }
    }

    __global__ void
    multf(float *a, const int pitch_a, const int width_res, const int height_res, const float multiplier) {
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < height_res & col < width_res) {    // make sure the thread is within the image
            const int idx_a = row * pitch_a + col * 3;

            a[idx_a + 0] *= multiplier;
            a[idx_a + 1] *= multiplier;
            a[idx_a + 2] *= multiplier;
        }
    }
}

namespace eigen_art {
    inline void scale_add(const cu::Image &img, cu::Image &total) {
        const float f_width = img.width / (float)total.width;
        const float f_height = img.height / (float)total.height;
        const dim3 dimGrid((total.width + dimBlock.x - 1) / dimBlock.x, (total.height + dimBlock.y - 1) / dimBlock.y);
        d::scale_add<<<dimGrid, dimBlock>>>(img.p(), total.p<float>(), img.pitch(), total.pitch<float>(), total.width, total.height, f_width, f_height);
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
    }

    inline void scale_meandiff_add(const cu::Image &img, cu::Image &mean, cu::Image &total) {
        const float f_width = img.width / (float)total.width;
        const float f_height = img.height / (float)total.height;
        const dim3 dimGrid((total.width + dimBlock.x - 1) / dimBlock.x, (total.height + dimBlock.y - 1) / dimBlock.y);
        d::scale_meandiff_add<<<dimGrid, dimBlock>>>(img.p(), mean.p<float>(), total.p<float>(), img.pitch(), mean.pitch<float>(), total.pitch<float>(), total.width, total.height, f_width, f_height);
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
    }

    inline void visualise(const cu::Image &total, cu::Image &res, const float &multiplier) {
        const dim3 dimGrid((total.width + dimBlock.x - 1) / dimBlock.x, (total.height + dimBlock.y - 1) / dimBlock.y);
        d::f32torgb_gray<<< dimGrid, dimBlock >>> (total.p<float>(), res.p(), total.pitch<float>(), res.pitch(), total.width, total.height, multiplier);
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
    }

    inline void addf(cu::Image &a, const cu::Image &b) { // a += b
        assert((a.type == CV_32FC3) & (a.type == b.type) & (a.width == b.width) & (a.height == b.height));

        const dim3 dimGrid((a.width + dimBlock.x - 1) / dimBlock.x, (a.height + dimBlock.y - 1) / dimBlock.y);
        d::addf<<< dimGrid, dimBlock >>> (a.p<float>(), b.p<float>(), a.pitch<float>(), b.pitch<float>(), a.width, a.height);
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
    }

    inline void multf(cu::Image &a, const float multiplier) { // a *= multipler
        const dim3 dimGrid((a.width + dimBlock.x - 1) / dimBlock.x, (a.height + dimBlock.y - 1) / dimBlock.y);
        d::multf<<< dimGrid, dimBlock >>> (a.p<float>(), a.pitch<float>(), a.width, a.height, multiplier);
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
    }
}

// TODO(morris) : actually use optimized pca for eigen vector extraction instead of this.
void eigen_art::run(const std::string &path, const std::string &cache_path) {
    const vector<string> files = my_utils::listdir(path);
    const float multiplier = 1.0f / (float)(files.size());
    cu::Image total(1080/2, 1920/2, CV_32FC3, true);

    string pca_path = cache_path + "/pca";
    boost::filesystem::create_directory(pca_path);

    cu::Image mean(total.height, total.width, CV_32FC3, true);
    cu::Image vis(total.height, total.width, CV_8UC3);

    boost::timer::cpu_timer watch;
    for (int j = 0; j < 10; j++) {
//#pragma omp parallel for ordered schedule(dynamic)
        for (size_t i = 0; i < files.size(); i++) {
            const cv::Mat img = cv::imread(files[i], cv::IMREAD_COLOR);

//#pragma omp ordered
            {
                if (i % 100 == 0)
                    cout << j << " :: " << i << " / " << files.size() << endl;

                cu::Image gpuimg(img);
                eigen_art::scale_meandiff_add(gpuimg, mean, total);
//                eigen_art::scale_add(gpuimg, total);

                if (omp_get_thread_num() == 0) {
                    static int show_count = 0;
                    show_count++;
                    if (show_count % 10 == 0) {
                        const float multiplier2 = 1.0f / (float)(i+1);
                        eigen_art::visualise(total, vis, multiplier2);
                        Mat res = vis.download();
//                        Mat res = total.downloadAsRGB(multiplier2);
                        imshow("frame", res);
                        if (waitKey(0) == 27)
                            exit(0);
                    }
                }
            }
        }
        eigen_art::multf(total, multiplier);
        eigen_art::addf(mean, total);
    }
    cout << "done in " << watch.format() << endl;

}
