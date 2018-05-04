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
#include "cu/vector.h"

using namespace std;
using namespace cv;

constexpr int BLOCK_SIZE = 32;
const dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

#define TESTINGS false

#if TESTINGS
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

            for (int chan = 0; chan < 3; chan++) {
                const float diff = (float) in[idx_in + chan] - mean[idx_mean + chan];
                res[idx_res + chan] += diff * diff;
            }
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

 TODO(morris) : actually use optimized pca for eigen vector extraction instead of this.
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
#endif

// CUDA covariance
// covariance can be computed sequentially, which is perfect for cuda. This function should accomplish the following:
// - scale
// - subtract mean
// - store intermediate value
// ----- new function ----------
// - compute covariance with itself
// - add to existing covariance matrix
// Dont forget to multiply with `1 / (n-1)` at the very end!
// https://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html
// https://docs.opencv.org/3.1.0/d1/dee/tutorial_introduction_to_pca.html

__global__ void eigen_scale_subtract_mean(const uchar *img, const uchar *mean, short *result, const int pitch_img, int pitch_mean, const int p, const int width, const int height, const float f_width, const float f_height) {
    const int row  = blockIdx.y * blockDim.y + threadIdx.y;
    const int col  = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height & col < width) {	// make sure the thread is within the image
        const int idx_mean = row * pitch_mean + col * 3;
        const int idx_res  = row * p + col * 3;
        const int idx_img  = int(f_height * row) * pitch_img + int(f_width * col) * 3;

        result[idx_res + 0] = (short)img[idx_img + 0] - (short)mean[idx_mean + 0];
        result[idx_res + 1] = (short)img[idx_img + 1] - (short)mean[idx_mean + 1];
        result[idx_res + 2] = (short)img[idx_img + 2] - (short)mean[idx_mean + 2];
    }
}

__global__ void eigen_add_covariance(const short *B, unsigned long *C, const int pitch_C, const int p) {
    const int row  = blockIdx.y * blockDim.y + threadIdx.y;
    const int col  = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < p & col < p) {
        const int idx_cov = row * pitch_C + col * 3;

        C[idx_cov + 0] += B[row * 3 + 0] * B[col * 3 + 0];
        C[idx_cov + 1] += B[row * 3 + 1] * B[col * 3 + 1];
        C[idx_cov + 2] += B[row * 3 + 2] * B[col * 3 + 2];
    }
}

namespace eigen_art {
    void scale_subtract_mean(const cu::Image &img, const cu::Image &mean, cu::Vector<short> &B) {
        const float f_width = img.width / (float)mean.width;
        const float f_height = img.height / (float)mean.height;
        const dim3 dimGrid((mean.width + dimBlock.x - 1) / dimBlock.x, (mean.height + dimBlock.y - 1) / dimBlock.y);

        eigen_scale_subtract_mean<<<dimGrid, dimBlock>>>(img.p(), mean.p(), B.p<short>(), img.pitch(), mean.pitch(), (int)B.size, mean.width, mean.height, f_width, f_height);
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
    }

    void add_covariance(cu::Image &C, const cu::Vector<short> &B) {
        const dim3 dimGrid((B.size + dimBlock.x - 1) / dimBlock.x, (B.size + dimBlock.y - 1) / dimBlock.y);

        eigen_add_covariance<<<dimGrid, dimBlock>>>(B.p<short>(), C.p<unsigned long>(), C.pitch<unsigned long>(), B.size);
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
    }
}
////////////////


Mat normMatf(Mat img) {
    float minVal = INFINITY, maxVal=-INFINITY;
    for (int y = 0; y < img.rows; y++) {
        const Vec3f *row = (Vec3f *)img.ptr(y);
        for (int x = 0; x < img.cols; x++) {
            const Vec3f &p = row[x];
            if (p[0] < minVal) minVal = p[0];
            if (p[1] < minVal) minVal = p[1];
            if (p[2] < minVal) minVal = p[2];

            if (p[0] > maxVal) maxVal = p[0];
            if (p[1] > maxVal) maxVal = p[1];
            if (p[2] > maxVal) maxVal = p[2];
        }
    }

    Mat res;
    img.convertTo(res, CV_8UC3, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
    return res;
}

void eigen_art::run(const std::string &path, const std::string &cache_path) {
    const vector<string> files = my_utils::listdir(path);
    Size size(1920/4, 1080/4);

    string pca_path = cache_path + "/pca";
    boost::filesystem::create_directory(pca_path);

    vector<Mat> images(files.size());
    Mat data(0, size.width * size.height, CV_8UC3);
    cout << "loading images" << endl;
    for (size_t i = 0; i < files.size(); i++) {
        Mat img = imread(files[i]);
        cv::resize(img, img, size);
        images[i] = img.reshape(1,1); // flatten
        data.push_back(images[i]);
    }

    cout << "computing pca: " << data.size() << endl;
    PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, 10);
    Mat averageFace = pca.mean.reshape(3,size.height);
    imwrite("average_pca.png", averageFace);

    vector<Mat> eigen_images;
    for(int i = 0; i < 10; i++) {
        Mat eigenFace = pca.eigenvectors.row(i).reshape(3,size.height);
        cout <<  "i = " << i << " :: " << eigenFace.size() << endl;
        eigen_images.push_back(eigenFace);
        my_utils::saveMat(pca_path + "/" + std::to_string(i) + ".raw", eigenFace);
        imwrite(pca_path + "/" + std::to_string(i) + ".png", normMatf(eigenFace));
    }
}
