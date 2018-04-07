//
// Created by morris on 3/30/18.
//
//
#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <boost/timer/timer.hpp>
#include <boost/filesystem.hpp>
#include <map>

#include "dominantColour.h"
#include "average.h"
#include "cuUtils/cuImage.h"
#include "cuUtils/cuVector.h"
#include "utils/utils.h"
#include "utils/binWriter.h"
#include "utils/binReader.h"

using namespace std;
using namespace cv;

constexpr int BLOCK_SIZE = 32;

constexpr char cache_path_files[] = "files.raw";
constexpr char cache_path_image_means[] = "image_means.raw";
constexpr char cache_path_global_means[] = "global_means.raw";

#pragma omp declare reduction(vec_int_plus  : std::vector<int>     : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<int>())) initializer(omp_priv = omp_orig)
#pragma omp declare reduction(vec_int3_plus : std::vector<Vec3i>   : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<Vec3i>())) initializer(omp_priv = omp_orig)
//#pragma omp declare reduction(arr_int_plus  : std::array<int, K>   : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<int>())) initializer(omp_priv = omp_orig)
//#pragma omp declare reduction(arr_int3_plus : std::array<Vec3i, K> : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<Vec3i>())) initializer(omp_priv = omp_orig)

__global__ void label_pixels(const uchar *img, const int width, int height, const int pitch_img, const uchar3 *centroids, const int k, int *labels, const int pitch_labels) {
    const int row  = blockIdx.y * blockDim.y + threadIdx.y;
    const int col  = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.y * BLOCK_SIZE + threadIdx.x;

    extern __shared__ uchar3 sCentroid[]; // NOTE(morris) : size is equal to `k` and determined in calling function <<< grid, block, memsize >> (where memsize is in bytes)
    if (tid < k) {
        sCentroid[tid] = centroids[tid]; // copy global centroids to local block memory
    }
    __syncthreads();

    if (row < height & col < width) {	// make sure the thread is within the image
        const int idx_img = row * pitch_img + col * 3;
        const int idx_label = row * pitch_labels + col;

        const uchar3 pix = *(uchar3 *)(img + idx_img);

        int best_i = 0;
        int best_val = 0x0fffffff;

        // compute the closest centroid
        for (int i = 0; i < k; i++) {
            const auto &centroid = sCentroid[i];
            const int dx = pix.x - centroid.x;
            const int dy = pix.y - centroid.y;
            const int dz = pix.z - centroid.z;
            const int dist = dx*dx + dy*dy + dz*dz;
//            printf("pix[%d,%d] :: [%d] [%d,%d,%d] -- [%d,%d,%d] = %d\n", row, col, i, pix.x, pix.y, pix.z, centroid.x, centroid.y, centroid.z, dist);
            if (dist < best_val) {
                best_val = dist;
                best_i = i;
            }
        }
        labels[idx_label] = best_i;
    }
}
__global__ void compute_centroid2(const uchar *img, const int width, const int height, const int pitch_img, int *labels, const int pitch_labels,
    const int window_size, int *centroids_temp, const int pitch_centroids_temp, int *centroids_count, const int pitch_centroids_count, const int k
) {
//    const int row  = blockIdx.y * blockDim.y + threadIdx.y;
//    int col  = (blockIdx.x * blockDim.x + threadIdx.x) * window_size;
    const int row  = threadIdx.x;
//    int *centroid_temp_local = new int[k * 3]{0};

//    if (row < height) {
        int *centroids_temp_row = centroids_temp + row * pitch_centroids_temp;
        int *centroids_count_row = centroids_count + row * pitch_centroids_count;
//        const int col_end = min(col + window_size, width - 1);
//        for (; col < col_end; col++) {
    for (int col = 0; col < width; col++) {
            const int idx_img = row * pitch_img + col * 3;
            const int idx_label = row * pitch_labels + col;
            const int &label = labels[idx_label];
//        centroid_temp_local[label * 3 + 0] += img[idx_img + 0];
//        centroid_temp_local[label * 3 + 1] += img[idx_img + 1];
//        centroid_temp_local[label * 3 + 2] += img[idx_img + 2];
            centroids_temp_row[label * 3 + 0] += img[idx_img + 0];
            centroids_temp_row[label * 3 + 1] += img[idx_img + 1];
            centroids_temp_row[label * 3 + 2] += img[idx_img + 2];
            centroids_count_row[label]++;
        }
//    }

//    for (int i = 0; i < k*3; i++) {
//        centroids_temp_row[i] = centroid_temp_local[i];
//    }
//
//    delete[] centroid_temp_local;
}
__global__ void sum_temp_centroids(uchar *centroids, const int height, const int *centroids_temp, const int pitch_centroids_temp, const int *centroids_count, const int pitch_centroids_count) {
    const auto tidx = threadIdx.x;

    int count = 0;
    int3 sum{0};

    for (int i = 0; i < height; i++) {
        const int c_idx = i * pitch_centroids_temp + tidx * 3;
        const int count_idx = i * pitch_centroids_count + tidx;
        sum.x += centroids_temp[c_idx + 0];
        sum.y += centroids_temp[c_idx + 1];
        sum.z += centroids_temp[c_idx + 2];
        count += centroids_count[count_idx];
    }

//    printf("[%d] :: [%d, %d, %d] (%d)\n", tidx, sum.x, sum.y, sum.z, count);
    if (count) {
        centroids[tidx * 3 + 0] = sum.x / count;
        centroids[tidx * 3 + 1] = sum.y / count;
        centroids[tidx * 3 + 2] = sum.z / count;
    }
}

__global__ void retrieve_result_kmeans(uchar *img, const int width, int height, const int pitch_img, const uchar3 *centroids, const int k, const int *labels, const int pitch_labels) {
    const int row  = blockIdx.y * blockDim.y + threadIdx.y;
    const int col  = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height & col < width) {	// make sure the thread is within the image
        const int idx_img = row * pitch_img + col * 3;
        const int idx_label = row * pitch_labels + col;

        const uchar3 &pix = centroids[labels[idx_label]];
        img[idx_img + 0] = pix.x;
        img[idx_img + 1] = pix.y;
        img[idx_img + 2] = pix.z;

    }
}

__global__ void scale_dominant(const uint *labels, uint *dominant, const int pitch_labels, const int pitch_dominant, const int width_dominant, const int height_dominant, const float f_width, const float f_height, const int k) {
    const int row  = blockIdx.y * blockDim.y + threadIdx.y;
    const int col  = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height_dominant & col < width_dominant) {	// make sure the thread is within the image
        const int idx_dominant = row * pitch_dominant + col * k;
        const int idx_labels = int(f_height * row) * pitch_labels + int(f_width * col);

        dominant[idx_dominant + labels[idx_labels]]++;
    }
}

__global__ void retrieve_result_dominant(uchar *img, const uint *dominant, const uchar *centroids, const int pitch_img, const int pitch_dominant, const int width, const int height, const int k) {
    const int row  = blockIdx.y * blockDim.y + threadIdx.y;
    const int col  = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height & col < width) {	// make sure the thread is within the image
        const int idx_img = row * pitch_img + col * 3;
        const int idx_dominant = row * pitch_dominant + col * k;

        int dom_idx = 0;
        uint dom_value = 0;
        const uint *dom_pixel = dominant + idx_dominant;
        for (int i = 0; i < k; i++) {
            if (dom_pixel[i] > dom_value) {
                dom_value = dom_pixel[i];
                dom_idx = i;
            }
        }

        img[idx_img + 0] = centroids[dom_idx*3 + 0];
        img[idx_img + 1] = centroids[dom_idx*3 + 1];
        img[idx_img + 2] = centroids[dom_idx*3 + 2];

    }
}

///////////////////////////////////

void compute_centroid_cpu(const Mat &img, const cu::Image &labels_, cu::Vector<cv::Vec3b> &centroids_, const int k) {
    const Mat labels = labels_.download();
    std::vector<Vec3i> centroids(k);
    std::vector<Vec3b> centroidsb(k);
    std::vector<int> count(k);
//    std::array<Vec3i, K> centroids{};
//    std::vector<Vec3b> centroidsb(K);
//    std::array<int, K> count{};

    // omp seem to only improve ~15% in local tests, not worth it considering vs image loading
//#pragma omp parallel for reduction(arr_int3_plus : centroids), reduction(arr_int_plus : count)
//#pragma omp parallel for reduction(vec_int3_plus : centroids), reduction(vec_int_plus : count)
    for (int y = 0; y < labels.rows; y++) {
        const Vec3b *row_img = reinterpret_cast<const Vec3b *>(img.ptr(y));
        const int *row_labels = reinterpret_cast<const int *>(labels.ptr(y));
        for (int x = 0; x < labels.cols; x++) {
            const int &label = row_labels[x];
            centroids[label] += row_img[x];
            count[label]++;
        }
    }

    for (int i = 0; i < k; i++) {
        centroidsb[i] = centroids[i] / count[i];
    }

    centroids_.upload(centroidsb);
}

vector<Vec3b> cuda_kmeans_viewer(const Mat &img_, const int k, const int iterations) {
    Mat labels_ = Mat::zeros(img_.size(), CV_32S);
    vector<Vec3b> centroids_(k);

    while(true) {
        for (int i = 0; i < k; i++) {
            centroids_[i] = img_.at<Vec3b>(rand() % img_.rows, rand() % img_.cols); // TODO(morris) :: pick better inital random centers
        }

        cu::Image img(img_);
        cu::Image labels(labels_); // img_.rows, img_.cols, CV_32S
        cu::Vector<cv::Vec3b> centroids(centroids_);
        cu::Image resimg(img_);

//        cout << "img = " << endl << img_ << endl;

        imshow("frame", img_);
        if (waitKey() == 27)
            return Mat();

        const dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
        const dim3 dimGrid((img.width + dimBlock.x - 1) / dimBlock.x, (img.height + dimBlock.y - 1) / dimBlock.y);

        boost::timer::cpu_timer watch;
        for (int i = 0; i < iterations; i++) {
            label_pixels <<< dimGrid, dimBlock, k * sizeof(uchar3) >>> (img.p(), img.width, img.height, img.pitch, (const uchar3 *) centroids.p(), k, (int *) labels.p(), labels.pitch / sizeof(int));
            CUDA_CHECK_RETURN(cudaPeekAtLastError());

            {
                retrieve_result_kmeans <<< dimGrid, dimBlock >>> (resimg.p(), img.width, img.height, img.pitch, (const uchar3 *) centroids.p(), k, (int *) labels.p(), labels.pitch / sizeof(int));
                CUDA_CHECK_RETURN(cudaPeekAtLastError());
                imshow("frame", resimg.download());
                if (waitKey() == 27)
                    return {};
            }

//            compute_centroid_cpu(img_, labels, centroids, k);
//            compute_centroid <<< 1, k >>> (img.p(), img.width, img.height, img.pitch, (uchar3 *) centroids.p(), (int *) labels.p(), labels.pitch / sizeof(int));
//            CUDA_CHECK_RETURN(cudaPeekAtLastError());
//            cout << "labels = " << endl << labels.download()  << endl;

            cu::Image centroids_temp(img.height, k, CV_32SC3, true);
            cu::Image centroids_count(img.height, k, CV_32S, true);
            compute_centroid2 <<< 1, img.height >>> (img.p(), img.width, img.height, img.pitch, (int *) labels.p(), labels.pitch / sizeof(int), 0, (int *)centroids_temp.p(), centroids_temp.pitch / sizeof(int), (int *)centroids_count.p(), centroids_count.pitch / sizeof(int), k);
            CUDA_CHECK_RETURN(cudaPeekAtLastError());

            sum_temp_centroids <<< 1, k >>> (centroids.p(), img.height, (int *)centroids_temp.p(), centroids_temp.pitch / sizeof(int), (int *)centroids_count.p(), centroids_count.pitch / sizeof(int));
            CUDA_CHECK_RETURN(cudaPeekAtLastError());

//            for (auto &p : centroids.download())
//                cout << p << endl;
        }
        cout << "done in " << watch.format() << endl;

        label_pixels <<< dimGrid, dimBlock, k * sizeof(uchar3) >>> (img.p(), img.width, img.height, img.pitch, (const uchar3 *) centroids.p(), k, (int *) labels.p(), labels.pitch / sizeof(int));
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
        retrieve_result_kmeans <<< dimGrid, dimBlock >>> (resimg.p(), img.width, img.height, img.pitch, (const uchar3 *) centroids.p(), k, (int *) labels.p(), labels.pitch / sizeof(int));
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
        cout << "done in " << watch.format() << endl;

        imshow("frame", resimg.download());
        if (waitKey() == 27)
            return {};

        vector<Vec3b> res = centroids.download();
    }


//    Mat res = centroids.download();
//    return {};
}

vector<Vec3b> cuda_kmeans(const Mat &img_, const int k, const int iterations) {
    vector<Vec3b> centroids_(k);

    for (int i = 0; i < k; i++) {
        centroids_[i] = img_.at<Vec3b>(rand() % img_.rows, rand() % img_.cols); // TODO(morris) :: pick better inital random centers
    }

    cu::Image img(img_);
    cu::Image labels(img.height, img.width, CV_32S);
    cu::Vector<cv::Vec3b> centroids(centroids_);
//    cu::Image centroids_temp(img.height, k, CV_32SC3, true);
//    cu::Image centroids_count(img.height, k, CV_32S, true);

    const dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    const dim3 dimGrid((img.width + dimBlock.x - 1) / dimBlock.x, (img.height + dimBlock.y - 1) / dimBlock.y);

    for (int i = 0; i < iterations; i++) {
        label_pixels <<< dimGrid, dimBlock, k * sizeof(uchar3) >>> (img.p(), img.width, img.height, img.pitch, (const uchar3 *) centroids.p(), k, (int *) labels.p(), labels.pitch / sizeof(int));
        CUDA_CHECK_RETURN(cudaPeekAtLastError());

        compute_centroid_cpu(img_, labels, centroids, k);

        // this implementation of GPU kmeans does not improve over CPU-version,- so just use CPU for now
//        int window_size = k
//        centroids_temp.memset(0);
//        centroids_count.memset(0);
//        compute_centroid2 <<< 1, img.height >>> (img.p(), img.width, img.height, img.pitch, (int *) labels.p(), labels.pitch / sizeof(int), 0, (int *)centroids_temp.p(), centroids_temp.pitch / sizeof(int), (int *)centroids_count.p(), centroids_count.pitch / sizeof(int), k);
//        CUDA_CHECK_RETURN(cudaPeekAtLastError());
//
//        sum_temp_centroids <<< 1, k >>> (centroids.p(), img.height, (int *)centroids_temp.p(), centroids_temp.pitch / sizeof(int), (int *)centroids_count.p(), centroids_count.pitch / sizeof(int));
//        CUDA_CHECK_RETURN(cudaPeekAtLastError());
    }
//    cudaDeviceSynchronize();

    return centroids.download();
}

void test() {
    cudaFree(0);
    Mat img = imread("/home/morris/Pictures/examples/example.jpg");
//    Mat img = Mat(3,3, CV_8UC3);
//    for (int y = 0; y < img.rows; y++) {
//        for (int x = 0; x < img.cols; x++) {
//            img.at<Vec3b>(y,x) = {static_cast<uchar>(rand() % 256), static_cast<uchar>(rand() % 256),
//                                  static_cast<uchar>(rand() % 256)};
//        }
//    }

    boost::timer::cpu_timer watch;
    auto codebook = cuda_kmeans(img, 128, 5);
    cout << "done in " << watch.format() << endl;
//    for (auto &c : codebook)
//        cout << c << endl;
    return;
}

cv::Mat dominant_colour::computeImageMeans(const vector<string> &files, const int k) {
    Mat means = Mat(files.size(), k, CV_8UC3);

    #pragma omp parallel for ordered schedule(dynamic)
    for (int i = 0; i < files.size(); i++) {
        const cv::Mat img = cv::imread(files[i], cv::IMREAD_COLOR);

        #pragma omp ordered
        {
            if (i % 100 == 0)
                cout << i << " / " << files.size() << endl;

            const auto image_means = cuda_kmeans(img, k, 5);
            mempcpy(means.ptr(i), image_means.data(), means.step);
        }
    }

    return means;
}

void dominant_colour::computeDominant(const vector<string> &files, const vector<Vec3b> &centroids_) {
    const cu::Vector<cv::Vec3b> centroids(centroids_);
    const int k = centroids_.size();

    cu::Image dominant(1080 / 2, 1920 / 2, CV_32SC(k), true);
    cu::Image dom_img(dominant.height, dominant.width, CV_8UC3);
    const dim3 dimBlock2(BLOCK_SIZE, BLOCK_SIZE, 1);
    const dim3 dimGrid2((dominant.width + dimBlock2.x - 1) / dimBlock2.x, (dominant.height + dimBlock2.y - 1) / dimBlock2.y);

#pragma omp parallel for ordered schedule(dynamic)
    for (int i = 0; i < files.size(); i++) {
        const cv::Mat img_ = cv::imread(files[i], cv::IMREAD_COLOR);

#pragma omp ordered
        {
            if (i % 100 == 0)
                cout << i << " / " << files.size() << endl;

            cu::Image img(img_);
            cu::Image labels(img.height, img.width, CV_32S);

            const dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
            const dim3 dimGrid((img.width + dimBlock.x - 1) / dimBlock.x, (img.height + dimBlock.y - 1) / dimBlock.y);

            label_pixels <<< dimGrid, dimBlock, k * sizeof(uchar3) >>> (img.p(), img.width, img.height, img.pitch, (const uchar3 *) centroids.p(), k, (int *) labels.p(), labels.pitch / sizeof(int));
            CUDA_CHECK_RETURN(cudaPeekAtLastError());
//            cu::Image resimg(img.height, img.width, CV_8UC3);
//            retrieve_result_kmeans <<< dimGrid, dimBlock >>> (resimg.p(), img.width, img.height, img.pitch, (const uchar3 *) centroids.p(), k, (int *) labels.p(), labels.pitch / sizeof(int));
//            CUDA_CHECK_RETURN(cudaPeekAtLastError());
//            imshow("frame", resimg.download());
//            if (waitKey() == 27)
//                return;

            const float f_width = img.width / (float)dominant.width;
            const float f_height = img.height / (float)dominant.height;
            scale_dominant<<<dimGrid2, dimBlock2>>>((uint *)labels.p(), (uint *)dominant.p(), labels.pitch / sizeof(int), dominant.pitch / sizeof(uint), dominant.width, dominant.height, f_width, f_height, k);
            CUDA_CHECK_RETURN(cudaPeekAtLastError());

            // show intermediate results
            if (omp_get_thread_num() == 0) {
                static int show_count = 0;
                show_count++;
                if (show_count%5==0) {
                    retrieve_result_dominant <<< dimGrid2, dimBlock2 >>> (dom_img.p(), (uint *) dominant.p(), centroids.p(), dom_img.pitch, dominant.pitch / sizeof(uint), dominant.width, dominant.height, k);
                    CUDA_CHECK_RETURN(cudaPeekAtLastError());

                    imshow("frame", dom_img.download());
                    if (waitKey(1) == 27)
                        exit(0);
                }
            }
        }
    }
}

void dominant_colour::runCached(const std::string &cache_path) {
    cout << "retrieving cached file-list" << endl;
    BinReader reader(cache_path + "/" + cache_path_files);
    string cache_dst = reader.readString();
    cout << "original image-path: " << cache_dst << endl;

    int size = reader.readInt32();
    vector<string> files(size);
    for (auto &f : files)
        f = reader.readString();

    cout << "retrieving cached GlobalsMeans" << endl;
    BinReader reader2(cache_path + "/" + cache_path_global_means);
    const vector<Vec3b> global_means = reader2.readVector<Vec3b>();

    cout << "running dominant extraction" << endl;
    computeDominant(files, global_means);
}

void dominant_colour::run(const std::string &path, const std::string &cache_path) {
    vector<string> files = my_utils::listdir(path);
    {
        BinWriter writer(cache_path + "/" + cache_path_files);
        writer.appendString(path);
        writer.appendInt32(files.size());
        for (const auto &f : files)
            writer.appendString(f);
    }

    cout << "computing image dominant colours" << endl;
    boost::timer::cpu_timer watch;
    Mat means = dominant_colour::computeImageMeans(files, 128);
    cout << "done in " << watch.format() << endl;

    {
        BinWriter writer(cache_path + "/" + cache_path_image_means);
        writer.appendMat(means);
    }

    cout << "computing global means" << endl;
    const vector<Vec3b> global_means = cuda_kmeans(means, 256, 5);

    {
        BinWriter writer(cache_path + "/" + cache_path_global_means);
        writer.appendVector(global_means);
    }

    cout << "running dominant extraction" << endl;
    computeDominant(files, global_means);
}
