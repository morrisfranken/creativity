//
// Created by morris on 4/24/18.
//
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <boost/timer/timer.hpp>
#include <stopwatch.h>

#include "kmeans.h"
#include "image.h"
#include "vector.h"
#include "utils.h"

using namespace std;
using namespace cv;

constexpr int BLOCK_SIZE = 32;
const dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

#pragma omp declare reduction(vec_int_plus  : std::vector<int>     : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<int>())) initializer(omp_priv = omp_orig)
#pragma omp declare reduction(vec_int3_plus : std::vector<Vec3i>   : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<Vec3i>())) initializer(omp_priv = omp_orig)
//#pragma omp declare reduction(arr_int_plus  : std::array<int, K>   : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<int>())) initializer(omp_priv = omp_orig)
//#pragma omp declare reduction(arr_int3_plus : std::array<Vec3i, K> : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<Vec3i>())) initializer(omp_priv = omp_orig)

__global__ void label_pixels(const uchar *img, const int width, int height, const int pitch_img, const uchar3 *centroids, const int k, kmeans_label *labels, const int pitch_labels) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int tidx = threadIdx.y * BLOCK_SIZE + threadIdx.x;

    extern __shared__ uchar3 sCentroid[]; // NOTE(morris) : size is equal to `k` and determined in calling function <<< grid, block, memsize >> (where memsize is in bytes)
    if (tidx < k) {
        sCentroid[tidx] = centroids[tidx]; // copy global centroids to local block memory
    }
    __syncthreads();

    if (row < height & col < width) {    // make sure the thread is within the image
        const int idx_img = row * pitch_img + col * 3;
        const int idx_label = row * pitch_labels + col;

        const uchar3 pix = *(uchar3 *) (img + idx_img);

        int best_i = 0;
        int best_val = 0x0fffffff;

        // compute the closest centroid
        for (int i = 0; i < k; i++) {
            const auto &centroid = sCentroid[i];
            const int dx = pix.x - centroid.x;
            const int dy = pix.y - centroid.y;
            const int dz = pix.z - centroid.z;
            const int dist = dx * dx + dy * dy + dz * dz;
//            printf("pix[%d,%d] :: [%d] [%d,%d,%d] -- [%d,%d,%d] = %d\n", row, col, i, pix.x, pix.y, pix.z, centroid.x, centroid.y, centroid.z, dist);
            if (dist < best_val) {
                best_val = dist;
                best_i = i;
            }
        }
        labels[idx_label] = (kmeans_label) best_i;
    }
}


__global__ void retrieve_result_kmeans(uchar *img, const int width, int height, const int pitch_img, const uchar3 *centroids, const int k, const kmeans_label *labels, const int pitch_labels) {
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

__global__ void compute_totals(const uchar *img, const int width, int height, const int pitch_img, uint *global_centroid_count, const int k, kmeans_label *labels, const int pitch_labels) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int tidx = threadIdx.y * BLOCK_SIZE + threadIdx.x;

    extern __shared__ uint local_centroid_count[]; // r/g/b/count
    if (tidx < k*4) { // this will be a problem for k > 256
        local_centroid_count[tidx] = 0;
    }
    __syncthreads();

    if (row < height & col < width) {    // make sure the thread is within the image
        const int idx_img = row * pitch_img + col * 3;
        const int idx_label = row * pitch_labels + col;

        const int label = labels[idx_label];
        atomicAdd(local_centroid_count + label * 4 + 0, img[idx_img + 0]);
        atomicAdd(local_centroid_count + label * 4 + 1, img[idx_img + 1]);
        atomicAdd(local_centroid_count + label * 4 + 2, img[idx_img + 2]);
        atomicAdd(local_centroid_count + label * 4 + 3, 1);
    }

    __syncthreads();

    // potential speedup; reserve global memory for every block to store `local_centroid_count`, and perform a sum in the end over all these arrays to avoid the atomicAdd
    if (tidx < k*4) { // this will be a problem for k > 256
        atomicAdd(global_centroid_count + tidx, local_centroid_count[tidx]);
    }
}

__device__ inline int roundi(const uint &value, const uint &devision) {
    return (value + devision / 2) / devision;
}

__global__ void compute_centroids(uint *global_centroid_count, uchar *centroids) {
    const uint count = global_centroid_count[threadIdx.x * 4 + 3];
//        printf("%d, %d, %d, %d\n", global_centroid_count[threadIdx.x * 4 + 0], global_centroid_count[threadIdx.x * 4 + 1], global_centroid_count[threadIdx.x * 4 + 2], global_centroid_count[threadIdx.x * 4 + 3]);
    if (count > 0) {
        centroids[threadIdx.x * 3 + 0] = roundi(global_centroid_count[threadIdx.x * 4 + 0], count);
        centroids[threadIdx.x * 3 + 1] = roundi(global_centroid_count[threadIdx.x * 4 + 1], count);
        centroids[threadIdx.x * 3 + 2] = roundi(global_centroid_count[threadIdx.x * 4 + 2], count);
    } else {
        centroids[threadIdx.x * 3 + 0] += threadIdx.x; // not really random... but close enough for this purpose
        centroids[threadIdx.x * 3 + 1] += threadIdx.x;
        centroids[threadIdx.x * 3 + 2] += threadIdx.x;
    }
}


namespace cu {
    void label(const cu::Image &img, cu::Image &labels, const cu::Vector<cv::Vec3b> &centroids, const int &k) {
        const dim3 dimGrid((img.width + dimBlock.x - 1) / dimBlock.x, (img.height + dimBlock.y - 1) / dimBlock.y);
        label_pixels <<< dimGrid, dimBlock, k * sizeof(uchar3) >>> (img.p(), img.width, img.height, img.pitch(), centroids.p<uchar3>(), k, labels.p<kmeans_label>(), labels.pitch<kmeans_label>());
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
    }

    void compute_centroid_gpu(const cu::Image &img, const cu::Image &labels, cu::Vector<cv::Vec3b> &centroids, cu::Vector<cv::Vec4i> &centroid_count, const int &k) {
        centroid_count.mem0();

        const dim3 dimGrid((img.width + dimBlock.x - 1) / dimBlock.x, (img.height + dimBlock.y - 1) / dimBlock.y);
        compute_totals <<< dimGrid, dimBlock, k * sizeof(uint4) >>> (img.p(), img.width, img.height, img.pitch(), centroid_count.p<uint>(), k, labels.p<kmeans_label>(), labels.pitch<kmeans_label>());
        CUDA_CHECK_RETURN(cudaPeekAtLastError());

        compute_centroids <<< 1, k >>>(centroid_count.p<uint>(), centroids.p());
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
    }

    void compute_centroid_cpu(const Mat &img, const cu::Image &labels_, cu::Vector<cv::Vec3b> &centroids_, const int &k) {
        const Mat labels = labels_.download();
        std::vector<Vec3i> centroidsi(k);
        std::vector<Vec3b> centroidsb(k);
        std::vector<int> count(k);

        // omp seem to only improve ~15% in local tests, not worth it considering vs image loading
        //#pragma omp parallel for reduction(vec_int3_plus : centroidsi), reduction(vec_int_plus : count)
        for (int y = 0; y < labels.rows; y++) {
            const Vec3b *row_img = reinterpret_cast<const Vec3b *>(img.ptr(y));
            const kmeans_label *row_labels = reinterpret_cast<const kmeans_label *>(labels.ptr(y));
            for (int x = 0; x < labels.cols; x++) {
                const kmeans_label &label = row_labels[x];
                centroidsi[label] += row_img[x];
                count[label]++;
            }
        }

        for (int i = 0; i < k; i++) {
//            printf("%d, %d, %d, %d\n", centroidsi[i][0], centroidsi[i][1], centroidsi[i][2], count[i]);
            if (count[i])
                centroidsb[i] = centroidsi[i] / count[i];
            else // centroid is already taken, create a random new one
                centroidsb[i] = img.at<Vec3b>(rand() % img.rows, rand() % img.cols);
        }

        centroids_.upload(centroidsb);
    }
}

std::vector<cv::Vec3b> cu::kmeans(const cv::Mat &img_, const int k, const int iterations) {
    vector<Vec3b> centroids_(k);

    for (int i = 0; i < k; i++) {
        centroids_[i] = img_.at<Vec3b>(rand() % img_.rows, rand() % img_.cols); // TODO(morris) :: pick better initial random centers
    }

    cu::Image img(img_);
    cu::Image labels(img.height, img.width, label_cvtype);
    cu::Vector<cv::Vec3b> centroids(centroids_);
//    cu::Vector<cv::Vec3b> centroids_cpu(centroids_);
    cu::Vector<cv::Vec4i> centroid_counts(k);

//    boost::timer::cpu_timer watch1;
    for (int i = 0; i < iterations; i++) {
        cu::label(img, labels, centroids, k);
        cu::compute_centroid_gpu(img_, labels, centroids, centroid_counts, k);
    }
//    cout << "gpu:" << watch1.format() << endl;
//
//    boost::timer::cpu_timer watch2;
//    for (int i = 0; i < iterations; i++) {
//        cu::label(img, labels, centroids_cpu, k);
//        cu::compute_centroid_cpu(img_, labels, centroids_cpu, k);
//    }
//    cout << "cpu:" << watch2.format() << endl;
//
//    auto res_gpu = centroids.download();
//    auto res_cpu = centroids_cpu.download();
//
//    for (auto &a : res_gpu)
//        cout << a << ", ";
//    cout << endl;
//
//    for (auto &a : res_cpu)
//        cout << a << ", ";
//    cout << endl;

    return centroids.download();
}

void cu::test_kmeans() {
    Mat m = Mat(500,500, CV_8UC3);
    for (int i = 0; i < (m.cols * m.rows); i++) {
        m.at<Vec3b>(i) = Vec3b(rand() % 256,rand() % 256, rand() % 256);
    }

    auto res = kmeans(m,5,1000);
//    for (const auto &r : res)
//        cout << r << ", ";
}

