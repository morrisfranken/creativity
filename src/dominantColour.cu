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
#include <iomanip>
#include <curand_kernel.h>

#include "dominantColour.h"
#include "average.h"
#include "cuUtils/cuImage.h"
#include "cuUtils/cuVector.h"
#include "utils/utils.h"
#include "utils/binWriter.h"
#include "utils/binReader.h"

using namespace std;
using namespace cv;

typedef uchar label_t;
constexpr int label_cvtype = CV_8U;// TODO(morris) should be dynamic based on ^  like cv::Mat_<label_t>().type()

constexpr int BLOCK_SIZE = 32;
constexpr char cache_path_files[] = "files.raw";
constexpr char cache_path_image_means[] = "image_means.raw";
constexpr char cache_path_global_means[] = "global_means.raw";

#pragma omp declare reduction(vec_int_plus  : std::vector<int>     : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<int>())) initializer(omp_priv = omp_orig)
#pragma omp declare reduction(vec_int3_plus : std::vector<Vec3i>   : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<Vec3i>())) initializer(omp_priv = omp_orig)
//#pragma omp declare reduction(arr_int_plus  : std::array<int, K>   : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<int>())) initializer(omp_priv = omp_orig)
//#pragma omp declare reduction(arr_int3_plus : std::array<Vec3i, K> : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<Vec3i>())) initializer(omp_priv = omp_orig)

__global__ void label_pixels(const uchar *img, const int width, int height, const int pitch_img, const uchar3 *centroids, const int k, label_t *labels, const int pitch_labels) {
    const int row  = blockIdx.y * blockDim.y + threadIdx.y;
    const int col  = blockIdx.x * blockDim.x + threadIdx.x;
    const int tidx = threadIdx.y * BLOCK_SIZE + threadIdx.x;

    extern __shared__ uchar3 sCentroid[]; // NOTE(morris) : size is equal to `k` and determined in calling function <<< grid, block, memsize >> (where memsize is in bytes)
    if (tidx < k) {
        sCentroid[tidx] = centroids[tidx]; // copy global centroids to local block memory
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
        labels[idx_label] = (label_t)best_i;
    }
}

__global__ void retrieve_result_kmeans(uchar *img, const int width, int height, const int pitch_img, const uchar3 *centroids, const int k, const label_t *labels, const int pitch_labels) {
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

__global__ void scale_dominant_add(const label_t *labels, uint *dominant, const int pitch_labels, const int pitch_dominant, const int width_dominant, const int height_dominant, const float f_width, const float f_height, const int k) {
    const int row  = blockIdx.y * blockDim.y + threadIdx.y;
    const int col  = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height_dominant & col < width_dominant) {	// make sure the thread is within the image
        const int idx_dominant = row * pitch_dominant + col * k;
        const int idx_labels = int(f_height * row) * pitch_labels + int(f_width * col);

        dominant[idx_dominant + labels[idx_labels]]++;
    }
}

__global__ void scale_dominant_subtract(const label_t *labels, uint *dominant, const int pitch_labels, const int pitch_dominant, const int width_dominant, const int height_dominant, const float f_width, const float f_height, const int k) {
    const int row  = blockIdx.y * blockDim.y + threadIdx.y;
    const int col  = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height_dominant & col < width_dominant) {	// make sure the thread is within the image
        const int idx_dominant = row * pitch_dominant + col * k;
        const int idx_labels = int(f_height * row) * pitch_labels + int(f_width * col);

        dominant[idx_dominant + labels[idx_labels]]--;
    }
}

/* Optimized version for retrieving dominant colours with a search-window to smooth the results
 * it first copies copies global dominant memory to local memory, including the additional padding as a result of the dom_margin.
 * since the local memory is rather limited, only a portion can be copied each time (it iterates in batches)
 * Once the memory is copied it will continue to compute the dominant colour for a given pixel adding all neighbouring dominant-counts to it
 *
 */
#define SKIP_MARGIN 30
constexpr int dom_margin = 1;
constexpr int ext_block_size = BLOCK_SIZE + dom_margin * 2;
constexpr int mem_limit = 49152 / (sizeof(int) * ext_block_size * ext_block_size); // 49152 is the amount of memory that fits in a block (for GTX_850M) TODO(morris) should make this dynamic somehow
__global__ void retrieve_result_dominant2(uchar *img, const uint *dominant, const uchar *centroids, const int pitch_img, const int pitch_dominant, const int width, const int height, const int k, const int skip_idx) {
    const int row  = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int col  = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    __shared__ uint local_dominant[mem_limit * (ext_block_size * ext_block_size)];

    int dom_idx = 0;
    uint dom_value = 0;
    const int iterations = (k + mem_limit -1) / mem_limit;
    for (int gi = 0; gi < iterations; gi++) {
        constexpr int local_iterations = (ext_block_size * ext_block_size + BLOCK_SIZE * BLOCK_SIZE - 1) / (BLOCK_SIZE * BLOCK_SIZE);
        for (int li = 0; li < local_iterations; li++) {
            const int tidx = BLOCK_SIZE * BLOCK_SIZE * li + threadIdx.y * BLOCK_SIZE + threadIdx.x;
            if (tidx < ext_block_size * ext_block_size) {
                const int local_row  = tidx / ext_block_size;
                const int local_col  = tidx - local_row * ext_block_size;    // cheap mod
                const int local_idx  = mem_limit  * tidx; //(local_row * ext_block_size + local_col);
                const int global_row = blockIdx.y * BLOCK_SIZE + local_row - dom_margin;
                const int global_col = blockIdx.x * BLOCK_SIZE + local_col - dom_margin;
                const int global_idx = global_row * pitch_dominant + global_col * k + mem_limit * gi;

                if (global_row >= 0 & global_row < height & global_col >= 0 & global_col < width) {
                    for (int i = 0; i < mem_limit; i++)
                        local_dominant[local_idx + i] = dominant[global_idx + i];
                } else {
                    for (int i = 0; i < mem_limit; i++)
                        local_dominant[local_idx + i] = 0; // fill remaining elements with 0
                }
            }
        }
        __syncthreads();

        const int maxi = min(k - gi * mem_limit, mem_limit);
        for (int i = 0; i < maxi; i++) {
#if SKIP_MARGIN // for testing to remove colours from consideration
            if ((gi * mem_limit + i) < skip_idx)
                continue;
#endif
            uint dom_y_value = 0;
            for (int y = 0; y <= dom_margin*2; y++) {
                for (int x = 0; x <= dom_margin*2; x++) {
                    const int newrow = threadIdx.y + y;
                    const int newcol = threadIdx.x + x;
                    dom_y_value += local_dominant[mem_limit * ((newrow) * ext_block_size + newcol) + i];
                }
            }

            if (dom_y_value > dom_value) {
                dom_value = dom_y_value;
                dom_idx = gi * mem_limit + i;
            }
        }
    }

    const int idx_img = row * pitch_img + col * 3;
    img[idx_img + 0] = centroids[dom_idx * 3 + 0];
    img[idx_img + 1] = centroids[dom_idx * 3 + 1];
    img[idx_img + 2] = centroids[dom_idx * 3 + 2];
}

///////////////////////////////////

void compute_centroid_cpu(const Mat &img, const cu::Image &labels_, cu::Vector<cv::Vec3b> &centroids_, const int k) {
    const Mat labels = labels_.download();
    std::vector<Vec3i> centroidsi(k);
    std::vector<Vec3b> centroidsb(k);
    std::vector<int> count(k);

    // omp seem to only improve ~15% in local tests, not worth it considering vs image loading
//#pragma omp parallel for reduction(vec_int3_plus : centroidsi), reduction(vec_int_plus : count)
    for (int y = 0; y < labels.rows; y++) {
        const Vec3b *row_img = reinterpret_cast<const Vec3b *>(img.ptr(y));
        const label_t *row_labels = reinterpret_cast<const label_t *>(labels.ptr(y));
        for (int x = 0; x < labels.cols; x++) {
            const label_t &label = row_labels[x];
            centroidsi[label] += row_img[x];
            count[label]++;
        }
    }

    for (int i = 0; i < k; i++) {
        if (count[i])
            centroidsb[i] = centroidsi[i] / count[i];
        else // centroid is already taken, create a random new one
            centroidsb[i] = img.at<Vec3b>(rand() % (img.cols * img.rows));
    }

    centroids_.upload(centroidsb);
}

vector<Vec3b> cuda_kmeans(const Mat &img_, const int k, const int iterations) {
    vector<Vec3b> centroids_(k);

    for (int i = 0; i < k; i++) {
        centroids_[i] = img_.at<Vec3b>(rand() % img_.rows, rand() % img_.cols); // TODO(morris) :: pick better initial random centers
    }

    cu::Image img(img_);
    cu::Image labels(img.height, img.width, label_cvtype);
    cu::Vector<cv::Vec3b> centroids(centroids_);

    const dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    const dim3 dimGrid((img.width + dimBlock.x - 1) / dimBlock.x, (img.height + dimBlock.y - 1) / dimBlock.y);

    for (int i = 0; i < iterations; i++) {
        label_pixels <<< dimGrid, dimBlock, k * sizeof(uchar3) >>> (img.p(), img.width, img.height, img.pitch(), (const uchar3 *) centroids.p(), k, (label_t *) labels.p(), labels.pitch() / sizeof(label_t));
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
        compute_centroid_cpu(img_, labels, centroids, k);
    }
    return centroids.download();
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

void dominant_colour::computeDominant(const vector<string> &files, const vector<Vec3b> &centroids_, const int swap_pos) {
    const cu::Vector<cv::Vec3b> centroids(centroids_);
    const int k = centroids_.size();
    assert(k < 256); // maximum k due to label_size limited to uchar = 256 centroids

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
            cu::Image labels(img.height, img.width, label_cvtype);

            const dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
            const dim3 dimGrid((img.width + dimBlock.x - 1) / dimBlock.x, (img.height + dimBlock.y - 1) / dimBlock.y);

            label_pixels <<< dimGrid, dimBlock, k * sizeof(uchar3) >>> (img.p(), img.width, img.height, img.pitch(), (const uchar3 *) centroids.p(), k, (label_t *) labels.p(), labels.pitch() / sizeof(label_t));
            CUDA_CHECK_RETURN(cudaPeekAtLastError());

            const float f_width = img.width / (float)dominant.width;
            const float f_height = img.height / (float)dominant.height;
            scale_dominant_add <<<dimGrid2, dimBlock2>>>((label_t *)labels.p(), (uint *)dominant.p(), labels.pitch() / sizeof(label_t), dominant.pitch() / sizeof(uint), dominant.width, dominant.height, f_width, f_height, k);
            CUDA_CHECK_RETURN(cudaPeekAtLastError());

            // show intermediate results
            if (omp_get_thread_num() == 0) {
                static int frame_count = 0;
                frame_count++;
                if (frame_count%1==0) {

//                    boost::timer::cpu_timer watch;
                    retrieve_result_dominant2 <<< dimGrid2, dimBlock2 >>> (dom_img.p(), (uint *) dominant.p(), centroids.p(), dom_img.pitch(), dominant.pitch() / sizeof(uint), dominant.width, dominant.height, k, swap_pos);
                    CUDA_CHECK_RETURN(cudaPeekAtLastError());
//                    cudaDeviceSynchronize();
//                    cout << "done in " << watch.format() << endl;

                    stringstream ss;
                    ss << "../results/dominant/" << setfill('0') << setw(4) << frame_count << ".png";

                    const Mat frame = dom_img.download();
                    imshow("frame", frame);
                    imwrite(ss.str(), frame);
                    if (waitKey(1) == 27)
                        exit(0);
                }
            }
        }
    }
}

void dominant_colour::computeDominantTemporal(const vector<string> &files, const vector<Vec3b> &centroids_, const int swap_pos) {
    const cu::Vector<cv::Vec3b> centroids(centroids_);
    const int k = centroids_.size();
    constexpr int temporal_size = 5000;

    cu::Image dominant(1080 / 2, 1920 / 2, CV_32SC(k), true);
    cu::Image dom_img(dominant.height, dominant.width, CV_8UC3);
    const dim3 dimBlock2(BLOCK_SIZE, BLOCK_SIZE, 1);
    const dim3 dimGrid2((dominant.width + dimBlock2.x - 1) / dimBlock2.x, (dominant.height + dimBlock2.y - 1) / dimBlock2.y);

    vector<shared_ptr<cu::Image>> label_hist(temporal_size);

#pragma omp parallel for ordered schedule(dynamic)
    for (int i = 0; i < files.size(); i++) {
        const cv::Mat img_ = cv::imread(files[i], cv::IMREAD_COLOR);

#pragma omp ordered
        {
            const int temporal_idx = i % temporal_size;
            if (i % 100 == 0)
                cout << i << " / " << files.size() << endl;

            cu::Image img(img_);
            if (i > temporal_size) {
                cu::Image &labels = *label_hist[temporal_idx];
                const float f_width = labels.width / (float)dominant.width;
                const float f_height = labels.height / (float)dominant.height;
                scale_dominant_subtract <<< dimGrid2, dimBlock2 >>> ((label_t *) label_hist[temporal_idx]->p(), (uint *) dominant.p(), label_hist[temporal_idx]->pitch() / sizeof(label_t), dominant.pitch() / sizeof(uint), dominant.width, dominant.height, f_width, f_height, k);
                CUDA_CHECK_RETURN(cudaPeekAtLastError());
            }
            label_hist[temporal_idx] = make_shared<cu::Image>(img.height, img.width, label_cvtype);
            cu::Image &labels = *label_hist[temporal_idx];

            const float f_width = img.width / (float)dominant.width;
            const float f_height = img.height / (float)dominant.height;
            const dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
            const dim3 dimGrid((img.width + dimBlock.x - 1) / dimBlock.x, (img.height + dimBlock.y - 1) / dimBlock.y);

            label_pixels <<< dimGrid, dimBlock, k * sizeof(uchar3) >>> (img.p(), img.width, img.height, img.pitch(), (const uchar3 *) centroids.p(), k, (label_t *) labels.p(), labels.pitch() / sizeof(label_t));
            CUDA_CHECK_RETURN(cudaPeekAtLastError());
            scale_dominant_add <<<dimGrid2, dimBlock2>>>((label_t *)labels.p(), (uint *)dominant.p(), labels.pitch() / sizeof(label_t), dominant.pitch() / sizeof(uint), dominant.width, dominant.height, f_width, f_height, k);
            CUDA_CHECK_RETURN(cudaPeekAtLastError());

            // show intermediate results
            if (omp_get_thread_num() == 0) {
                static int frame_count = 0;
                frame_count++;
                if (frame_count%1==0) {

//                    boost::timer::cpu_timer watch;
                    retrieve_result_dominant2 <<< dimGrid2, dimBlock2 >>> (dom_img.p(), (uint *) dominant.p(), centroids.p(), dom_img.pitch(), dominant.pitch() / sizeof(uint), dominant.width, dominant.height, k, swap_pos);
                    CUDA_CHECK_RETURN(cudaPeekAtLastError());
//                    cudaDeviceSynchronize();
//                    cout << "done in " << watch.format() << endl;

                    stringstream ss;
                    ss << "../results/dominant/" << setfill('0') << setw(4) << frame_count << ".png";

                    const Mat frame = dom_img.download();
                    imshow("frame", frame);
                    imwrite(ss.str(), frame);
                    if (waitKey(1) == 27)
                        exit(0);
                }
            }
        }
    }
}

Vec3b BGR2HSV(Vec3b &in) {
    Mat hsv;
    Mat rgb(1,1, CV_8UC3, in);
    cvtColor(rgb, hsv, CV_BGR2HSV);
    return hsv.at<Vec3b>(0);
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
    vector<Vec3b> global_means = reader2.readVector<Vec3b>();

    int skip_pos = 0;
#if SKIP_MARGIN
    for (int i = 0; i < global_means.size(); i++) {
        Vec3b hsv = BGR2HSV(global_means[i]);
        if (hsv[1] < SKIP_MARGIN || hsv[2] < SKIP_MARGIN) {
            std::swap(global_means[skip_pos++], global_means[i]);
        }
    }
    cout << "skipped colours: " << skip_pos << endl;

    Mat centroids_img(16,16, CV_8UC3, (void *)global_means.data());
    cv::resize(centroids_img, centroids_img, Size(16*40, 16*40), 0, 0, INTER_NEAREST);
    imwrite("centroids.png", centroids_img);
#endif

    cout << "running dominant extraction" << endl;
    computeDominantTemporal(files, global_means, skip_pos);
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
    const vector<Vec3b> global_means = cuda_kmeans(means, 256, 50);

    {
        BinWriter writer(cache_path + "/" + cache_path_global_means);
        writer.appendVector(global_means);
    }

    cout << "running dominant extraction" << endl;
    computeDominant(files, global_means, 0);
}
