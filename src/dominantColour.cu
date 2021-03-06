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
#include "cu/image.h"
#include "cu/vector.h"
#include "cu/kmeans.h"
#include "cu/utils.h"
#include "utils/utils.h"
#include "utils/binWriter.h"
#include "utils/binReader.h"

using namespace std;
using namespace cv;

constexpr int BLOCK_SIZE = 32;
constexpr char cache_path_files[]        = "files.raw";
constexpr char cache_path_image_means[]  = "image_means.raw";
constexpr char cache_path_global_means[] = "global_means.raw";

#define SKIP_MARGIN 30
constexpr int dom_margin = 1;
constexpr int ext_block_size = BLOCK_SIZE + dom_margin * 2;
constexpr int mem_limit = 49152 / (sizeof(int) * ext_block_size * ext_block_size); // 49152 is the amount of memory that fits in a block (for GTX_850M) TODO(morris) should make this dynamic somehow

const dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

namespace dominant_colour {
    cv::Mat computeImageMeans(const std::vector<std::string> &files, int k);
    void computeDominant(const std::vector<std::string> &files, const std::vector<cv::Vec3b> &centroids_, int swap_pos);
    void computeDominantTemporal(const std::vector<std::string> &files, const std::vector<cv::Vec3b> &centroids_, int swap_pos);
}

__global__ void scale_dominant_add(const kmeans_label *labels, uint *dominant, const int pitch_labels, const int pitch_dominant, const int width_dominant, const int height_dominant, const float f_width, const float f_height, const int k) {
    const int row  = blockIdx.y * blockDim.y + threadIdx.y;
    const int col  = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height_dominant & col < width_dominant) {	// make sure the thread is within the image
        const int idx_dominant = row * pitch_dominant + col * k;
        const int idx_labels = int(f_height * row) * pitch_labels + int(f_width * col);

        dominant[idx_dominant + labels[idx_labels]]++;
    }
}

__global__ void scale_dominant_subtract(const kmeans_label *labels, uint *dominant, const int pitch_labels, const int pitch_dominant, const int width_dominant, const int height_dominant, const float f_width, const float f_height, const int k) {
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

__global__ void transition_subtract(const uchar *src, uint *dst, const int pitch_src, const int pitch_dst, const int width, const int height) {
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

__global__ void transition_add_retrieve (const uchar *src, uint *dst, uchar *img, const int pitch_src, const int pitch_dst, const int pitch_img, const int width, const int height, const int count) {
    const int row  = blockIdx.y * blockDim.y + threadIdx.y;
    const int col  = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width & row < height) {
        const int idx_src = row * pitch_src + col * 3;
        const int idx_dst = row * pitch_dst + col * 3;
        const int idx_img = row * pitch_img + col * 3;

        dst[idx_dst + 0] += src[idx_src + 0];
        dst[idx_dst + 1] += src[idx_src + 1];
        dst[idx_dst + 2] += src[idx_src + 2];

        img[idx_img + 0] = dst[idx_dst + 0] / count;
        img[idx_img + 1] = dst[idx_dst + 1] / count;
        img[idx_img + 2] = dst[idx_dst + 2] / count;
    }
}

namespace dominant {
    inline void scale_subtract(const cu::Image &labels, cu::Image &dominant, const int &k) {
        const float f_width = labels.width / (float)dominant.width;
        const float f_height = labels.height / (float)dominant.height;
        const dim3 dimGrid((dominant.width + dimBlock.x - 1) / dimBlock.x, (dominant.height + dimBlock.y - 1) / dimBlock.y);
        scale_dominant_subtract <<< dimGrid, dimBlock >>> (labels.p<kmeans_label>(), dominant.p<uint>(), labels.pitch<kmeans_label>(), dominant.pitch<uint>(), dominant.width, dominant.height, f_width, f_height, k);
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
    }

    inline void scale_add(const cu::Image &labels, cu::Image &dominant, const int &k) {
        const float f_width = labels.width / (float)dominant.width;
        const float f_height = labels.height / (float)dominant.height;
        const dim3 dimGrid((dominant.width + dimBlock.x - 1) / dimBlock.x, (dominant.height + dimBlock.y - 1) / dimBlock.y);
        scale_dominant_add <<<dimGrid, dimBlock>>>(labels.p<kmeans_label>(), dominant.p<uint>(), labels.pitch<kmeans_label>(), dominant.pitch<uint>(), dominant.width, dominant.height, f_width, f_height, k);
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
    }

    inline void retrieve(cu::Image &dom_img, cu::Image &dominant, const cu::Vector<cv::Vec3b> &centroids, const int &k, const int &swap_pos) {
        const dim3 dimGrid((dominant.width + dimBlock.x - 1) / dimBlock.x, (dominant.height + dimBlock.y - 1) / dimBlock.y);
        retrieve_result_dominant2 <<< dimGrid, dimBlock >>> (dom_img.p(), dominant.p<uint>(), centroids.p(), dom_img.pitch(), dominant.pitch<uint>(), dominant.width, dominant.height, k, swap_pos);
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
    }
}

namespace transition {
    inline void subtract(const cu::Image &prev_img, cu::Image &total_trans) {
        const dim3 dimGrid((total_trans.width + dimBlock.x - 1) / dimBlock.x, (total_trans.height + dimBlock.y - 1) / dimBlock.y);
        transition_subtract <<< dimGrid, dimBlock >>> (prev_img.p(), total_trans.p<uint>(), prev_img.pitch(), total_trans.pitch<uint>(), total_trans.width, total_trans.height);
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
    }

    inline void add_retrieve(const cu::Image &dom_img, cu::Image &total_trans, cu::Image &dom_trans_img, const int count) {
        const dim3 dimGrid((total_trans.width + dimBlock.x - 1) / dimBlock.x, (total_trans.height + dimBlock.y - 1) / dimBlock.y);
        transition_add_retrieve <<< dimGrid, dimBlock >>> (dom_img.p(), total_trans.p<uint>(), dom_trans_img.p(), dom_img.pitch(), total_trans.pitch<uint>(), dom_trans_img.pitch(), total_trans.width, total_trans.height, count);
        CUDA_CHECK_RETURN(cudaPeekAtLastError());
    }
}

///////////////////////////////////

cv::Mat dominant_colour::computeImageMeans(const vector<string> &files, const int k) {
    Mat means = Mat(files.size(), k, CV_8UC3);

    #pragma omp parallel for ordered schedule(dynamic)
    for (int i = 0; i < files.size(); i++) {
        const cv::Mat img = cv::imread(files[i], cv::IMREAD_COLOR);

        #pragma omp ordered
        {
            if (i % 100 == 0)
                cout << i << " / " << files.size() << endl;

            const auto image_means = cu::kmeans(img, k, 5);
            mempcpy(means.ptr(i), image_means.data(), means.step);
        }
    }

    return means;
}

// @deprecated : use computeDominantTemporal
void dominant_colour::computeDominant(const vector<string> &files, const vector<Vec3b> &centroids_, const int swap_pos) {
    const cu::Vector<cv::Vec3b> centroids(centroids_);
    const int k = centroids_.size();
    assert(k < 256); // maximum k due to label_size limited to uchar = 256 centroids

    cu::Image dominant(1080 / 2, 1920 / 2, CV_32SC(k), true);
    cu::Image dom_img(dominant.height, dominant.width, CV_8UC3);

#pragma omp parallel for ordered schedule(dynamic)
    for (int i = 0; i < files.size(); i++) {
        const cv::Mat img_ = cv::imread(files[i], cv::IMREAD_COLOR);

#pragma omp ordered
        {
            if (i % 100 == 0)
                cout << i << " / " << files.size() << endl;

            cu::Image img(img_);
            cu::Image labels(img.height, img.width, label_cvtype);

            cu::label(img, labels, centroids, k);
            dominant::scale_add(labels, dominant, k);

            // show intermediate results
            if (omp_get_thread_num() == 0) {
                static int frame_count = 0;
                frame_count++;
                if (frame_count%1==0) {
//                    boost::timer::cpu_timer watch;
                    dominant::retrieve(dom_img, dominant, centroids, k, swap_pos);
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

/* Compute the running-dominant color for 5000 frames
 * On top of that, average results for the pas 100 frames
 */
void dominant_colour::computeDominantTemporal(const vector<string> &files, const vector<Vec3b> &centroids_, const int swap_pos) {
    const cu::Vector<cv::Vec3b> centroids(centroids_);
    const int k = centroids_.size();
    constexpr int temporal_size = 5000;
    constexpr int transition_size = 100; // fade-frames
    assert(k < 256); // since label is uchar (efficient memory consumption)

    cu::Image dominant(1080 / 2, 1920 / 2, CV_32SC(k), true);
    cu::Image total_trans(dominant.height, dominant.width, CV_32SC3, true);
    cu::Image dom_trans_img(dominant.height, dominant.width, CV_8UC3);

    vector<shared_ptr<cu::Image>> label_hist(temporal_size);
    vector<shared_ptr<cu::Image>> dom_hist(transition_size);

#pragma omp parallel for ordered schedule(dynamic)
    for (int i = 0; i < files.size(); i++) {
        const cv::Mat img_ = cv::imread(files[i], cv::IMREAD_COLOR);

#pragma omp ordered
        {
            const int temporal_idx = i % temporal_size;
            if (i % 100 == 0)
                cout << i << " / " << files.size() << endl;

            cu::Image img(img_);
            if (i >= temporal_size) {
                cu::Image &labels = *label_hist[temporal_idx];
                dominant::scale_subtract(labels, dominant, k);
            }
            label_hist[temporal_idx] = make_shared<cu::Image>(img.height, img.width, label_cvtype);
            cu::Image &labels = *label_hist[temporal_idx];

            cu::label(img, labels, centroids, k);
            dominant::scale_add(labels, dominant, k);

            // show results
            if (omp_get_thread_num() == 0) {
                static int frame_count = 0;
                static int frame_nr = 0;
                if (frame_count%1==0) {
                    const int transition_idx = frame_nr % transition_size;
                    if (frame_nr >= transition_size) {
                        cu::Image &prev_img = *dom_hist[transition_idx];
                        transition::subtract(prev_img, total_trans);
                    } else {
                        dom_hist[transition_idx] = make_shared<cu::Image>(dominant.height, dominant.width, CV_8UC3);
                    }
                    cu::Image &dom_img = *dom_hist[transition_idx];

//                    boost::timer::cpu_timer watch;
                    dominant::retrieve(dom_img, dominant, centroids, k, swap_pos);
//                    cudaDeviceSynchronize();
//                    cout << "done in " << watch.format() << endl;

                    const int count = min(frame_nr+1, transition_size);
                    transition::add_retrieve(dom_img, total_trans, dom_trans_img, count);

                    stringstream ss;
                    ss << "../results/dominant/" << setfill('0') << setw(5) << frame_nr << ".png";

                    const Mat frame = dom_trans_img.download();
                    imshow("frame", frame);
                    imwrite(ss.str(), frame);
                    if (waitKey(1) == 27)
                        exit(0);

                    frame_nr++;
                }
                frame_count++;
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

void dominant_colour::runDominantExtaction(const std::string &cache_path) {
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

void dominant_colour::computeColors(const std::string &path, const std::string &cache_path) {
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
    const vector<Vec3b> global_means = cu::kmeans(means, 256, 50);

    {
        BinWriter writer(cache_path + "/" + cache_path_global_means);
        writer.appendVector(global_means);
    }

    cout << "running dominant extraction" << endl;
    computeDominantTemporal(files, global_means, 0);
}
