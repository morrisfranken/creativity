//
// Created by morris on 4/24/18.
//

#ifndef CREATIVITYPROJECT_CUKMEANS_H
#define CREATIVITYPROJECT_CUKMEANS_H

#include <vector>
#include <opencv2/core.hpp>

#include "image.h"
#include "vector.h"

typedef uchar kmeans_label;  // change this based on K; K < 256 : uchar, K < 65536 : ushort
constexpr int label_cvtype = CV_8U;// TODO(morris) should be dynamic based on ^  like cv::Mat_<kmeans_label>().type()

namespace cu {
    void label(const cu::Image &img, cu::Image &labels, const cu::Vector<cv::Vec3b> &centroids, const int &k);
    std::vector<cv::Vec3b> kmeans(const cv::Mat &img_, int k, int iterations);

    void test_kmeans();
};


#endif //CREATIVITYPROJECT_CUKMEANS_H
