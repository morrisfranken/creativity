//
// Created by morris on 2/28/18.
//

#ifndef CREATIVITYPROJECT_UTILS_H
#define CREATIVITYPROJECT_UTILS_H

#include <vector>
#include <string>
#include <opencv2/core/mat.hpp>

namespace my_utils {
    std::vector<std::string> listdir(const std::string &path);
    void saveMat(const std::string &path, cv::Mat &m);
};


#endif //CREATIVITYPROJECT_UTILS_H
