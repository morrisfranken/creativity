//
// Created by morris on 3/30/18.
//

#ifndef DOMINANTCOLOUR_H
#define DOMINANTCOLOUR_H

#include <string>
#include <map>
#include <opencv2/core/matx.hpp>

namespace dominant_colour {
    cv::Mat computeImageMeans(const std::vector<std::string> &files, int k);
    void computeDominant(const std::vector<std::string> &files, const std::vector<cv::Vec3b> &centroids_, int swap_pos);
    void computeDominantTemporal(const std::vector<std::string> &files, const std::vector<cv::Vec3b> &centroids_, const int swap_pos);

    // run the full pipeline and cache intermediate steps in the cache-dir
    void run(const std::string &path, const std::string &cache_path);

    // load cached steps and execute the last step: computeDominant()
    void runCached(const std::string &cache_path);
};


#endif //DOMINANTCOLOUR_H
