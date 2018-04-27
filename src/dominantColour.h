//
// Created by morris on 3/30/18.
//

#ifndef DOMINANTCOLOUR_H
#define DOMINANTCOLOUR_H

#include <string>
#include <map>
#include <opencv2/core/matx.hpp>

namespace dominant_colour {
    // computeColors the full pipeline and cache intermediate steps in the cache-dir
    void computeColors(const std::string &path, const std::string &cache_path);

    // load cached steps and execute the last step: computeDominant()
    void runDominantExtaction(const std::string &cache_path);
};


#endif //DOMINANTCOLOUR_H
