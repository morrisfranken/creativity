/*
 * DeviceImage.h
 *
 *  Created on: 28 Aug 2016
 *      Author: morris
 */

#ifndef CUVECTOR_H_
#define CUVECTOR_H

#include <vector>
#include <array>
#include "cudaUtils.h"

namespace cu {
    template <class T>
    class Vector : public cuSharedPointer {
    public:
        std::size_t size;

        ~Vector();
        Vector();
        explicit Vector(std::size_t size);
        explicit Vector(const std::vector<T> &v);

        void upload(const std::vector<T> &v);

        std::vector<T> download() const;
    };
}

#endif /* CUVECTOR_H */
