/*
 * DeviceImage.h
 *
 *  Created on: 28 Aug 2016
 *      Author: morris
 */

#ifndef CUVECTOR_H_
#define CUVECTOR_H_

#include <vector>
#include <array>
#include "shared_pointer.h"

namespace cu {
    template <class T>
    class Vector : public SharedPointer {
    public:
        std::size_t size;

        ~Vector();
        Vector();
        explicit Vector(std::size_t size, bool fill0=false);
        explicit Vector(const std::vector<T> &v);

        void upload(const std::vector<T> &v);

        std::vector<T> download() const;
        void mem0();
    };
}

#endif /* CUVECTOR_H */
