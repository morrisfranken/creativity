//
// Created by morris on 4/26/18.
//

#include "shared_pointer.h"
#include "utils.h"

namespace cu {
    SharedPointer::~SharedPointer() {
        if (--(*uses) <= 0 && data != NULL) {
            CUDA_CHECK_RETURN(cudaFree(data));
            delete uses;
        }
    }

    SharedPointer::SharedPointer() : data(NULL), uses(new short(1)) {}

    SharedPointer::SharedPointer(const SharedPointer &a) : data(a.data), uses(a.uses) {
        (*uses)++;
    }

    SharedPointer &SharedPointer::operator=(const SharedPointer &in) {
        if (this != &in) { // prevent self assignment
            (*in.uses)++;
            if (--(*uses) <= 0 && data != NULL) {
                CUDA_CHECK_RETURN(cudaFree(data));
                delete uses;
            }
            uses = in.uses;
            data = in.data;
        }

        return *this;
    }
}