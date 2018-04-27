//
// Created by morris on 4/26/18.
//

#ifndef CREATIVITYPROJECT_CUSHAREDPOINTER_H
#define CREATIVITYPROJECT_CUSHAREDPOINTER_H

namespace cu {
    class SharedPointer {
    protected:
        unsigned char *data;
        short *uses;

    public:
        virtual ~SharedPointer();
        SharedPointer();
        SharedPointer(const SharedPointer& a);
        SharedPointer& operator=(const SharedPointer& in);

        template <class T = unsigned char> T *p() const {
            return (T *)data;
        }
    };
}


#endif //CREATIVITYPROJECT_CUSHAREDPOINTER_H
