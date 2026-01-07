#ifndef CUDA_CAMERA_H
#define CUDA_CAMERA_H

#include <Eigen/Core>

struct CudaCamera {
    bool isAvailable;
    float mvParameters[8];
    Eigen::Matrix3f toK;
};


#endif