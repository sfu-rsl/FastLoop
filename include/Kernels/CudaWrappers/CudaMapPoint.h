#ifndef CUDA_MAPPOINT_H
#define CUDA_MAPPOINT_H

#include "MapPoint.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#define MAX_NUM_OBSERVATIONS 200

namespace LOOP_CLOSING_DATA_WRAPPER {

class CudaKeyFrame;

class CudaMapPoint {
    public:
        CudaMapPoint();
        CudaMapPoint(ORB_SLAM3::MapPoint* mp);
        bool isBad();
    
    public:
        // For creating empty mapPoints instead of using null ptr
        bool isEmpty;

    public:
        long unsigned int mnId;
        bool mbBad;
        int nObs;
        int mObservations_size;
        int mObservations_leftIdx[200];
        int mObservations_rightIdx[200];
        Eigen::Vector3f mWorldPos;
        float mfMaxDistance;
        float mfMinDistance;
        Eigen::Vector3f mNormalVector;
        uint8_t mDescriptor[32];
        CudaKeyFrame* mObsKeyFrames[40];
    };
}

#endif // CUDA_MAPPOINT_H