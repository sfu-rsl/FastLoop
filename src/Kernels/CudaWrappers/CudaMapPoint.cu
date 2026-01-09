#include "Kernels/CudaWrappers/CudaMapPoint.h"
#include <iostream>
#include <map>
#include <tuple>

#ifdef TIME_MEASURMENT
#define TIMESTAMP_PRINT(msg) std::cout << "TimeStamp [CudaFrame]: " << msg << std::endl
#else
#define TIMESTAMP_PRINT(msg) do {} while (0)
#endif

namespace LOOP_CLOSING_DATA_WRAPPER
{
    CudaMapPoint::CudaMapPoint() {
        isEmpty = true;
    }

    CudaMapPoint::CudaMapPoint(ORB_SLAM3::MapPoint* mp) {
        isEmpty = false;
        mnId = mp->mnId;

        // cv::Mat descriptor;
        mp->AssignWorldPos_MaxD_MinD_Normal_Descriptor(
            mWorldPos,
            mfMaxDistance,
            mfMinDistance,
            mNormalVector,
            mDescriptor
        );
        
        // mWorldPos = mp->GetWorldPos();
        // mfMaxDistance = mp->GetMaxDistance();
        // mfMinDistance = mp->GetMinDistance();
        // mNormalVector = mp->GetNormal();
        // const cv::Mat& descriptor = mp->GetDescriptor();
        // cout << descriptor.cols << "    " << descriptor.cols * sizeof(uint8_t) << endl;
        // std::memcpy(mDescriptor, descriptor.ptr<uint8_t>(0), descriptor.cols * sizeof(uint8_t));
        mbBad = false;
    }

    bool CudaMapPoint::isBad(){
        return mbBad;
    }
}