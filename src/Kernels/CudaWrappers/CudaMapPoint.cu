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
        if (mp)
            isEmpty = false;
        else {
            isEmpty = true;
            return;
        }
        mnId = mp->mnId;
        mWorldPos = mp->GetWorldPos();
        mfMaxDistance = mp->GetMaxDistance();
        mfMinDistance = mp->GetMinDistance();
        mNormalVector = mp->GetNormal();
        const cv::Mat& descriptor = mp->GetDescriptor();
        std::memcpy(mDescriptor, descriptor.ptr<uint8_t>(0), descriptor.cols * sizeof(uint8_t));
        mbBad = mp->isBad();
    }

    bool CudaMapPoint::isBad(){
        return mbBad;
    }
}