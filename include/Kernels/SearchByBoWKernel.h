#ifndef SEARCH_BY_BOW_KERNEL_H
#define SEARCH_BY_BOW_KERNEL_H

#include "KernelInterface.h"
#include <iostream>
#include "CudaWrappers/CudaMapPoint.h"
#include "CudaWrappers/CudaKeyFrame.h"
#include "CudaUtils.h"
#include <Eigen/Core>
#include <csignal> 



class SearchByBoWKernel{

    public:
        void initialize();
        void shutdown();
        int launch(ORB_SLAM3::KeyFrame *pKF1, ORB_SLAM3::KeyFrame *pKF2, vector<ORB_SLAM3::MapPoint *> &vpMatches12);
        
        void origSearchByBoW(ORB_SLAM3::KeyFrame *pKF1, ORB_SLAM3::KeyFrame *pKF2, vector<ORB_SLAM3::MapPoint *> &vpMatches12);
        int origDescriptorDistance(const cv::Mat &a, const cv::Mat &b);

    private:
        bool memory_is_initialized;
               
};

#endif