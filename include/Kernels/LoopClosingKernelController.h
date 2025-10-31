#ifndef LOOP_CLOSING_KERNEL_CONTROLLER_H
#define LOOP_CLOSING_KERNEL_CONTROLLER_H

#include "CudaWrappers/CudaKeyFrame.h"
// #include "CudaKeyFrameStorage.h"
#include "CudaUtils.h"
#include "SearchAndFuseKernel.h"
#include "SearchByProjectionKernel.h"
#include "SearchByBoWKernel.h"
#include "SearchAndFuseKernel.h"
#include "SearchForTriangulationKernel.h"
#include <memory> 

using namespace std;

class LoopClosingKernelController{
public:
    

    static void shutdownKernels();

    static void launchFuseKernel(
        std::vector<ORB_SLAM3::KeyFrame*> connectedKFs, vector<Sophus::Sim3f> connectedScws, const float th,
        std::vector<ORB_SLAM3::MapPoint*> &vpMapPoints,  
        std::vector<ORB_SLAM3::MapPoint*> &validMapPoints, int* bestDists, int* bestIdxs
    );

    static void launchSearchByProjectionKernel(ORB_SLAM3::KeyFrame* pKF, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints,
                                    Sophus::Sim3<float> &Scw, const std::vector<ORB_SLAM3::KeyFrame*> &vpPointsKFs, std::vector<ORB_SLAM3::MapPoint*> &vpMatched, std::vector<ORB_SLAM3::KeyFrame*> &vpMatchedKF, int th, float ratioHamming,
                                    Sophus::Sim3<float> &Scw1, std::vector<ORB_SLAM3::MapPoint*> &vpMatched1, int th1, float ratioHamming1,
                                    int &numProjMatches, int &numProjOptMatches);
    static int launchSearchByBoWKernel(ORB_SLAM3::KeyFrame *pKF1, ORB_SLAM3::KeyFrame *pKF2, vector<ORB_SLAM3::MapPoint *> &vpMatches12);
    static void launchWarmUp();
        
    

private:
    static std::unique_ptr<SearchByProjectionKernel> mpSearchByProjectionKernel;
    static std::unique_ptr<SearchByBoWKernel> mpSearchByBoWKernel;
    static std::unique_ptr<SearchAndFuseKernel> mpSearchAndFuseKernel;

};

#endif
