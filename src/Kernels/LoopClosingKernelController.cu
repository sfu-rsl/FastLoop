#include "Kernels/LoopClosingKernelController.h"

// #define DEBUG


std::unique_ptr<SearchByProjectionKernel> LoopClosingKernelController::mpSearchByProjectionKernel = std::make_unique<SearchByProjectionKernel>();


void LoopClosingKernelController::shutdownKernels() {
    mpSearchByProjectionKernel->shutdown();
    CudaUtils::shutdown();
    cudaDeviceSynchronize();
}


void LoopClosingKernelController::launchFuseKernel(
        std::vector<ORB_SLAM3::KeyFrame*> connectedKFs, vector<Sophus::Sim3f> connectedScws, const float th,
        std::vector<ORB_SLAM3::MapPoint*> &vpMapPoints,  
        std::vector<ORB_SLAM3::MapPoint*> &validMapPoints, int* bestDists, int* bestIdxs
    )
{


    return;

}

void LoopClosingKernelController::launchSearchByProjectionKernel(ORB_SLAM3::KeyFrame* pKF, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints,
                                Sophus::Sim3<float> &Scw, const std::vector<ORB_SLAM3::KeyFrame*> &vpPointsKFs, std::vector<ORB_SLAM3::MapPoint*> &vpMatched, std::vector<ORB_SLAM3::KeyFrame*> &vpMatchedKF, int th, float ratioHamming,
                                Sophus::Sim3<float> &Scw1, std::vector<ORB_SLAM3::MapPoint*> &vpMatched1, int th1, float ratioHamming1,
                                int &numProjMatches, int &numProjOptMatches)
{

    // SearchByProjectionKernel kernel;
    // kernel.mergedlaunch(pKF, vpPoints,
    //                     Scw, vpPointsKFs, vpMatched, vpMatchedKF, th, ratioHamming,
    //                     Scw1, vpMatched1, th1, ratioHamming1,
    //                     numProjMatches, numProjOptMatches);
    mpSearchByProjectionKernel->mergedlaunch(pKF, vpPoints,
                        Scw, vpPointsKFs, vpMatched, vpMatchedKF, th, ratioHamming,
                        Scw1, vpMatched1, th1, ratioHamming1,
                        numProjMatches, numProjOptMatches);

    return;

}