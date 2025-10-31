#include "Kernels/LoopClosingKernelController.h"

// #define DEBUG


std::unique_ptr<SearchByProjectionKernel> LoopClosingKernelController::mpSearchByProjectionKernel = std::make_unique<SearchByProjectionKernel>();
std::unique_ptr<SearchByBoWKernel> LoopClosingKernelController::mpSearchByBoWKernel = std::make_unique<SearchByBoWKernel>();
std::unique_ptr<SearchAndFuseKernel> LoopClosingKernelController::mpSearchAndFuseKernel = std::make_unique<SearchAndFuseKernel>();

__global__ void warmupKernel() {}


void LoopClosingKernelController::shutdownKernels() {
    mpSearchByProjectionKernel->shutdown();
    mpSearchAndFuseKernel->shutdown();
    mpSearchByBoWKernel->shutdown();
    CudaUtils::shutdown();
    cudaDeviceSynchronize();
}


void LoopClosingKernelController::launchSearchAndFuseKernel(
        std::vector<ORB_SLAM3::KeyFrame*> connectedKFs, vector<Sophus::Sim3f> connectedScws, const float th,
        std::vector<ORB_SLAM3::MapPoint*> &vpMapPoints,  
        std::vector<ORB_SLAM3::MapPoint*> &validMapPoints, int* bestDists, int* bestIdxs
    )
{

    mpSearchAndFuseKernel->launch(connectedKFs, connectedScws, th,
                        vpMapPoints,
                        validMapPoints, bestDists, bestIdxs);
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

int LoopClosingKernelController::launchSearchByBoWKernel(ORB_SLAM3::KeyFrame *pKF1, ORB_SLAM3::KeyFrame *pKF2, vector<ORB_SLAM3::MapPoint *> &vpMatches12)
{
    
    return mpSearchByBoWKernel->launch(pKF1, pKF2, vpMatches12);
}


void LoopClosingKernelController::launchWarmUp()
{
    // cudaFree(0);
    warmupKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    std::cout << "[CUDA Warm-up] Completed GPU context initialization." << std::endl;
    return;

    // std::thread([](){
    //     cudaFree(0);
    //     std::cout << "[CUDA Warm-up] GPU context initialized in background." << std::endl;
    // }).detach();
}