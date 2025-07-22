#ifndef LOOP_CLOSING_KERNEL_CONTROLLER_H
#define LOOP_CLOSING_KERNEL_CONTROLLER_H

#include "CudaWrappers/CudaKeyFrame.h"
// #include "CudaKeyFrameStorage.h"
#include "CudaUtils.h"
#include "SearchAndFuseKernel.h"
#include "SearchForTriangulationKernel.h"
#include <memory> 

using namespace std;

class LoopClosingKernelController{
public:
    
    static bool fuseOnGPU;

    static void initializeKernels();

    static void shutdownKernels();

    static void saveKernelsStats(const std::string &file_path);

    static void launchFuseKernel(
        std::vector<ORB_SLAM3::KeyFrame*> connectedKFs, vector<Sophus::Sim3f> connectedScws, const float th,
        std::vector<ORB_SLAM3::MapPoint*> &vpMapPoints,  
        std::vector<ORB_SLAM3::MapPoint*> &validMapPoints, int* bestDists, int* bestIdxs
    );
    

private:
    static bool memory_is_initialized;
    static std::unique_ptr<SearchAndFuseKernel> mpSearchAndFuseKernel;

};

#endif
