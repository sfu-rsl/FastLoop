#include "Kernels/LoopClosingKernelController.h"

// #define DEBUG

#ifdef DEBUG
#define DEBUG_PRINT(msg) std::cout << "Debug [LoopClosing KernelController::]  " << msg << std::endl
#else
#define DEBUG_PRINT(msg) do {} while (0)
#endif


bool LoopClosingKernelController::fuseOnGPU;
bool LoopClosingKernelController::memory_is_initialized = false;

std::unique_ptr<FuseKernel> LoopClosingKernelController::mpFuseKernel = std::make_unique<FuseKernel>();


void LoopClosingKernelController::initializeKernels(){

    DEBUG_PRINT("Initializing Kernels");
    
    // CudaKeyFrameStorage::initializeMemory();
    
    if (fuseOnGPU)
        mpFuseKernel->initialize();

    checkCudaError(cudaDeviceSynchronize(), "[Loop Closing Kernel Controller:] Failed to initialize kernels.");
    memory_is_initialized = true;
}


void LoopClosingKernelController::shutdownKernels() {
    // unique_lock<mutex> lock(shutDownMutex);

    // localMappingFinished = _localMappingFinished ? true : localMappingFinished;
    // loopClosingFinished = _localMappingFinished ? true : loopClosingFinished;
    
    // if (!localMappingFinished || !loopClosingFinished || isShuttingDown)
    //     return;

    // isShuttingDown = true;

    cout << "Shutting kernels down...\n";

    if (memory_is_initialized) {
        // CudaKeyFrameStorage::shutdown();

        if (fuseOnGPU == 1)
            mpFuseKernel->shutdown();
    }

    CudaUtils::shutdown();
    cudaDeviceSynchronize();
}

void LoopClosingKernelController::saveKernelsStats(const std::string &file_path){

    DEBUG_PRINT("Saving Kernels Stats");
    
    mpFuseKernel->saveStats(file_path);
}

void LoopClosingKernelController::launchFuseKernel(
    std::vector<ORB_SLAM3::KeyFrame*> connectedKFs, vector<Sophus::Sim3f> connectedScws, const float th,
    std::vector<ORB_SLAM3::MapPoint*> &vpMapPoints,  
    std::vector<ORB_SLAM3::MapPoint*> &validMapPoints, int* bestDists, int* bestIdxs
) {

    DEBUG_PRINT("Launching Fuse Kernel");

    mpFuseKernel->launch(connectedKFs, connectedScws, th, vpMapPoints, validMapPoints, bestDists, bestIdxs);
}