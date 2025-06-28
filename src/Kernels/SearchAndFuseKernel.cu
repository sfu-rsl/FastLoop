#include <iostream>
#include "Kernels/SearchAndFuseKernel.h"
#include "Kernels/MappingKernelController.h"
#include "Kernels/LoopClosingKernelController.h"

void SearchAndFuseKernel::initialize() {
    if (memory_is_initialized)
        return;

    int maxFeatures = CudaUtils::nFeatures_with_th;
    size_t mapPointVecSize, connectedKFCount;

    if (CudaUtils::cameraIsFisheye){
        mapPointVecSize = maxFeatures*2;
        connectedKFCount = MAX_CONNECTED_KF_COUNT*2;
    }
    else {
        mapPointVecSize = maxFeatures;
        connectedKFCount = MAX_CONNECTED_KF_COUNT;
    }

    checkCudaError(cudaMalloc((void**)&d_mapPoints, mapPointVecSize * sizeof(LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint)), "Failed to allocate memory for d_mapPoints");
    checkCudaError(cudaMalloc((void**)&d_connectedKFs, connectedKFCount * sizeof(CudaKeyFrame*)), "Failed to allocate memory for d_connectedKFs");
    checkCudaError(cudaMalloc((void**)&d_Tcw, connectedKFCount * sizeof(Sophus::SE3f)), "Failed to allocate memory for d_Tcw");
    checkCudaError(cudaMalloc((void**)&d_Ow, connectedKFCount * sizeof(Eigen::Vector3f)), "Failed to allocate memory for d_Ow");
    checkCudaError(cudaMalloc((void**)&d_bestDists, connectedKFCount * mapPointVecSize * sizeof(int)), "Failed to allocate memory for d_bestDists");
    checkCudaError(cudaMalloc((void**)&d_bestIdxs, connectedKFCount * mapPointVecSize * sizeof(int)), "Failed to allocate memory for d_bestIdxs");

    memory_is_initialized = true;
}

void SearchAndFuseKernel::shutdown() {
    if (!memory_is_initialized) 
        return;

    checkCudaError(cudaFree(d_mapPoints),"Failed to free search and fuse kernel memory: d_mapPoints");
    checkCudaError(cudaFree(d_connectedKFs),"Failed to free search and fuse kernel memory: d_connectedKFs");
    checkCudaError(cudaFree(d_Tcw),"Failed to free search and fuse kernel memory: d_Tcw");
    checkCudaError(cudaFree(d_Ow),"Failed to free search and fuse kernel memory: d_Ow");
    checkCudaError(cudaFree(d_bestDists),"Failed to free search and fuse kernel memory: d_bestDists");
    checkCudaError(cudaFree(d_bestIdxs),"Failed to free search and fuse kernel memory: d_bestIdxs");
}

__device__ Eigen::Vector2f pinholeProject1(const Eigen::Vector3f &v3D, float* mvParameters) {
    Eigen::Vector2f res;
    res[0] = mvParameters[0] * v3D[0] / v3D[2] + mvParameters[2];
    res[1] = mvParameters[1] * v3D[1] / v3D[2] + mvParameters[3];

    return res;
}

__device__ inline bool isInImage1(CudaKeyFrame* keyframe, const float &x, const float &y) {
    return (x>=keyframe->mnMinX && x<keyframe->mnMaxX && y>=keyframe->mnMinY && y<keyframe->mnMaxY);
}

__device__ int predictScale1(float currentDist, float maxDistance, CudaKeyFrame* pKF) {
    float ratio = maxDistance/currentDist;
    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels-1;

    return nScale;
}

__global__ void searchAndFuseKernel(
    LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint* mapPoints, CudaKeyFrame** connectedKFs, int numPoints, int numKFs, 
    Eigen::Vector3f *Ow, Sophus::SE3f *Tcw, float th,
    int* bestDists, int* bestIdxs
) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int maxIdx, connectedKFIdx, mapPointIdx;
    maxIdx = numPoints * numKFs;
    connectedKFIdx = idx / numPoints;
    mapPointIdx = idx % numPoints;

    if (idx >= maxIdx || connectedKFIdx >= numKFs || mapPointIdx >= numPoints)
        return;

    bestDists[idx] = 256;
    bestIdxs[idx] = -1;

    LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint pMP = mapPoints[mapPointIdx];
    CudaKeyFrame *connectedKF = connectedKFs[connectedKFIdx];

    Sophus::SE3f currTcw = Tcw[connectedKFIdx];
    Eigen::Vector3f currOw = Ow[connectedKFIdx];

    Eigen::Vector3f p3Dw = pMP.mWorldPos;
    Eigen::Vector3f p3Dc = currTcw * p3Dw;

    Eigen::Vector2f uv;
    uv = pinholeProject1(p3Dc, connectedKF->camera1.mvParameters);

    if ((p3Dc(2) < 0.0f) || (!isInImage1(connectedKF, uv(0), uv(1))))
        return;
    

    const float maxDistance = 1.2 * pMP.mfMaxDistance;
    const float minDistance = 0.8 * pMP.mfMinDistance;
    Eigen::Vector3f PO = p3Dw - currOw;
    const float dist3D = PO.norm();

    if (dist3D < minDistance || dist3D > maxDistance)
        return;
    
    Eigen::Vector3f Pn = pMP.mNormalVector;
    if (PO.dot(Pn) < 0.5*dist3D)
        return;

    int nPredictedLevel = predictScale1(dist3D, pMP.mfMaxDistance, connectedKF);
    const float radius = th * connectedKF->mvScaleFactors[nPredictedLevel];

    const uint8_t* MPdescriptor = &pMP.mDescriptor[0];
    int bestDist = 256;
    int bestIdx = -1;

    float factorX = radius;
    float factorY = radius;
    float x = uv.x();
    float y = uv.y();

    const int nMinCellX = max(0,(int)floor((x - connectedKF->mnMinX - factorX) * connectedKF->mfGridElementWidthInv));
    if (nMinCellX >= connectedKF->mnGridCols) 
        return;
    
    const int nMaxCellX = min((int)connectedKF->mnGridCols-1,(int)ceil((x - connectedKF->mnMinX + factorX) * connectedKF->mfGridElementWidthInv));
    if (nMaxCellX < 0)
        return;
    
    const int nMinCellY = max(0,(int)floor((y - connectedKF->mnMinY - factorY) * connectedKF->mfGridElementHeightInv));
    if (nMinCellY >= connectedKF->mnGridRows)
        return;
    
    const int nMaxCellY = min((int)connectedKF->mnGridRows-1,(int)ceil((y - connectedKF->mnMinY + factorY) * connectedKF->mfGridElementHeightInv));
    if (nMaxCellY < 0)
        return;

    for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
        for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {   
            std::size_t* vCell;
            int vCell_size;
            
            vCell = &connectedKF->flatMGrid[ix * connectedKF->mnGridRows * KEYPOINTS_PER_CELL + iy * KEYPOINTS_PER_CELL];
            vCell_size = connectedKF->flatMGrid_size[ix * connectedKF->mnGridRows + iy];
            
            for (size_t j=0, jend=vCell_size; j<jend; j++) {
                size_t temp_idx = vCell[j];

                const CudaKeyPoint &kpUn = (connectedKF->Nleft == -1) ? connectedKF->mvKeysUn[temp_idx]
                                                                                        : (!false) ? connectedKF->mvKeys[temp_idx]
                                                                                                    : connectedKF->mvKeysRight[temp_idx];
                
                const float distx = kpUn.ptx-x;
                const float disty = kpUn.pty-y;

                if (fabs(distx) < radius && fabs(disty) < radius) {
                    const int &kpLevel= connectedKF->mvKeysUn[temp_idx].octave;

                    if (kpLevel < nPredictedLevel-1 || kpLevel > nPredictedLevel)
                        continue;
                    
                    const uint8_t* dKF = &connectedKF->mDescriptors[temp_idx * DESCRIPTOR_SIZE];

                    int dist = DescriptorDistance(MPdescriptor,dKF);

                    if (dist<bestDist) {
                        bestDist = dist;
                        bestIdx = temp_idx;
                    }
                } 
            }
        }
    }

    bestDists[idx] = bestDist;
    bestIdxs[idx] = bestIdx;
}


void SearchAndFuseKernel::launch(std::vector<ORB_SLAM3::KeyFrame*> connectedKFs, vector<Sophus::Sim3f> connectedScws, float th,
                        std::vector<ORB_SLAM3::MapPoint*> &vpMapPoints,
                        vector<ORB_SLAM3::MapPoint*> &validMapPoints, int* bestDists, int* bestIdxs) {
    
    if (!memory_is_initialized)
        initialize();

    int connectedKFSize = connectedKFs.size();
    if (connectedKFSize == 0 || vpMapPoints.size() == 0)
        return;

    CudaKeyFrame* connectedKFsGPUAddress[connectedKFSize];

    for (int i = 0; i < connectedKFSize; i++) {
        connectedKFsGPUAddress[i] = CudaKeyFrameStorage::getCudaKeyFrame(connectedKFs[i]->mnId);
        if (connectedKFsGPUAddress[i] == nullptr) {
            cerr << "[ERROR] SearchAndFuseKernel::launch: ] CudaKeyFrameStorage doesn't have the keyframe: " << connectedKFs[i]->mnId << "\n";
            LoopClosingKernelController::shutdownKernels();
            exit(EXIT_FAILURE);
        }
    }

    set<ORB_SLAM3::MapPoint*> spAlreadyFound;
    for (int i = 0; i < connectedKFSize; i++) {
        const std::set<ORB_SLAM3::MapPoint*>& mps = connectedKFs[i]->GetMapPoints();
        spAlreadyFound.insert(mps.begin(), mps.end());
    }

    int numValidPoints = 0;
    LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint wrappedMapPoints[vpMapPoints.size()];
    for (int i = 0; i < vpMapPoints.size(); i++) {
        ORB_SLAM3::MapPoint* pMP = vpMapPoints[i];
        if (!pMP || pMP->isBad() || spAlreadyFound.count(pMP))
            continue;
        else {
            wrappedMapPoints[numValidPoints] = LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint(pMP);
            validMapPoints.push_back(pMP);
            numValidPoints++;
        }
    }

    if (numValidPoints == 0)
        return;

    Sophus::SE3f Tcw[connectedKFSize];
    Eigen::Vector3f Ow[connectedKFSize];
    for (int i = 0; i < connectedKFSize; i++) {
        Tcw[i] = Sophus::SE3f(connectedScws[i].rotationMatrix(),connectedScws[i].translation()/connectedScws[i].scale());
        Ow[i] = Tcw[i].inverse().translation();
    }

    checkCudaError(cudaMemcpy(d_connectedKFs, connectedKFsGPUAddress, connectedKFSize * sizeof(CudaKeyFrame*), cudaMemcpyHostToDevice), "Failed to copy vector connectedKFsGPUAddress from host to device");
    checkCudaError(cudaMemcpy(d_mapPoints, wrappedMapPoints, numValidPoints * sizeof(LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint), cudaMemcpyHostToDevice), "Failed to copy vector wrappedMapPoints from host to device");
    checkCudaError(cudaMemcpy(d_Tcw, Tcw, connectedKFSize * sizeof(Sophus::SE3f), cudaMemcpyHostToDevice), "Failed to copy vector Tcw from host to device");
    checkCudaError(cudaMemcpy(d_Ow, Ow, connectedKFSize * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice), "Failed to copy vector Ow from host to device");

    int keyFramesToProcessCount = connectedKFSize;
    int blockSize = 256;
    int numBlocks = (numValidPoints*keyFramesToProcessCount + blockSize - 1) / blockSize;

    searchAndFuseKernel<<<numBlocks, blockSize>>>(
        d_mapPoints, d_connectedKFs, numValidPoints, connectedKFSize, d_Ow, d_Tcw,
        th, d_bestDists, d_bestIdxs
    );

    return;
}


void SearchAndFuseKernel::saveStats(const std::string &file_path) {
    std::string data_path = file_path + "/SearchAndFuseKernel/";
    std::cout << "[SearchAndFuseKernel:] writing stats data into file: " << data_path << '\n';
    if (mkdir(data_path.c_str(), 0755) == -1) {
        std::cerr << "[SearchAndFuseKernel:] Error creating directory: " << strerror(errno) << std::endl;
    }
    std::ofstream myfile;
    
    myfile.open(data_path + "/kernel_exec_time.txt");
    for (const auto& p : kernel_exec_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/input_data_wrap_time.txt");
    for (const auto& p : input_data_wrap_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/input_data_transfer_time.txt");
    for (const auto& p : input_data_transfer_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();
    
    myfile.open(data_path + "/output_data_transfer_time.txt");
    for (const auto& p : output_data_transfer_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();

    myfile.open(data_path + "/total_exec_time.txt");
    for (const auto& p : total_exec_time) {
        myfile << p.first << ": " << p.second << std::endl;
    }
    myfile.close();
}