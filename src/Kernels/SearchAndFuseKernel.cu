#include <iostream>
#include "Kernels/SearchAndFuseKernel.h"
#include "Kernels/MappingKernelController.h"
#include "Kernels/LoopClosingKernelController.h"

void SearchAndFuseKernel::initialize() {
    std::cout << "I am in search and fuse initialize\n";
    if (memory_is_initialized)
        return;

    int maxFeatures = CudaUtils::nFeatures_with_th;
    size_t mapPointVecSize, connectedKFCount;

    
    // mapPointVecSize = maxFeatures;
    mapPointVecSize = 1600;
    connectedKFCount = MAX_CONNECTED_KF_COUNT;

    checkCudaError(cudaMalloc((void**)&d_mapPoints, mapPointVecSize * sizeof(LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint)), "Failed to allocate memory for d_mapPoints");
    checkCudaError(cudaMalloc((void**)&d_connectedKFs, connectedKFCount * sizeof(CudaKeyFrame)), "Failed to allocate memory for d_connectedKFs");
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
    LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint* mapPoints, CudaKeyFrame* connectedKFs, int numPoints, int numKFs, 
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
    CudaKeyFrame connectedKF = connectedKFs[connectedKFIdx];

    for (int i = 0; i < connectedKF.mapPointsId_size; ++i) {
        if (connectedKF.mapPointsId[i] == pMP.mnId) {
            return;
        }
    }

    Sophus::SE3f currTcw = Tcw[connectedKFIdx];
    Eigen::Vector3f currOw = Ow[connectedKFIdx];

    Eigen::Vector3f p3Dw = pMP.mWorldPos;
    Eigen::Vector3f p3Dc = currTcw * p3Dw;

    Eigen::Vector2f uv;
    uv = pinholeProject1(p3Dc, connectedKF.camera1.mvParameters);
    // if ((p3Dc(2) < 0.0f) || (!isInImage1(&connectedKF, uv(0), uv(1))))
    //     return;

    const float maxDistance = 1.2 * pMP.mfMaxDistance;
    const float minDistance = 0.8 * pMP.mfMinDistance;
    Eigen::Vector3f PO = p3Dw - currOw;
    const float dist3D = PO.norm();

    if (dist3D < minDistance || dist3D > maxDistance)
        return;
    
    Eigen::Vector3f Pn = pMP.mNormalVector;
    if (PO.dot(Pn) < 0.5*dist3D)
        return;

    int nPredictedLevel = predictScale1(dist3D, pMP.mfMaxDistance, &connectedKF);
    const float radius = th * connectedKF.mvScaleFactors[nPredictedLevel];

    const uint8_t* MPdescriptor = &pMP.mDescriptor[0];
    int bestDist = 256;
    int bestIdx = -1;

    float factorX = radius;
    float factorY = radius;
    float x = uv.x();
    float y = uv.y();

    const int nMinCellX = max(0,(int)floor((x - connectedKF.mnMinX - factorX) * connectedKF.mfGridElementWidthInv));
    if (nMinCellX >= connectedKF.mnGridCols) 
        return;
    
    const int nMaxCellX = min((int)connectedKF.mnGridCols-1,(int)ceil((x - connectedKF.mnMinX + factorX) * connectedKF.mfGridElementWidthInv));
    if (nMaxCellX < 0)
        return;
    
    const int nMinCellY = max(0,(int)floor((y - connectedKF.mnMinY - factorY) * connectedKF.mfGridElementHeightInv));
    if (nMinCellY >= connectedKF.mnGridRows)
        return;
    
    const int nMaxCellY = min((int)connectedKF.mnGridRows-1,(int)ceil((y - connectedKF.mnMinY + factorY) * connectedKF.mfGridElementHeightInv));
    if (nMaxCellY < 0)
        return;

    for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
        for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {   
            std::size_t* vCell;
            int vCell_size;
            
            vCell = &connectedKF.flatMGrid[ix * connectedKF.mnGridRows * KEYPOINTS_PER_CELL + iy * KEYPOINTS_PER_CELL];
            vCell_size = connectedKF.flatMGrid_size[ix * connectedKF.mnGridRows + iy];
            
            for (size_t j=0, jend=vCell_size; j<jend; j++) {
                size_t temp_idx = vCell[j];

                const CudaKeyPoint &kpUn = (connectedKF.Nleft == -1) ? connectedKF.mvKeysUn[temp_idx]
                                                                                        : (!false) ? connectedKF.mvKeys[temp_idx]
                                                                                                    : connectedKF.mvKeysRight[temp_idx];
                
                const float distx = kpUn.ptx-x;
                const float disty = kpUn.pty-y;

                if (fabs(distx) < radius && fabs(disty) < radius) {
                    const int &kpLevel= connectedKF.mvKeysUn[temp_idx].octave;

                    if (kpLevel < nPredictedLevel-1 || kpLevel > nPredictedLevel)
                        continue;
                    
                    const uint8_t* dKF = &connectedKF.mDescriptors[temp_idx * DESCRIPTOR_SIZE];
                    
                    // int dist = DescriptorDistance(MPdescriptor,dKF);

                    // if (dist<bestDist) {
                    //     bestDist = dist;
                    //     bestIdx = temp_idx;
                    // }
                } 
            }
        }
    }

    // bestDists[idx] = bestDist;
    // bestIdxs[idx] = bestIdx;
}


void SearchAndFuseKernel::launch(std::vector<ORB_SLAM3::KeyFrame*> connectedKFs, vector<Sophus::Sim3f> connectedScws, float th,
                        std::vector<ORB_SLAM3::MapPoint*> &vpMapPoints,
                        vector<ORB_SLAM3::MapPoint*> &validMapPoints, int* bestDists, int* bestIdxs) {
    
    if (!memory_is_initialized)
        initialize();

    int connectedKFSize = connectedKFs.size();
    if (connectedKFSize == 0 || vpMapPoints.size() == 0)
        return;

    std::ofstream timing("./test/timing.txt", std::ios::app);

    auto start1 = std::chrono::high_resolution_clock::now();
    CudaKeyFrame* wrappedKeyFrames = new CudaKeyFrame[connectedKFSize];
    for (int i=0; i < connectedKFSize; i++){
        ORB_SLAM3::KeyFrame* pKF = connectedKFs[i];
        wrappedKeyFrames[i] = CudaKeyFrame();
        wrappedKeyFrames[i].setMemory(pKF);
        printf("------------- In CPU mnMinX: %f \n", pKF->mnMinX);
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed1 = end1 - start1;
    timing << "3 Prepare Wrapped CudaKeyFrames: " << elapsed1.count() << " ms" << std::endl;


    auto start2 = std::chrono::high_resolution_clock::now();
    int numValidPoints = 0;
    LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint wrappedMapPoints[vpMapPoints.size()];    
    for (int i = 0; i < vpMapPoints.size(); i++) {
        ORB_SLAM3::MapPoint* pMP = vpMapPoints[i];
        if (!pMP || pMP->isBad())
            continue;
        else {
            wrappedMapPoints[numValidPoints] = LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint(pMP);
            validMapPoints.push_back(pMP);
            numValidPoints++;
        }
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed2 = end2 - start2;
    timing << "3 Prepare Wrapped CudaMapPoints: " << elapsed2.count() << " ms" << std::endl;


    if (numValidPoints == 0)
        return;


    auto start3 = std::chrono::high_resolution_clock::now();
    Sophus::SE3f Tcw[connectedKFSize];
    Eigen::Vector3f Ow[connectedKFSize];
    for (int i = 0; i < connectedKFSize; i++) {
        Tcw[i] = Sophus::SE3f(connectedScws[i].rotationMatrix(),connectedScws[i].translation()/connectedScws[i].scale());
        Ow[i] = Tcw[i].inverse().translation();
    }
    auto end3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed3 = end3 - start3;
    timing << "3 Prepare Wrapped Tcw and Ow: " << elapsed3.count() << " ms" << std::endl;


    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Something happened CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    auto start4 = std::chrono::high_resolution_clock::now();
    checkCudaError(cudaMemcpy(d_connectedKFs, wrappedKeyFrames, connectedKFSize * sizeof(CudaKeyFrame), cudaMemcpyHostToDevice), "Failed to copy vector wrappedKeyFrames from host to device");
    checkCudaError(cudaMemcpy(d_mapPoints, wrappedMapPoints, numValidPoints * sizeof(LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint), cudaMemcpyHostToDevice), "[ERROR] SearchAndFuseKernel::launch: ] Failed to copy vector wrappedMapPoints from host to device");
    checkCudaError(cudaMemcpy(d_Tcw, Tcw, connectedKFSize * sizeof(Sophus::SE3f), cudaMemcpyHostToDevice), "Failed to copy vector Tcw from host to device");
    checkCudaError(cudaMemcpy(d_Ow, Ow, connectedKFSize * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice), "Failed to copy vector Ow from host to device");
    auto end4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed4 = end4 - start4;
    timing << "3 Memcpy: " << elapsed4.count() << " ms" << std::endl;


    auto start5 = std::chrono::high_resolution_clock::now();

    int keyFramesToProcessCount = connectedKFSize;
    int blockSize = 256;
    int numBlocks = (numValidPoints*keyFramesToProcessCount + blockSize - 1) / blockSize;

    searchAndFuseKernel<<<1, 1>>>(
        d_mapPoints, d_connectedKFs, numValidPoints, connectedKFSize, d_Ow, d_Tcw,
        th, d_bestDists, d_bestIdxs
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    auto end5 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed5 = end5 - start5;
    timing << "3 Main Kernel: " << elapsed5.count() << " ms" << std::endl;


    auto start6 = std::chrono::high_resolution_clock::now();

    checkCudaError(cudaGetLastError(), "[SearchAndFuseKernel:] Failed to launch kernel");
    checkCudaError(cudaDeviceSynchronize(), "[SearchAndFuseKernel:] cudaDeviceSynchronize failed after kernel launch");

    checkCudaError(cudaMemcpy(bestDists, d_bestDists, numValidPoints * keyFramesToProcessCount * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestDists back to host");
    checkCudaError(cudaMemcpy(bestIdxs, d_bestIdxs, numValidPoints * keyFramesToProcessCount * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestIdxs back to host");

    auto end6 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed6 = end6 - start6;
    timing << "3 After main kernel: " << elapsed6.count() << " ms" << std::endl;

    // std::ofstream gpuOutFile("./test/GPU-Side.txt", std::ios::app);
    // gpuOutFile << "\n\n////////////////////////////////////////// Current KF: " << connectedKFs[0]->mnId << " //////////////////////////////////////////\n";
    
    // std::ofstream cpuOutFile("./test/CPU-Side.txt", std::ios::app);
    // cpuOutFile << "\n\n////////////////////////////////////////// Current KF: " << connectedKFs[0]->mnId << " //////////////////////////////////////////\n";
        
    // for (int iKF = 0; iKF < connectedKFSize; iKF++) {
    //     gpuOutFile << "================================= Connected KF: " << connectedKFs[iKF]->mnId << " =================================\n";
    //     std::cout << "numValidPoints: " << numValidPoints << std::endl;
    //     cpuOutFile << "================================= Connected KF: " << connectedKFs[iKF]->mnId << " =================================\n";
    //     cpuOutFile.flush();
    //     for (int iMP = 0; iMP < numValidPoints; iMP++) {
    //         int idx = iKF*numValidPoints + iMP;
    //         if (bestDists[idx] != 256 && !validMapPoints[iMP]->IsInKeyFrame(connectedKFs[iKF]))
    //             gpuOutFile << "(i: " << iMP << ", bestDist: " << bestDists[idx] << ", bestIdx: " << bestIdxs[idx] << ")\n";
    //             // printf("(i: %d, bestDist: %d, bestIdx: %d), ", iMP, bestDists[idx], bestIdxs[idx]);
    //     }
    //     gpuOutFile << "\n\n";
    //     // printf("\n");
        
    //     // cout << "************ CPU Side ************\n";
    //     origFuse(connectedKFs[iKF], connectedScws[iKF], vpMapPoints, th);
    // }

    // gpuOutFile << "**********************************************************\n";
    // cpuOutFile << "**********************************************************\n";
    // count << "**********************************************************\n";

    return;
}


void SearchAndFuseKernel::origFuse(ORB_SLAM3::KeyFrame *pKF, Sophus::Sim3f &Scw, const vector<ORB_SLAM3::MapPoint*> &vpPoints, const float th) {
    
    std::ofstream count("./test/count.txt", std::ios::app);
    count << "Fuse: pKF-ID = " << pKF->mnId << "\n";

    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const float &bf = pKF->mbf;

    Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(),Scw.translation()/Scw.scale());
    Eigen::Vector3f Ow = Tcw.inverse().translation();

    const set<ORB_SLAM3::MapPoint*> spAlreadyFound = pKF->GetMapPoints();

    int nFused=0;

    const int nPoints = vpPoints.size();

    int validMapPointCounter = -1;

    std::ofstream cpuOutFile("./test/CPU-Side.txt", std::ios::app);
    
    for(int iMP=0; iMP<nPoints; iMP++) {
        ORB_SLAM3::MapPoint* pMP = vpPoints[iMP];

        if(pMP->isBad()|| spAlreadyFound.count(pMP))
            continue;
        
        validMapPointCounter++;

        // Get 3D Coords.
        Eigen::Vector3f p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        Eigen::Vector3f p3Dc = Tcw * p3Dw;

        // Depth must be positive
        if(p3Dc(2)<0.0f)
            continue;

        // Project into Image
        const Eigen::Vector2f uv = pKF->mpCamera->project(p3Dc);

        // Point must be inside the image
        if(!pKF->IsInImage(uv(0),uv(1)))
            continue;

        // Depth must be inside the scale pyramid of the image
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        Eigen::Vector3f PO = p3Dw-Ow;
        const float dist3D = PO.norm();

        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        Eigen::Vector3f Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        // Compute predicted scale level
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0),uv(1),radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;

        for(vector<size_t>::const_iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++){
            const size_t idx = *vit;

            const int &kpLevel = pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel) {
                continue;
            }

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = origDescriptorDistance(dMP,dKF);

            if(dist<bestDist) {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if (bestDist != INT_MAX)
            cpuOutFile << "(i: " << validMapPointCounter << ", bestDist: " << bestDist << ", bestIdx: " << bestIdx << ")\n";
    }
    std::cout << "validMapPointCounter: " << validMapPointCounter << std::endl;
    cpuOutFile << "\n\n";
}


int SearchAndFuseKernel::origDescriptorDistance(const cv::Mat &a, const cv::Mat &b) {
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
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