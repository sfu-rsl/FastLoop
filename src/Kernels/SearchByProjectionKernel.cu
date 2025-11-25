#include <iostream>
#include "Kernels/SearchByProjectionKernel.h"


void SearchByProjectionKernel::initialize(){
    if (memory_is_initialized)
        return;
    
    size_t mapPointVecSize = 1500;
    cudaMallocHost((void**)&h_MapPoints, mapPointVecSize * sizeof(LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint));
    // cudaMallocHost((void**)&h_KeyFrame, sizeof(CudaKeyFrame));
    cudaMallocHost((void**)&bestDists1, mapPointVecSize * sizeof(int));
    cudaMallocHost((void**)&bestIdxs1, mapPointVecSize * sizeof(int));
    cudaMallocHost((void**)&bestDists, mapPointVecSize * sizeof(int));
    cudaMallocHost((void**)&bestIdxs, mapPointVecSize * sizeof(int));

    cudaMalloc(&d_MapPoints, mapPointVecSize * sizeof(LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint));
    cudaMalloc(&d_KeyFrame, sizeof(CudaKeyFrame));
    cudaMalloc(&d_bestDists1, mapPointVecSize * sizeof(int));
    cudaMalloc(&d_bestIdxs1, mapPointVecSize * sizeof(int));
    cudaMalloc(&d_bestDists, mapPointVecSize * sizeof(int));
    cudaMalloc(&d_bestIdxs, mapPointVecSize * sizeof(int));

    // *h_KeyFrame = CudaKeyFrame();

    memory_is_initialized = true;
}


void SearchByProjectionKernel::shutdown() {
    if (!memory_is_initialized) 
        return;

    // h_KeyFrame->freeMemory();

    cudaFreeHost(h_MapPoints);
    // cudaFreeHost(h_KeyFrame);
    cudaFreeHost(bestDists1);
    cudaFreeHost(bestIdxs1);
    cudaFreeHost(bestDists);
    cudaFreeHost(bestIdxs);
    cudaFree(d_MapPoints);
    cudaFree(d_KeyFrame);
    cudaFree(d_bestDists1);
    cudaFree(d_bestIdxs1);
    cudaFree(d_bestDists);
    cudaFree(d_bestIdxs);
}

__device__ inline bool isInImage(CudaKeyFrame* keyframe, const float &x, const float &y) {
    return (x>=keyframe->mnMinX && x<keyframe->mnMaxX && y>=keyframe->mnMinY && y<keyframe->mnMaxY);
}


__device__ inline int predictScale(float currentDist, float maxDistance, CudaKeyFrame* pKF) {
    float ratio = maxDistance/currentDist;
    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels-1;

    return nScale;
}


__device__ inline Eigen::Vector2f KannalaBrandt8Project(const Eigen::Vector3f &v3D, float* mvParameters) {
    const float x2_plus_y2 = v3D[0] * v3D[0] + v3D[1] * v3D[1];
    const float theta = atan2f(sqrtf(x2_plus_y2), v3D[2]);
    const float psi = atan2f(v3D[1], v3D[0]);
    
    const float theta2 = theta * theta;
    const float theta3 = theta * theta2;
    const float theta5 = theta3 * theta2;
    const float theta7 = theta5 * theta2;
    const float theta9 = theta7 * theta2;
    const float r = theta + mvParameters[4] * theta3 + mvParameters[5] * theta5
                         + mvParameters[6] * theta7 + mvParameters[7] * theta9;

    Eigen::Vector2f res;
    res[0] = mvParameters[0] * r * cos(psi) + mvParameters[2];
    res[1] = mvParameters[1] * r * sin(psi) + mvParameters[3];
    return res;
}


__global__ void searchByProjectionKernel(Eigen::Vector3f Ow, Sophus::SE3f Tcw,
                            CudaKeyFrame *connectedKF, LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint* mapPoints,
                            int numPoints, float th,
                            int* bestDists, int* bestIdxs) 
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numPoints)
        return;
    
    bestDists[idx] = 256;
    bestIdxs[idx] = -1;

    LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint pMP = mapPoints[idx];
    
    // printf("Device:\n");
    // Eigen::Matrix3f Rcw = Tcw.rotationMatrix();
    // Eigen::Vector3f tcw = Tcw.translation();
    // printf("Tcw rotation:\n");
    // for(int i = 0; i < 3; i++) {
    //     printf("%f %f %f\n", Rcw(i,0), Rcw(i,1), Rcw(i,2));
    // }
    // printf("Tcw translation: %f %f %f\n", tcw(0), tcw(1), tcw(2));
    // printf("Ow: %f %f %f\n", Ow(0), Ow(1), Ow(2));
    // printf("Device: KeyFrame = %llu\n", connectedKF->mnId);

    const float &fx = connectedKF->fx;
    const float &fy = connectedKF->fy;
    const float &cx = connectedKF->cx;
    const float &cy = connectedKF->cy;

    Eigen::Vector3f p3Dw = pMP.mWorldPos;
    Eigen::Vector3f p3Dc = Tcw * p3Dw;

    if(p3Dc(2)<0.0)
        return;

    const float invz = 1/p3Dc(2);
    const float x1 = p3Dc(0)*invz;
    const float y1 = p3Dc(1)*invz;

    const float u = fx*x1+cx;
    const float v = fy*y1+cy;

    // Point must be inside the image
    if(!isInImage(connectedKF, u, v))
        return;

    const float maxDistance = 1.2f * pMP.mfMaxDistance; // pMP->GetMaxDistanceInvariance();
    const float minDistance = 0.8f * pMP.mfMinDistance; // pMP->GetMinDistanceInvariance();
    Eigen::Vector3f PO = p3Dw-Ow;
    const float dist = PO.norm();

    // printf("Device: idx = %llu, id = %llu, dist = %f, minDistance = %f, maxDistance = %f\n", idx, pMP.mnId, dist, minDistance, maxDistance);

    if(dist<minDistance || dist>maxDistance)
        return;

    Eigen::Vector3f Pn = pMP.mNormalVector;

    if(PO.dot(Pn)<0.5*dist)
        return;

    int nPredictedLevel = predictScale(dist, pMP.mfMaxDistance, connectedKF); // pMP->PredictScale(dist,pKF);
    const float radius = th*connectedKF->mvScaleFactors[nPredictedLevel];

    const uint8_t* MPdescriptor = &pMP.mDescriptor[0];
    int bestDist = 256;
    int bestIdx = -1;

    float factorX = radius;
    float factorY = radius;
    float x = u;
    float y = v;

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
                    // const size_t idx = *vit; idx=temp_idx
                    // if(vpMatched[idx])
                    //     continue;

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



__global__ void searchByProjectionKernel2(Eigen::Vector3f Ow, Sophus::SE3f Tcw,
                            CudaKeyFrame *connectedKF, LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint* mapPoints,
                            int numPoints, float th,
                            int* bestDists, int* bestIdxs) 
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numPoints)
        return;
    
    bestDists[idx] = 256;
    bestIdxs[idx] = -1;
    
    LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint pMP = mapPoints[idx];
    
    // printf("Device:\n");
    // Eigen::Matrix3f Rcw = Tcw.rotationMatrix();
    // Eigen::Vector3f tcw = Tcw.translation();
    // printf("Tcw rotation:\n");
    // for(int i = 0; i < 3; i++) {
    //     printf("%f %f %f\n", Rcw(i,0), Rcw(i,1), Rcw(i,2));
    // }
    // printf("Tcw translation: %f %f %f\n", tcw(0), tcw(1), tcw(2));
    // printf("Ow: %f %f %f\n", Ow(0), Ow(1), Ow(2));
    // printf("Device: KeyFrame = %llu\n", connectedKF->mnId);

    const float &fx = connectedKF->fx;
    const float &fy = connectedKF->fy;
    const float &cx = connectedKF->cx;
    const float &cy = connectedKF->cy;

    Eigen::Vector3f p3Dw = pMP.mWorldPos;
    Eigen::Vector3f p3Dc = Tcw * p3Dw;
    

    if(p3Dc(2)<0.0){
        return;
    }

    Eigen::Vector2f uv;
    uv = KannalaBrandt8Project(p3Dc, connectedKF->camera1.mvParameters);

    if (!isInImage(connectedKF, uv(0), uv(1))){
        return;
    }

    const float maxDistance = 1.2f * pMP.mfMaxDistance; // pMP->GetMaxDistanceInvariance();
    const float minDistance = 0.8f * pMP.mfMinDistance; // pMP->GetMinDistanceInvariance();
    Eigen::Vector3f PO = p3Dw-Ow;
    const float dist = PO.norm();

    if(dist<minDistance || dist>maxDistance)
        return;
    
    Eigen::Vector3f Pn = pMP.mNormalVector;

    if(PO.dot(Pn)<0.5*dist)
        return;

    int nPredictedLevel = predictScale(dist, pMP.mfMaxDistance, connectedKF); // pMP->PredictScale(dist,pKF);
    const float radius = th*connectedKF->mvScaleFactors[nPredictedLevel];
    // printf("Device: idx = %llu, id = %llu, radius = %f\n", idx, pMP.mnId, radius);
    
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
                    // const size_t idx = *vit; idx=temp_idx
                    // if(vpMatched[idx])
                    //     continue;

                    const int &kpLevel= connectedKF->mvKeysUn[temp_idx].octave;
                    // printf("Device: idx = %llu, id = %llu, kpLevel = %d\n", idx, pMP.mnId, kpLevel);

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


int SearchByProjectionKernel::launch(ORB_SLAM3::KeyFrame* pKF, Sophus::Sim3<float> &Scw, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints, const std::vector<ORB_SLAM3::KeyFrame*> &vpPointsKFs,
                    std::vector<ORB_SLAM3::MapPoint*> &vpMatched, std::vector<ORB_SLAM3::KeyFrame*> &vpMatchedKF, int th, float ratioHamming) {
    
    // printf("Hello\n");
    int numValidPoints = 0;
    const int TH_LOW = 50;
    int nmatches=0;

    size_t mapPointVecSize = vpPoints.size();
    // cout << "mapPointVecSize: " << mapPointVecSize << endl;

    LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint *h_MapPoints, *d_MapPoints;
    CudaKeyFrame *h_KeyFrame, *d_KeyFrame;
    int *d_bestDists, *d_bestIdxs;
    int *bestDists, *bestIdxs;

    cudaMallocHost((void**)&h_MapPoints, mapPointVecSize * sizeof(LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint));
    cudaMallocHost((void**)&h_KeyFrame, sizeof(CudaKeyFrame));
    cudaMallocHost((void**)&bestDists, mapPointVecSize * sizeof(int));
    cudaMallocHost((void**)&bestIdxs, mapPointVecSize * sizeof(int));

    cudaMalloc(&d_MapPoints, mapPointVecSize * sizeof(LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint));
    cudaMalloc(&d_KeyFrame, sizeof(CudaKeyFrame));
    cudaMalloc(&d_bestDists, mapPointVecSize * sizeof(int));
    cudaMalloc(&d_bestIdxs, mapPointVecSize * sizeof(int));

    Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(),Scw.translation()/Scw.scale());
    Eigen::Vector3f Ow = Tcw.inverse().translation();

    // printf("Host:\n");
    // Eigen::Matrix3f Rcw = Tcw.rotationMatrix();
    // Eigen::Vector3f tcw = Tcw.translation();
    // printf("Tcw rotation:\n");
    // for(int i = 0; i < 3; i++) {
    //     printf("%f %f %f\n", Rcw(i,0), Rcw(i,1), Rcw(i,2));
    // }
    // printf("Tcw translation: %f %f %f\n", tcw(0), tcw(1), tcw(2));
    // printf("Ow: %f %f %f\n", Ow(0), Ow(1), Ow(2));
    
    // Set of MapPoints already found in the KeyFrame
    set<ORB_SLAM3::MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<ORB_SLAM3::MapPoint*>(NULL));
    // cout << "spAlreadyFound: " << spAlreadyFound.size() << endl;

    for (int i = 0; i < mapPointVecSize; i++) {
        ORB_SLAM3::MapPoint* pMP = vpPoints[i];
        if (!pMP || pMP->isBad() || spAlreadyFound.count(pMP))
            continue;
        else {
            // printf("Host: id = %llu\n", pMP->mnId);
            h_MapPoints[numValidPoints] = LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint(pMP);
            numValidPoints++;
        }
    }

    // cout << "numValidPoints: " << numValidPoints << endl;

    // printf("Host: KeyFrame = %llu\n", pKF->mnId);
    *h_KeyFrame = CudaKeyFrame();
    h_KeyFrame->setMemory(pKF);

    
    cudaMemcpy(d_MapPoints, h_MapPoints, mapPointVecSize * sizeof(LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_KeyFrame, h_KeyFrame, sizeof(CudaKeyFrame), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (mapPointVecSize + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    searchByProjectionKernel<<<blocks, threads>>>(Ow, Tcw,
                                        d_KeyFrame, d_MapPoints, 
                                        mapPointVecSize, th, 
                                        d_bestDists, d_bestIdxs);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_pinned = 0;
    cudaEventElapsedTime(&ms_pinned, start, stop);
    // std::cout << ms_pinned << "\n";


    checkCudaError(cudaMemcpy(bestDists, d_bestDists, mapPointVecSize * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestDists back to host");
    checkCudaError(cudaMemcpy(bestIdxs, d_bestIdxs, mapPointVecSize * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestIdxs back to host");

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize(); // ensure kernel errors propagate

    
    // printf("Buy\n");

    // std::ofstream gpuOutFile("./test/GPU-Side.txt", std::ios::app);
    // gpuOutFile << "\n\n////////////////////////////////////////// Current KF: " << pKF->mnId << " //////////////////////////////////////////" << endl;
    
    // std::ofstream cpuOutFile("./test/CPU-Side.txt", std::ios::app);
    // cpuOutFile << "\n\n////////////////////////////////////////// Current KF: " << pKF->mnId << " //////////////////////////////////////////" << endl;
        
    // for (int i = 0; i < numValidPoints; i++) {
    //     if(bestDists[i] != 256)
    //         gpuOutFile << "(i: " << i << ", bestDist: " << bestDists[i] << ", bestIdx: " << bestIdxs[i] << ")\n";
    //         // printf("(i: %d, bestDist: %d, bestIdx: %d), ", iMP, bestDists[idx], bestIdxs[idx]);
    // }
    // gpuOutFile << "\n\n";
    // // printf("\n");
    
    // // cout << "************ CPU Side ************\n";
    // // origFuse(connectedKFs[iKF], connectedScws[iKF], vpMapPoints, th);
    // origSearchByProjection(pKF, Scw, vpPoints, vpPointsKFs, vpMatched, vpMatchedKF, th, ratioHamming);

    // gpuOutFile << "**********************************************************\n";
    // cpuOutFile << "**********************************************************\n";

    
    for(size_t iMP = 0; iMP < mapPointVecSize; iMP++) {
        ORB_SLAM3::MapPoint* pMP = vpPoints[iMP];
        ORB_SLAM3::KeyFrame* pKFi = vpPointsKFs[iMP];
        int bestDist = bestDists[iMP];
        int bestIdx = bestIdxs[iMP];

        if (bestDist == 256 || bestIdx == -1)
            continue;

        if (bestDist <= TH_LOW*ratioHamming) {
            vpMatched[bestIdx] = pMP;
            vpMatchedKF[bestIdx] = pKFi;
            nmatches++;
        }
    }
    
    cudaFreeHost(h_MapPoints);
    cudaFreeHost(h_KeyFrame);
    cudaFreeHost(bestDists);
    cudaFreeHost(bestIdxs);
    cudaFree(d_MapPoints);
    cudaFree(d_KeyFrame);
    cudaFree(d_bestDists);
    cudaFree(d_bestIdxs);


    

    return nmatches;
}

__global__ void mergedSearchByProjectionKernel(Eigen::Vector3f Ow1, Sophus::SE3f Tcw1, Eigen::Vector3f Ow, Sophus::SE3f Tcw,
                            CudaKeyFrame *connectedKF, LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint* mapPoints,
                            int numPoints, float th1, float th,
                            int* bestDists1, int* bestIdxs1, int* bestDists, int* bestIdxs) 
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numPoints)
        return;
    
    bool terminated = false;
    bool terminated1 = false;

    bestDists1[idx] = 256;
    bestIdxs1[idx] = -1;
    bestDists[idx] = 256;
    bestIdxs[idx] = -1;

    LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint pMP = mapPoints[idx];

    const float &fx = connectedKF->fx;
    const float &fy = connectedKF->fy;
    const float &cx = connectedKF->cx;
    const float &cy = connectedKF->cy;

    Eigen::Vector3f p3Dw = pMP.mWorldPos;
    Eigen::Vector3f p3Dc1 = Tcw1 * p3Dw;
    Eigen::Vector3f p3Dc = Tcw * p3Dw;

    int nPredictedLevel1, nPredictedLevel;

    if(p3Dc1(2)<0.0)
        terminated1 = true;
    
    if(p3Dc(2)<0.0)
        terminated = true;;

    Eigen::Vector2f uv;
    if(!terminated1){
        uv = KannalaBrandt8Project(p3Dc1, connectedKF->camera1.mvParameters);

        if (!isInImage(connectedKF, uv(0), uv(1)))
            terminated1 = true;
        
        // printf("Device: idx = %llu, id = %llu, uv(0) = %f, uv(1) = %f\n", idx, pMP.mnId, uv(0), uv(1));
    }
    
    float u, v;
    if(!terminated){
        const float invz = 1/p3Dc(2);
        const float x1 = p3Dc(0)*invz;
        const float y1 = p3Dc(1)*invz;

        u = fx*x1+cx;
        v = fy*y1+cy;

        // Point must be inside the image
        if(!isInImage(connectedKF, u, v))
            terminated = true;
    }

    const float maxDistance = 1.2f * pMP.mfMaxDistance; // pMP->GetMaxDistanceInvariance();
    const float minDistance = 0.8f * pMP.mfMinDistance; // pMP->GetMinDistanceInvariance();
    Eigen::Vector3f Pn = pMP.mNormalVector;
    float dist1, dist;

    if(!terminated1){
        Eigen::Vector3f PO1 = p3Dw-Ow1;
        dist1 = PO1.norm();

        if(dist1<minDistance || dist1>maxDistance)
            terminated1 = true;
        
        if(PO1.dot(Pn)<0.5*dist1)
            terminated1 = true;
    }

    if(!terminated){
        Eigen::Vector3f PO = p3Dw-Ow;
        dist = PO.norm();
        
        if(dist<minDistance || dist>maxDistance)
            terminated = true;
        
        if(PO.dot(Pn)<0.5*dist)
            terminated = true;
    }
    
    float radius1, radius;

    if(!terminated1){
        nPredictedLevel1 = predictScale(dist1, pMP.mfMaxDistance, connectedKF); // pMP->PredictScale(dist,pKF);
        radius1 = th1*connectedKF->mvScaleFactors[nPredictedLevel1];
    }

    if(!terminated){
        nPredictedLevel = predictScale(dist, pMP.mfMaxDistance, connectedKF); // pMP->PredictScale(dist,pKF);
        radius = th*connectedKF->mvScaleFactors[nPredictedLevel];
    }

    const uint8_t* MPdescriptor = &pMP.mDescriptor[0];

    int bestDist1 = 256;
    int bestIdx1 = -1;
    int bestDist = 256;
    int bestIdx = -1;

    if(!terminated1){
        float factorX1 = radius1;
        float factorY1 = radius1;
        float x1 = uv.x();
        float y1 = uv.y();

        const int nMinCellX = max(0,(int)floor((x1 - connectedKF->mnMinX - factorX1) * connectedKF->mfGridElementWidthInv));
        if (nMinCellX >= connectedKF->mnGridCols) 
            return;

        const int nMaxCellX = min((int)connectedKF->mnGridCols-1,(int)ceil((x1 - connectedKF->mnMinX + factorX1) * connectedKF->mfGridElementWidthInv));
        if (nMaxCellX < 0)
            return;
        
        const int nMinCellY = max(0,(int)floor((y1 - connectedKF->mnMinY - factorY1) * connectedKF->mfGridElementHeightInv));
        if (nMinCellY >= connectedKF->mnGridRows)
            return;
        
        const int nMaxCellY = min((int)connectedKF->mnGridRows-1,(int)ceil((y1 - connectedKF->mnMinY + factorY1) * connectedKF->mfGridElementHeightInv));
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
                    const float distx = kpUn.ptx-x1;
                    const float disty = kpUn.pty-y1;

                    if (fabs(distx) < radius1 && fabs(disty) < radius1) {
                        // const size_t idx = *vit; idx=temp_idx
                        // if(vpMatched[idx])
                        //     continue;

                        const int &kpLevel= connectedKF->mvKeysUn[temp_idx].octave;
                        // printf("Device: idx = %llu, id = %llu, kpLevel = %d\n", idx, pMP.mnId, kpLevel);

                        if (kpLevel < nPredictedLevel1-1 || kpLevel > nPredictedLevel1)
                            continue;

                        const uint8_t* dKF = &connectedKF->mDescriptors[temp_idx * DESCRIPTOR_SIZE];

                        int dist1 = DescriptorDistance(MPdescriptor,dKF);

                        if (dist1<bestDist1) {
                            bestDist1 = dist1;
                            bestIdx1 = temp_idx;
                        }
                    }
                }
            }
        }
    }
    if(!terminated){
        float factorX = radius;
        float factorY = radius;
        float x = u;
        float y = v;

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
                        // const size_t idx = *vit; idx=temp_idx
                        // if(vpMatched[idx])
                        //     continue;

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
    }

    bestDists1[idx] = bestDist1;
    bestIdxs1[idx] = bestIdx1;
    bestDists[idx] = bestDist;
    bestIdxs[idx] = bestIdx; 
}



int SearchByProjectionKernel::launch2(ORB_SLAM3::KeyFrame* pKF, Sophus::Sim3<float> &Scw, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints,
                    std::vector<ORB_SLAM3::MapPoint*> &vpMatched, int th, float ratioHamming)
{
    std::ofstream timing("./test/timing.txt", std::ios::app);

    auto start1 = std::chrono::high_resolution_clock::now();
    if (!memory_is_initialized)
        initialize();
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed1 = end1 - start1;
    // timing << "? initialize: " << elapsed1.count() << " ms" << std::endl;


    int numValidPoints = 0;
    const int TH_LOW = 50;
    int nmatches=0;

    size_t mapPointVecSize = vpPoints.size();

    // LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint *h_MapPoints, *d_MapPoints;
    // CudaKeyFrame *h_KeyFrame, *d_KeyFrame;
    // int *d_bestDists, *d_bestIdxs;
    // int *bestDists, *bestIdxs;

    // cudaMallocHost((void**)&h_MapPoints, mapPointVecSize * sizeof(LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint));
    // cudaMallocHost((void**)&h_KeyFrame, sizeof(CudaKeyFrame));
    // cudaMallocHost((void**)&bestDists, mapPointVecSize * sizeof(int));
    // cudaMallocHost((void**)&bestIdxs, mapPointVecSize * sizeof(int));

    // cudaMalloc(&d_MapPoints, mapPointVecSize * sizeof(LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint));
    // cudaMalloc(&d_KeyFrame, sizeof(CudaKeyFrame));
    // cudaMalloc(&d_bestDists, mapPointVecSize * sizeof(int));
    // cudaMalloc(&d_bestIdxs, mapPointVecSize * sizeof(int));

    auto start2 = std::chrono::high_resolution_clock::now();
    Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(),Scw.translation()/Scw.scale());
    Eigen::Vector3f Ow = Tcw.inverse().translation();
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed2 = end2 - start2;
    // timing << "? Tcw: " << elapsed2.count() << " ms" << std::endl;

    // Set of MapPoints already found in the KeyFrame
    // set<ORB_SLAM3::MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    // spAlreadyFound.erase(static_cast<ORB_SLAM3::MapPoint*>(NULL));

    auto start3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < mapPointVecSize; i++) {
        ORB_SLAM3::MapPoint* pMP = vpPoints[i];
        if (!pMP || pMP->isBad() ) // || spAlreadyFound.count(pMP)) todo1
            continue;
        else {
            // printf("mnId = %llu\n", pMP->mnId);
            // printf("Before Host: mfMinDistance = %f, mfMaxDistance = %f\n", pMP->mfMinDistance, pMP->mfMaxDistance);
            h_MapPoints[numValidPoints] = LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint(pMP);
            // printf("After Host: mfMinDistance = %f, mfMaxDistance = %f\n", h_MapPoints[numValidPoints].mfMinDistance, h_MapPoints[numValidPoints].mfMaxDistance);
            numValidPoints++;
        }
    }
    auto end3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed3 = end3 - start3;
    // timing << "? h_MapPoints: " << elapsed3.count() << " ms" << std::endl;


    auto start4 = std::chrono::high_resolution_clock::now();
    // *h_KeyFrame = CudaKeyFrame();
    // h_KeyFrame->setMemory(pKF);
    d_KeyFrame = LoopClosingCudaKeyFrameStorage::getCudaKeyFrame(pKF->mnId);
    if (d_KeyFrame == nullptr){
        // timing << "No " << pKF->mnId << std::endl;
        d_KeyFrame = LoopClosingCudaKeyFrameStorage::addCudaKeyFrame(pKF);
    }
    // else{
    //     timing << "Yes " << pKF->mnId << std::endl;
    // }
    auto end4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed4 = end4 - start4;
    // timing << "? h_KeyFrames Single: " << elapsed4.count() << " ms" << std::endl;

    auto start5 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_MapPoints, h_MapPoints, mapPointVecSize * sizeof(LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_KeyFrame, h_KeyFrame, sizeof(CudaKeyFrame), cudaMemcpyHostToDevice);
    auto end5 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed5 = end5 - start5;
    // timing << "? cudaMemcpy: " << elapsed5.count() << " ms" << std::endl;

    auto start6 = std::chrono::high_resolution_clock::now();
    int threads = 256;
    int blocks = (mapPointVecSize + threads - 1) / threads;
    searchByProjectionKernel2<<<blocks, threads>>>(Ow, Tcw,
                                        d_KeyFrame, d_MapPoints, 
                                        mapPointVecSize, th, 
                                        d_bestDists, d_bestIdxs);
    
    cudaDeviceSynchronize(); // ensure kernel errors propagate
    auto end6 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed6 = end6 - start6;
    // timing << "? Kernel: " << elapsed6.count() << " ms" << std::endl;

    auto start7 = std::chrono::high_resolution_clock::now();
    checkCudaError(cudaMemcpy(bestDists, d_bestDists, mapPointVecSize * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestDists back to host");
    checkCudaError(cudaMemcpy(bestIdxs, d_bestIdxs, mapPointVecSize * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestIdxs back to host");
    auto end7 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed7 = end7 - start7;
    // timing << "? cudaMemcpy back: " << elapsed7.count() << " ms" << std::endl;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    // std::ofstream gpuOutFile("./test/GPU-Side.txt", std::ios::app);
    // gpuOutFile << "\n\n////////////////////////////////////////// Current KF: " << pKF->mnId << " //////////////////////////////////////////" << endl;
    
    // std::ofstream cpuOutFile("./test/CPU-Side.txt", std::ios::app);
    // cpuOutFile << "\n\n////////////////////////////////////////// Current KF: " << pKF->mnId << " //////////////////////////////////////////" << endl;
    
    // const set<ORB_SLAM3::MapPoint*> spAlreadyFound = pKF->GetMapPoints();

    // for (int i = 0; i < numValidPoints; i++) {
    //     ORB_SLAM3::MapPoint* pMP = vpPoints[i];
    //     if(spAlreadyFound.count(pMP))
    //             continue;

    //     if(bestDists[i] != 256)
    //         gpuOutFile << "(i: " << i << ", bestDist: " << bestDists[i] << ", bestIdx: " << bestIdxs[i] << ")\n";
    // }
    // gpuOutFile << "\n\n";
    
    // origSearchByProjection2(pKF, Scw, vpPoints, vpMatched, th, ratioHamming);

    // gpuOutFile << "**********************************************************\n";
    // cpuOutFile << "**********************************************************\n";

    auto start8 = std::chrono::high_resolution_clock::now();
    const set<ORB_SLAM3::MapPoint*> spAlreadyFound = pKF->GetMapPoints();
    for(size_t iMP = 0; iMP < mapPointVecSize; iMP++) {
        ORB_SLAM3::MapPoint* pMP = vpPoints[iMP];
        int bestDist = bestDists[iMP];
        int bestIdx = bestIdxs[iMP];

        if(spAlreadyFound.count(pMP))
                continue;

        if (bestDist == 256 || bestIdx == -1)
            continue;

        if (bestDist <= TH_LOW*ratioHamming) {
            vpMatched[bestIdx] = pMP;
            nmatches++;
        }
    }
    auto end8 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed8 = end8 - start8;
    // timing << "? result: " << elapsed8.count() << " ms" << std::endl;

    
    // cudaFreeHost(h_MapPoints);
    // cudaFreeHost(h_KeyFrame);
    // cudaFreeHost(bestDists);
    // cudaFreeHost(bestIdxs);
    // cudaFree(d_MapPoints);
    // cudaFree(d_KeyFrame);
    // cudaFree(d_bestDists);
    // cudaFree(d_bestIdxs);

    return nmatches;
}

void SearchByProjectionKernel::mergedlaunch(ORB_SLAM3::KeyFrame* pKF, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints,
                        Sophus::Sim3<float> &Scw, const std::vector<ORB_SLAM3::KeyFrame*> &vpPointsKFs, std::vector<ORB_SLAM3::MapPoint*> &vpMatched, std::vector<ORB_SLAM3::KeyFrame*> &vpMatchedKF, int th, float ratioHamming,
                        Sophus::Sim3<float> &Scw1, std::vector<ORB_SLAM3::MapPoint*> &vpMatched1, int th1, float ratioHamming1,
                        int &numProjMatches, int &numProjOptMatches)
{

    std::ofstream timing("./test/timing.txt", std::ios::app);

    auto start1 = std::chrono::high_resolution_clock::now();
    if (!memory_is_initialized)
        initialize();

    int numValidPoints = 0;
    const int TH_LOW = 50;
    numProjOptMatches = 0;
    numProjMatches = 0;

    size_t mapPointVecSize = vpPoints.size();
    // timing << "mapPointVecSize: " << mapPointVecSize << std::endl;
    // cout << "mapPointVecSize: " << mapPointVecSize << endl;

    // LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint *h_MapPoints, *d_MapPoints;
    // CudaKeyFrame *h_KeyFrame, *d_KeyFrame;
    // int *d_bestDists1, *d_bestIdxs1;
    // int *bestDists1, *bestIdxs1;
    // int *d_bestDists, *d_bestIdxs;
    // int *bestDists, *bestIdxs;
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed1 = end1 - start1;
    // timing << "? Initialization 1: " << elapsed1.count() << " ms" << std::endl;

    // auto start2 = std::chrono::high_resolution_clock::now();
    // cudaMallocHost((void**)&h_MapPoints, mapPointVecSize * sizeof(LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint));
    // cudaMallocHost((void**)&h_KeyFrame, sizeof(CudaKeyFrame));
    // cudaMallocHost((void**)&bestDists1, mapPointVecSize * sizeof(int));
    // cudaMallocHost((void**)&bestIdxs1, mapPointVecSize * sizeof(int));
    // cudaMallocHost((void**)&bestDists, mapPointVecSize * sizeof(int));
    // cudaMallocHost((void**)&bestIdxs, mapPointVecSize * sizeof(int));
    // auto end2 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> elapsed2 = end2 - start2;
    // timing << "? cudaMallocHost: " << elapsed2.count() << " ms" << std::endl;

    // auto start3 = std::chrono::high_resolution_clock::now();
    // cudaMalloc(&d_MapPoints, mapPointVecSize * sizeof(LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint));
    // cudaMalloc(&d_KeyFrame, sizeof(CudaKeyFrame));
    // cudaMalloc(&d_bestDists1, mapPointVecSize * sizeof(int));
    // cudaMalloc(&d_bestIdxs1, mapPointVecSize * sizeof(int));
    // cudaMalloc(&d_bestDists, mapPointVecSize * sizeof(int));
    // cudaMalloc(&d_bestIdxs, mapPointVecSize * sizeof(int));
    // auto end3 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> elapsed3 = end3 - start3;
    // timing << "? cudaMalloc: " << elapsed3.count() << " ms" << std::endl;

    auto start4 = std::chrono::high_resolution_clock::now();
    Sophus::SE3f Tcw1 = Sophus::SE3f(Scw1.rotationMatrix(),Scw1.translation()/Scw1.scale());
    Eigen::Vector3f Ow1 = Tcw1.inverse().translation();
    Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(),Scw.translation()/Scw.scale());
    Eigen::Vector3f Ow = Tcw.inverse().translation();
    auto end4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed4 = end4 - start4;
    // timing << "? Tcw: " << elapsed4.count() << " ms" << std::endl;

    // set<ORB_SLAM3::MapPoint*> spAlreadyFound1(vpMatched1.begin(), vpMatched1.end());
    // spAlreadyFound1.erase(static_cast<ORB_SLAM3::MapPoint*>(NULL));
    
    auto start5 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < mapPointVecSize; i++) {
        ORB_SLAM3::MapPoint* pMP = vpPoints[i];
        if (!pMP || pMP->isBad() )//|| spAlreadyFound1.count(pMP)) //todo1
            continue;
        else {
            // printf("mnId = %llu\n", pMP->mnId);
            // printf("Before Host: mfMinDistance = %f, mfMaxDistance = %f\n", pMP->mfMinDistance, pMP->mfMaxDistance);
            h_MapPoints[numValidPoints] = LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint(pMP);
            // printf("After Host: mfMinDistance = %f, mfMaxDistance = %f\n", h_MapPoints[numValidPoints].mfMinDistance, h_MapPoints[numValidPoints].mfMaxDistance);
            numValidPoints++;
        }
    }
    auto end5 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed5 = end5 - start5;
    // timing << "? CudaMapPoint: " << elapsed5.count() << " ms" << std::endl;

    auto start6 = std::chrono::high_resolution_clock::now();
    // *h_KeyFrame = CudaKeyFrame();
    // h_KeyFrame->setMemory(pKF);
    d_KeyFrame = LoopClosingCudaKeyFrameStorage::getCudaKeyFrame(pKF->mnId);
    if (d_KeyFrame == nullptr){
        // timing << "No " << pKF->mnId << std::endl;
        d_KeyFrame = LoopClosingCudaKeyFrameStorage::addCudaKeyFrame(pKF);
    }
    // else{
    //     timing << "Yes " << pKF->mnId << std::endl;
    // }
    auto end6 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed6 = end6 - start6;
    // timing << "? h_KeyFrame Merged: " << elapsed6.count() << " ms" << std::endl;

    // auto start7 = std::chrono::high_resolution_clock::now();
    // cudaStream_t stream;
    // cudaStreamCreate(&stream);
    cudaMemcpy(d_MapPoints, h_MapPoints, numValidPoints * sizeof(LOOP_CLOSING_DATA_WRAPPER::CudaMapPoint), cudaMemcpyHostToDevice); //todo2
    // cudaMemcpy(d_KeyFrame, h_KeyFrame, sizeof(CudaKeyFrame), cudaMemcpyHostToDevice);
    // auto end7 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> elapsed7 = end7 - start7;
    // timing << "? cudaMemcpy: " << elapsed7.count() << " ms" << std::endl;

    int threads = 256;
    int blocks = (numValidPoints + threads - 1) / threads;

    auto start75 = std::chrono::high_resolution_clock::now();
    mergedSearchByProjectionKernel<<<blocks, threads>>>(Ow1, Tcw1, Ow, Tcw,
                                        d_KeyFrame, d_MapPoints, 
                                        numValidPoints, th1, th, 
                                        d_bestDists1, d_bestIdxs1, d_bestDists, d_bestIdxs);
    
    cudaDeviceSynchronize(); // ensure kernel errors propagate
    auto end75 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed75 = end75 - start75;
    // timing << "? Merged Kernel 1: " << elapsed75.count() << " ms" << "\n";

    auto start8 = std::chrono::high_resolution_clock::now();
    checkCudaError(cudaMemcpy(bestDists1, d_bestDists1, numValidPoints * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestDists back to host"); //todo3
    checkCudaError(cudaMemcpy(bestIdxs1, d_bestIdxs1, numValidPoints * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestIdxs back to host"); //todo4
    checkCudaError(cudaMemcpy(bestDists, d_bestDists, numValidPoints * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestDists back to host"); //todo5
    checkCudaError(cudaMemcpy(bestIdxs, d_bestIdxs, numValidPoints * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bestIdxs back to host"); //todo6
    auto end8 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed8 = end8 - start8;
    // timing << "? back cudaMemcpy: " << elapsed8.count() << " ms" << std::endl;

    // auto start9 = std::chrono::high_resolution_clock::now();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    // cudaStreamSynchronize(stream);
    // cudaStreamDestroy(stream);
    // auto end9 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> elapsed9 = end9 - start9;
    // timing << "? cudaGetLastError: " << elapsed9.count() << " ms" << std::endl;

    // std::ofstream gpuOutFile("./test/GPU-Side.txt", std::ios::app);
    // gpuOutFile << "\n\n////////////////////////////////////////// Current KF: " << pKF->mnId << " //////////////////////////////////////////" << endl;
    
    // std::ofstream cpuOutFile("./test/CPU-Side.txt", std::ios::app);
    // cpuOutFile << "\n\n////////////////////////////////////////// Current KF: " << pKF->mnId << " //////////////////////////////////////////" << endl;
        
    // const set<ORB_SLAM3::MapPoint*> spAlreadyFound = pKF->GetMapPoints();

    // for (int i = 0; i < numValidPoints; i++) {
    //     ORB_SLAM3::MapPoint* pMP = vpMapPoints[i];
        
    //     if(spAlreadyFound.count(pMP))
    //             continue;

    //     if(bestDists[i] != 256)
    //         gpuOutFile << "(i: " << i << ", bestDist: " << bestDists[i] << ", bestIdx: " << bestIdxs[i] << ")\n";
    // }
    // gpuOutFile << "\n\n";
    
    // for (int i = 0; i < numValidPoints; i++) {
    //     ORB_SLAM3::MapPoint* pMP = vpMapPoints[i];
        
    //     if(spAlreadyFound.count(pMP))
    //             continue;
    
    //     if(bestDists1[i] != 256)
    //         gpuOutFile << "(i: " << i << ", bestDist1: " << bestDists1[i] << ", bestIdx1: " << bestIdxs1[i] << ")\n";
    // }
    // gpuOutFile << "\n\n";

    // origSearchByProjection(pKF, Scw, vpPoints, vpPointsKFs, vpMatched, vpMatchedKF, th, ratioHamming);
    // origSearchByProjection2(pKF, Scw1, vpPoints, vpMatched1, th1, ratioHamming1);

    // gpuOutFile << "**********************************************************\n";
    // cpuOutFile << "**********************************************************\n";

    // auto start10 = std::chrono::high_resolution_clock::now();
    const set<ORB_SLAM3::MapPoint*> spAlreadyFound = pKF->GetMapPoints();

    for(size_t iMP = 0; iMP < mapPointVecSize; iMP++) {
        ORB_SLAM3::MapPoint* pMP = vpPoints[iMP];
        ORB_SLAM3::KeyFrame* pKFi = vpPointsKFs[iMP];
        int bestDist = bestDists[iMP];
        int bestIdx = bestIdxs[iMP];

        if (!pMP || pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        if (bestDist == 256 || bestIdx == -1)
            continue;

        if (bestDist <= TH_LOW*ratioHamming) {
            vpMatched[bestIdx] = pMP;
            vpMatchedKF[bestIdx] = pKFi;
            numProjMatches++;
        }
    }

    for(size_t iMP = 0; iMP < mapPointVecSize; iMP++) {
        ORB_SLAM3::MapPoint* pMP = vpPoints[iMP];
        int bestDist1 = bestDists1[iMP];
        int bestIdx1 = bestIdxs1[iMP];

        if (!pMP || pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        if (bestDist1 == 256 || bestIdx1 == -1)
            continue;
        
        if (bestDist1 <= TH_LOW*ratioHamming1) {
            vpMatched1[bestIdx1] = pMP;
            numProjOptMatches++;
        }
    }
    // auto end10 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> elapsed10 = end10 - start10;
    // timing << "? result: " << elapsed10.count() << " ms" << std::endl;
    
    // auto start11 = std::chrono::high_resolution_clock::now();
    // cudaFreeHost(h_MapPoints);
    // cudaFreeHost(h_KeyFrame);
    // cudaFreeHost(bestDists1);
    // cudaFreeHost(bestIdxs1);
    // cudaFreeHost(bestDists);
    // cudaFreeHost(bestIdxs);
    // cudaFree(d_MapPoints);
    // cudaFree(d_KeyFrame);
    // cudaFree(d_bestDists1);
    // cudaFree(d_bestIdxs1);
    // cudaFree(d_bestDists);
    // cudaFree(d_bestIdxs);
    // auto end11 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> elapsed11 = end11 - start11;
    // timing << "? cudaFree: " << elapsed11.count() << " ms" << std::endl;

}

void SearchByProjectionKernel::origSearchByProjection(ORB_SLAM3::KeyFrame* pKF, Sophus::Sim3<float> &Scw, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints, const std::vector<ORB_SLAM3::KeyFrame*> &vpPointsKFs,
                                       std::vector<ORB_SLAM3::MapPoint*> &vpMatched, std::vector<ORB_SLAM3::KeyFrame*> &vpMatchedKF, int th, float ratioHamming)
{
    
    std::ofstream cpuOutFile("./test/CPU-Side.txt", std::ios::app);
    int validMapPointCounter = -1;

    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(),Scw.translation()/Scw.scale());
    Eigen::Vector3f Ow = Tcw.inverse().translation();

    // Set of MapPoints already found in the KeyFrame
    set<ORB_SLAM3::MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<ORB_SLAM3::MapPoint*>(NULL));

    int nmatches=0;

    // For each Candidate MapPoint Project and Match
    for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
    {
        ORB_SLAM3::MapPoint* pMP = vpPoints[iMP];
        ORB_SLAM3::KeyFrame* pKFi = vpPointsKFs[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;
        validMapPointCounter++;

        // Get 3D Coords.
        Eigen::Vector3f p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        Eigen::Vector3f p3Dc = Tcw * p3Dw;

        // Depth must be positive
        if(p3Dc(2)<0.0){
            continue;
        }

        // Project into Image
        const float invz = 1/p3Dc(2);
        const float x = p3Dc(0)*invz;
        const float y = p3Dc(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        Eigen::Vector3f PO = p3Dw-Ow;
        const float dist = PO.norm();

        // cpuOutFile << "Host: idx = " << validMapPointCounter << ", id = " << pMP->mnId << ", dist = " << dist << ", minDistance = " << minDistance << ", maxDistance = " << maxDistance << endl;

        if(dist<minDistance || dist>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        Eigen::Vector3f Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];
        // cpuOutFile << "Host: idx = " << validMapPointCounter << ", id = " << pMP->mnId << ", radius = " << radius << endl;
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();
        
        // cpuOutFile << "Host: idx = " << validMapPointCounter << ", id = " << pMP->mnId << endl;

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;
            if(vpMatched[idx])
                continue;

            const int &kpLevel= pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = origDescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if (bestDist != 256)
            cpuOutFile << "(i: " << validMapPointCounter << ", bestDist: " << bestDist << ", bestIdx: " << bestIdx << ")\n";
    }
    cpuOutFile << "\n\n";         
}


void SearchByProjectionKernel::origSearchByProjection2(ORB_SLAM3::KeyFrame* pKF, Sophus::Sim3<float> &Scw, const std::vector<ORB_SLAM3::MapPoint*> &vpPoints,
                    std::vector<ORB_SLAM3::MapPoint*> &vpMatched, int th, float ratioHamming)
{
    
    std::ofstream cpuOutFile("./test/CPU-Side.txt", std::ios::app);
    int validMapPointCounter = -1;

    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(),Scw.translation()/Scw.scale());
    Eigen::Vector3f Ow = Tcw.inverse().translation();

    // Set of MapPoints already found in the KeyFrame
    set<ORB_SLAM3::MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<ORB_SLAM3::MapPoint*>(NULL));

    int nmatches=0;

    // For each Candidate MapPoint Project and Match
    for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
    {
        ORB_SLAM3::MapPoint* pMP = vpPoints[iMP];
        

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;
        validMapPointCounter++;

        // Get 3D Coords.
        Eigen::Vector3f p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        Eigen::Vector3f p3Dc = Tcw * p3Dw;
        
        // Depth must be positive
        if(p3Dc(2)<0.0){
            continue;
        }

        // Project into Image
        const Eigen::Vector2f uv = pKF->mpCamera->project(p3Dc);

        // Point must be inside the image
        if(!pKF->IsInImage(uv(0),uv(1)))
            continue;

        // Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        Eigen::Vector3f PO = p3Dw-Ow;
        const float dist = PO.norm();

        if(dist<minDistance || dist>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        Eigen::Vector3f Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0),uv(1),radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();
        
        // cpuOutFile << "Host: idx = " << validMapPointCounter << ", id = " << pMP->mnId << endl;

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;
            if(vpMatched[idx])
                continue;

            const int &kpLevel= pKF->mvKeysUn[idx].octave;
            // cpuOutFile << "Host: idx = " << validMapPointCounter << ", id = " << pMP->mnId << ", kpLevel = " << kpLevel << endl;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = origDescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if (bestDist != 256)
            cpuOutFile << "(i: " << validMapPointCounter << ", bestDist1: " << bestDist << ", bestIdx1: " << bestIdx << ")\n";
    }
    cpuOutFile << "\n\n";         
}
    

int SearchByProjectionKernel::origDescriptorDistance(const cv::Mat &a, const cv::Mat &b) {
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