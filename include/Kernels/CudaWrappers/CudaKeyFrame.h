#ifndef CUDA_KEYFRAME_H
#define CUDA_KEYFRAME_H

#include "CudaKeyPoint.h"
#include "CudaMapPoint.h"
#include "CudaCamera.h"
#include "KeyFrame.h"
#include "../CudaUtils.h"


#define MAX_FEAT_VEC_SIZE 100
#define MAX_FEAT_PER_WORD 100
#define KEYPOINTS_PER_CELL 20


class CudaKeyFrame {
    private:
        void initializeMemory();
        void copyGPUCamera(MAPPING_DATA_WRAPPER::CudaCamera *out, ORB_SLAM3::GeometricCamera *camera);
        void copyFeatVec(unsigned int *out, int *outIndexes, DBoW2::FeatureVector inp);

    public:
        CudaKeyFrame();
        void setGPUAddress(CudaKeyFrame* ptr);
        void setMemory(ORB_SLAM3::KeyFrame* KF);
        void addFeatureVector(DBoW2::FeatureVector featVec);
        void setAsEmpty() { isEmpty = true; };
        void freeMemory();

    public:
        bool isEmpty;
        long unsigned int mnId;
        int Nleft;
        float mfLogScaleFactor;
        int mnScaleLevels;
        float mnMinX;
        float mnMaxX;
        float mnMinY;
        float mnMaxY;
        float mfGridElementWidthInv;
        float mfGridElementHeightInv;
        int mnGridCols;
        int mnGridRows;
        float fx;
        float fy;
        float cx;
        float cy;

        size_t mapPointsId_size;
        long unsigned int* mapPointsId;

        size_t mvScaleFactors_size;
        float* mvScaleFactors;

        size_t mvKeys_size, mvKeysRight_size, mvKeysUn_size;
        const CudaKeyPoint *mvKeys, *mvKeysRight;
        CudaKeyPoint *mvKeysUn;

        int mDescriptor_rows;
        const uint8_t* mDescriptors;

        size_t flatMGrid_size[FRAME_GRID_COLS * FRAME_GRID_ROWS];
        std::size_t flatMGrid[FRAME_GRID_COLS * FRAME_GRID_ROWS * KEYPOINTS_PER_CELL];        
        
        MAPPING_DATA_WRAPPER::CudaCamera camera1;

        int mFeatCount;
        unsigned int *mFeatVec;
        int *mFeatVecStartIndexes;

        long unsigned int *temp_mapPointsId;
};


#endif // CUDA_KEYFRAME_H