#include <iostream>
#include "Kernels/SearchByBoWKernel.h"


void SearchByBoWKernel::initialize()
{
    if (memory_is_initialized)
        return;
    
    
}


void SearchByBoWKernel::shutdown()
{
    if (!memory_is_initialized) 
        return;

}




__global__ void searchByBoWKernel() 
{
    
                
}


int SearchByBoWKernel::launch(ORB_SLAM3::KeyFrame *pKF1, ORB_SLAM3::KeyFrame *pKF2, vector<ORB_SLAM3::MapPoint *> &vpMatches12)
{
    if (!memory_is_initialized)
        initialize();

    int nmatches = 0;

    // const int MAX_FEATURES = 2000;
    // std::vector<unsigned int> indices1[MAX_FEATURES]; // array of vectors
    // std::vector<unsigned int> indices2[MAX_FEATURES]; // same for indices2
    // int numIndices = 0;
    
    // const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
    // const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    // const vector<ORB_SLAM3::MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    // const cv::Mat &Descriptors1 = pKF1->mDescriptors;

    // const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
    // const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
    // const vector<ORB_SLAM3::MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    // const cv::Mat &Descriptors2 = pKF2->mDescriptors;

    // vpMatches12 = vector<ORB_SLAM3::MapPoint*>(vpMapPoints1.size(),static_cast<ORB_SLAM3::MapPoint*>(NULL));
    // vector<bool> vbMatched2(vpMapPoints2.size(),false);

    // // vector<int> rotHist[HISTO_LENGTH];
    // // for(int i=0;i<HISTO_LENGTH;i++)
    // //     rotHist[i].reserve(500);

    // // const float factor = 1.0f/HISTO_LENGTH;

    // DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    // DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    // DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    // DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();
        
    // while (f1it != f1end && f2it != f2end) 
    // {
    //     if (f1it->first == f2it->first)
    //     {
    //         indices1[numIndices] = f1it->second;
    //         indices2[numIndices] = f2it->second;
    //         numIndices++;
    //         f1it++;
    //         f2it++;
    //     }
    //     else if (f1it->first < f2it->first)
    //     {
    //         f1it = vFeatVec1.lower_bound(f2it->first);
    //     }
    //     else
    //     {
    //         f2it = vFeatVec2.lower_bound(f1it->first);
    //     }
    // }

    // int* bestDist1 = new int[numIndices];
    // int* bestIdx2  = new int[numIndices];
    // int* bestDist2 = new int[numIndices];

    // for(size_t i=0; i<numIndices; i++){
    //     for(size_t i1=0; i1<indices1[i].size(); i1++)
    //     {
    //         const size_t idx1 = indices1[i][i1];
    //         if(pKF1 -> NLeft != -1 && idx1 >= pKF1 -> mvKeysUn.size()){
    //             continue;
    //         }

    //         ORB_SLAM3::MapPoint* pMP1 = vpMapPoints1[idx1];
    //         if(!pMP1)
    //             continue;
    //         if(pMP1->isBad())
    //             continue;

    //         const cv::Mat &d1 = Descriptors1.row(idx1);

    //         bestDist1[i1]=256;
    //         bestIdx2[i1]=-1;
    //         bestDist2[i1]=256;

    //         for(size_t i2=0; i2<indices2[i].size(); i2++)
    //         {
    //             const size_t idx2 = indices2[i][i2];

    //             if(pKF2 -> NLeft != -1 && idx2 >= pKF2 -> mvKeysUn.size()){
    //                 continue;
    //             }

    //             ORB_SLAM3::MapPoint* pMP2 = vpMapPoints2[idx2];

    //             if(vbMatched2[idx2] || !pMP2)
    //                 continue;

    //             if(pMP2->isBad())
    //                 continue;

    //             const cv::Mat &d2 = Descriptors2.row(idx2);

    //             int dist = origDescriptorDistance(d1,d2);

    //             if(dist<bestDist1[i1])
    //             {
    //                 bestDist2[i1]=bestDist1[i1];
    //                 bestDist1[i1]=dist;
    //                 bestIdx2[i1]=idx2;
    //             }
    //             else if(dist<bestDist2[i1])
    //             {
    //                 bestDist2[i1]=dist;
    //             }
    //         }
    //     }
    // }

    // for(size_t i1=0; i1<numIndices; i1++)
    // {
    //     const size_t idx1 = indices1[i1];
    //     if(pKF1 -> NLeft != -1 && idx1 >= pKF1 -> mvKeysUn.size()){
    //         continue;
    //     }

    //     ORB_SLAM3::MapPoint* pMP1 = vpMapPoints1[idx1];
    //     if(!pMP1)
    //         continue;
    //     if(pMP1->isBad())
    //         continue;

    //     if(bestDist1[i1]<TH_LOW)
    //     {
    //         if(static_cast<float>(bestDist1[i1])<mfNNratio*static_cast<float>(bestDist2[i1]))
    //         {
    //             vpMatches12[idx1]=vpMapPoints2[bestIdx2[i1]];
    //             vbMatched2[bestIdx2[i1]]=true;

    //             if(mbCheckOrientation)
    //             {
    //                 float rot = vKeysUn1[idx1].angle-vKeysUn2[bestIdx2].angle;
    //                 if(rot<0.0)
    //                     rot+=360.0f;
    //                 int bin = round(rot*factor);
    //                 if(bin==HISTO_LENGTH)
    //                     bin=0;
    //                 assert(bin>=0 && bin<HISTO_LENGTH);
    //                 rotHist[bin].push_back(idx1);
    //             }
    //             nmatches++;
    //         }
    //     }
    // }

    // delete[] bestDist1;
    // delete[] bestIdx2;
    // delete[] bestDist2;


    // int threads = 256;
    // int blocks = (mapPointVecSize + threads - 1) / threads;

    // searchByBoWKernel<<<blocks, threads>>>(d_KeyFrame1, d_KeyFrame2, 
    //                                     mapPointVecSize, th1, th, 
    //                                     d_bestDists1, d_bestIdxs1, d_bestDists, d_bestIdxs);
    
    return nmatches;
}


void SearchByBoWKernel::origSearchByBoW(ORB_SLAM3::KeyFrame *pKF1, ORB_SLAM3::KeyFrame *pKF2, vector<ORB_SLAM3::MapPoint *> &vpMatches12)
{
    
    
}


int SearchByBoWKernel::origDescriptorDistance(const cv::Mat &a, const cv::Mat &b) {
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