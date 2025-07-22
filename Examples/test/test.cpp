#include "../../include/LoopClosing.h" 
#include <iostream>
#include <map>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <g2o/types/sim3/sim3.h>

typedef std::map<class KeyFrame*, g2o::Sim3, std::less<KeyFrame*>,
        Eigen::aligned_allocator<std::pair<KeyFrame* const, g2o::Sim3>>> KeyFrameAndPose;


// ---------------------- Mock Classes ----------------------
class KeyFrame {
public:
    int id;
    KeyFrame(int _id) : id(_id) {}
};


class MapPoint {
public:
    int id;
    bool fused = false;

    MapPoint(int _id) : id(_id) {}
    void SetFused(bool val) { fused = val; }
    bool IsFused() const { return fused; }
};


// ---------------------- Main Test ----------------------
int main() {
    KeyFrameAndPose correctedSim3;

    // ساخت KeyFrame و Sim3
    KeyFrame* kf = new KeyFrame(1);
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t(0, 0, 0);
    double s = 1.0;
    g2o::Sim3 sim3(R, t, s);

    correctedSim3[kf] = sim3;

    // ساخت MapPoints
    std::vector<MapPoint*> mapPoints;
    for (int i = 0; i < 5; ++i) {
        mapPoints.push_back(new MapPoint(i));
    }

    // صدا زدن تابع
    GPUSearchAndFuse(correctedSim3, mapPoints);

    // چک کردن خروجی
    for (auto mp : mapPoints) {
        std::cout << "MapPoint " << mp->id 
                  << " fused: " << (mp->IsFused() ? "yes" : "no") << std::endl;
    }

    // پاکسازی
    delete kf;
    for (auto mp : mapPoints) {
        delete mp;
    }

    return 0;
}
