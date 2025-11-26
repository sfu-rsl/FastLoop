#pragma once

// This file contains types used for 4DoF pose graph optimization in Graphite

#include <graphite/core.hpp>
#include <graphite/types.hpp>
#include <graphite/utils.hpp>

#include "GPUPose.h"

namespace gpu {

    using namespace graphite;

    // Pose
    template <typename T>
    class Pose4DoF {
        public:
        // Methods
        Pose4DoF() {}

        Pose4DoF(ORB_SLAM3::KeyFrame *pKF) : its(0)
        {

            // Load IMU pose
            twb = pKF->GetImuPosition().cast<T>();
            Rwb = pKF->GetImuRotation().cast<T>();

            // Left camera
            tcw[0] = pKF->GetTranslation().cast<T>();
            Rcw[0] = pKF->GetRotation().cast<T>();
            tcb[0] = pKF->mImuCalib.mTcb.translation().cast<T>();
            Rcb[0] = pKF->mImuCalib.mTcb.rotationMatrix().cast<T>();
            Rbc[0] = Rcb[0].transpose();
            tbc[0] = pKF->mImuCalib.mTbc.translation().cast<T>();
            bf = pKF->mbf;


            // For posegraph 4DoF
            Rwb0 = Rwb;
            DR.setIdentity();   
        }

        Pose4DoF(Mat3<T> &_Rwc, Vec3<T> &_twc, ORB_SLAM3::KeyFrame* pKF): its(0)
        {

            tcb[0] = pKF->mImuCalib.mTcb.translation().cast<T>();
            Rcb[0] = pKF->mImuCalib.mTcb.rotationMatrix().cast<T>();
            Rbc[0] = Rcb[0].transpose();
            tbc[0] = pKF->mImuCalib.mTbc.translation().cast<T>();
            twb = _Rwc * tcb[0] + _twc;
            Rwb = _Rwc * Rcb[0];
            Rcw[0] = _Rwc.transpose();
            tcw[0] = -Rcw[0] * _twc;
            // pCamera[0] = pKF->mpCamera;
            bf = pKF->mbf;

            // For posegraph 4DoF
            Rwb0 = Rwb;
            DR.setIdentity();
        }

        void UpdateW(const T *pu)
        {
            Vec3<T> ur, ut;
            ur << pu[0], pu[1], pu[2];
            ut << pu[3], pu[4], pu[5];


            const Mat3<T> dR = ExpSO3(ur);
            DR = dR * DR;
            Rwb = DR * Rwb0;
            // Update body pose
            twb += ut;

            // Normalize rotation after 5 updates
            its++;
            if(its>=5)
            {
                DR(0,2) = 0.0;
                DR(1,2) = 0.0;
                DR(2,0) = 0.0;
                DR(2,1) = 0.0;
                NormalizeRotation(DR);
                its = 0;
            }

            // Update camera pose
            const Mat3<T> Rbw = Rwb.transpose();
            const Vec3<T> tbw = -Rbw * twb;

            for(int i=0; i< num_cams; i++)
            {
                Rcw[i] = Rcb[i] * Rbw;
                tcw[i] = Rcb[i] * tbw+tcb[i];
            }
        }

    public:
        // Member variables

        // For IMU
        Mat3<T> Rwb;
        Vec3<T> twb;

        // For set of cameras
        static constexpr size_t num_cams = 1;
        static constexpr size_t max_cams = 1;
        std::array<Mat3<T>, max_cams>  Rcw;
        std::array<Vec3<T>, max_cams> tcw;
        std::array<Mat3<T>, max_cams> Rcb, Rbc;
        std::array<Vec3<T>, max_cams> tcb, tbc;
        T bf;

        // For posegraph 4DoF
        Mat3<T> Rwb0;
        Mat3<T> DR;

        int its;

    };


    template <typename T> struct Pose4DoFTraits {
        static constexpr size_t dimension = 4;
        using Vertex = Pose4DoF<T>;

        template <typename P>
        hd_fn static void parameters(const Vertex &vertex, P* params) {
            // // First parameter is for yaw and last three are for position
            // // We only deal with Rcw[0] and tcw[0] for 4DoF optimization
            // Vec3<T> rvec = LogSO3(vertex.Rcw[0]);

            // params[0] = rvec[2]; // yaw

            // // position
            // params[1] = vertex.tcw[0](0);
            // params[2] = vertex.tcw[0](1);
            // params[3] = vertex.tcw[0](2);
            
        }

        hd_fn static void update(Vertex &vertex, const T *delta) {
            T update6DoF[6];
            update6DoF[0] = 0;
            update6DoF[1] = 0;
            update6DoF[2] = delta[0];
            update6DoF[3] = delta[1];
            update6DoF[4] = delta[2];
            update6DoF[5] = delta[3];
            vertex.UpdateW(delta);
        }
    };

    template <typename T, typename S>
    using Pose4DoFDescriptor = VertexDescriptor<T, S, Pose4DoFTraits<T>>;


    // template <typename T>
    // class Factor4DoFData {
    //     public:

    //     Factor4DoFData(const Mat4<T> &deltaT) {
    //         dTij = deltaT;
    //         Rij = dTij.block<3, 3>(0, 0);
    //         dtij = dTij.block<3, 1>(0, 3);
    //     }

    //     Mat4<T> dTij;
    //     Mat3<T> Rij;
    //     Vec3<T> dtij;

    // };

    template <typename T, typename S, typename L> struct Factor4DoFTraits {
    static constexpr size_t dimension = 6;
    using VertexDescriptors = std::tuple<Pose4DoFDescriptor<T, S>, Pose4DoFDescriptor<T, S>>;
    using Observation = Mat4<T>;
    using Data = Empty;
    using Loss = L;
    using Differentiation = DifferentiationMode::Manual;
    // using Differentiation = DifferentiationMode::Auto;

    using Pose = typename Pose4DoFDescriptor<T, S>::VertexType;

    template <typename D, typename M>
    hd_fn static void
    error(const D *p1, const D *p2, const M *obs, D *error,
            const std::tuple<Pose *, Pose*> &vertices, const Data *data) {
        // const Pose* VPi = std::get<0>(vertices);
        // const Pose* VPj = std::get<1>(vertices);
        const auto yaw_i = p1[0];
        Vec3<D> ti; 
        ti << p1[1], p1[2], p1[3];

        Vec3<D> rvec_i;
        rvec_i << D(0.0), D(0.0), yaw_i;


        const auto yaw_j = p2[0];
        Vec3<D> tj;
        tj << p2[1], p2[2], p2[3];

        Vec3<D> rvec_j;
        rvec_j << D(0.0), D(0.0), yaw_j;

        Mat3<D> Ri = ExpSO3(rvec_i);
        Mat3<D> Rj = ExpSO3(rvec_j);

        Eigen::Map<Eigen::Matrix<D, 6, 1>> err(error);

        const auto Rij = (*obs).template block<3, 3>(0, 0);
        const auto dtij = (*obs).template block<3, 1>(0, 3);

        err << LogSO3<D>(Ri * Rj.transpose() * Rij.transpose()),
               Ri * (-Rj.transpose() * tj) + ti - dtij;

        // _error << LogSO3(VPi->Rcw[0]*VPj->Rcw[0].transpose()*dRij.transpose()),
        //          VPi->Rcw[0]*(-VPj->Rcw[0].transpose()*VPj->tcw[0])+VPi->tcw[0] - dtij;

    }

    template <typename D, size_t I>
    hd_fn static void jacobian(const Pose *p1, const Pose *p2,
                                const Observation *obs, D *jacobian,
                                const Data *data) {

        if constexpr (I == 0) {
            Eigen::Map<Eigen::Matrix<D, 3, 3>> J(jacobian);
            J = Eigen::Matrix<D, 3, 3>::Identity(); // shouldn't this be negative?
        }
        else {
            
        }

    }

    };

    template <typename T, typename S, typename L>
    using Factor4DoFDescriptor = FactorDescriptor<T, S, Factor4DoFTraits<T, S, L>>;



}