#pragma once

#include <Eigen/Eigen>
#include <sophus/se3.hpp>
#include "Utils.h"
#include <calibu/Calibu.h>

namespace ba
{

template<int LmSize>
struct PoseT
{
    Sophus::SE3d Twp;
    Eigen::Vector3d V;
    bool IsActive;
    unsigned int Id;
    unsigned int OptId;
    double Time;
    std::vector<int> ProjResiduals;
    std::vector<int> ImuResiduals;
    std::vector<int> BinaryResiduals;
    std::vector<int> UnaryResiduals;
    std::vector<Sophus::SE3d> Tsw;

    const Sophus::SE3d& GetTsw(const unsigned int camId, const calibu::CameraRig& rig)
    {
        while(Tsw.size() <= camId ){
          Tsw.push_back( (Twp*rig.cameras[Tsw.size()].T_wc).inverse());
        }
        return Tsw[camId];
    }
};

template<int LmSize>
struct LandmarkT
{
    Eigen::Vector4d Xs;
    Eigen::Vector4d Xw;
    std::vector<int> ProjResiduals;
    unsigned int OptId;
    unsigned int RefPoseId;
    unsigned int RefCamId;
};

struct ImuCalibration
{
    ImuCalibration(const Sophus::SE3d& tvi, const Eigen::Vector3d& bg, const Eigen::Vector3d& ba, const Eigen::Vector2d& g):
        Tvi(tvi),Bg(bg),Ba(ba),G(g) {}
    ///
    /// \brief Calibration from vehicle to inertial reference frame
    ///
    Sophus::SE3d Tvi;
    ///
    /// \brief Gyroscope bias vector
    ///
    Eigen::Vector3d Bg;
    ///
    /// \brief Accelerometer bias vector
    ///
    Eigen::Vector3d Ba;
    ///
    /// \brief Gravity vector (2D, parametrized by roll and pitch of the vector wrt the ground plane)
    ///
    Eigen::Vector2d G;


    template <typename T>
    ///
    /// \brief GetGravityVector Returns the 3d gravity vector from the 2d direction vector
    /// \param direction The 2d gravity direction vector
    /// \param g Gravity of 1 g in m/s^2
    /// \return The 3d gravity vecto
    ///
    static Eigen::Matrix<T,3,1> GetGravityVector(const Eigen::Matrix<T,2,1>& direction, const T g = (T)9.80665)
    {
        T sp = sin(direction[0]);
        T cp = cos(direction[0]);
        T sq = sin(direction[1]);
        T cq = cos(direction[1]);
        Eigen::Matrix<T,3,1> vec(cp*sq,-sp,cp*cq);
        vec *= -g;
        return vec;
    }

    template <typename T>
    ///
    /// \brief dGravity_dDirection Returns the jacobian associated with getting the 3d gravity vector from the 2d direction
    /// \param direction The 2d gravity direction vector
    /// \param g Gravity of 1 g in m/s^2
    /// \return The 3x2 jacobian matrix
    ///
    static Eigen::Matrix<T,3,2> dGravity_dDirection(const Eigen::Matrix<T,2,1>& direction, const T g = (T)9.80665)
    {
        T sp = sin(direction[0]);
        T cp = cos(direction[0]);
        T sq = sin(direction[1]);
        T cq = cos(direction[1]);
        Eigen::Matrix<T,3,2> vec;
        vec << -sp*sq, cp*cq,
                  -cp,     0,
               -cq*sp,-cp*sq;
        vec *= -g;
        return vec;
    }
};

struct ImuPose
{
    ImuPose(const Sophus::SE3d& twp, const Eigen::Vector3d& v, const Eigen::Vector3d& w, const double time) :
        Twp(twp), V(v), W(w), Time(time) {}
    Sophus::SE3d Twp;
    Eigen::Vector3d V;
    Eigen::Vector3d W;
    double Time;
};

struct ImuMeasurement
{
    ImuMeasurement(const Eigen::Vector3d& w,const Eigen::Vector3d& a, const double time): W(w), A(a), Time(time) {}
    Eigen::Vector3d W;
    Eigen::Vector3d A;
    double Time;
    ImuMeasurement operator*(const double &rhs) {
        return ImuMeasurement( W*rhs, A*rhs, Time );
    }
    ImuMeasurement operator+(const ImuMeasurement &rhs) {
        return ImuMeasurement( W+rhs.W, A+rhs.A, Time );
    }
};

struct UnaryResidual
{
    static const unsigned int ResSize = 6;
    unsigned int PoseId;
    unsigned int ResidualId;
    unsigned int ResidualOffset;
    double       W;
    Sophus::SE3d Twp;
    Eigen::Matrix<double,ResSize,6> dZ_dX;
    Eigen::Vector6d Residual;
};

struct BinaryResidual
{
    static const unsigned int ResSize = 6;
    unsigned int PoseAId;
    unsigned int PoseBId;
    unsigned int ResidualId;
    unsigned int ResidualOffset;
    double       W;
    Sophus::SE3d Tab;
    Eigen::Matrix<double,ResSize,6> dZ_dX1;
    Eigen::Matrix<double,ResSize,6> dZ_dX2;
    Eigen::Vector6d Residual;
};

template<int LmSize>
struct ProjectionResidualT
{
    static const unsigned int ResSize = 2;
    Eigen::Vector2d Z;
    unsigned int PoseId;
    unsigned int LandmarkId;
    unsigned int CameraId;
    unsigned int ResidualId;
    unsigned int ResidualOffset;
    double       W;

    Eigen::Matrix<double,ResSize,LmSize> dZ_dX;
    Eigen::Matrix<double,2,6> dZ_dP;
    Eigen::Vector2d Residual;
};

struct ImuResidual
{
    static const unsigned int ResSize = 9;
    unsigned int PoseAId;
    unsigned int PoseBId;
    unsigned int ResidualId;
    unsigned int ResidualOffset;
    double       W;
    std::vector<ImuMeasurement> Measurements;
    std::vector<ImuPose> Poses;
    Eigen::Matrix<double,ResSize,9> dZ_dX1;
    Eigen::Matrix<double,ResSize,9> dZ_dX2;
    Eigen::Matrix<double,ResSize,2> dZ_dG;
    Eigen::Matrix<double,ResSize,6> dZ_dB;
    Eigen::Vector9d Residual;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    static ImuPose IntegratePose(const ImuPose& pose, const Eigen::Matrix<double,9,1>& derivative, const double dt)
    {
        ImuPose res = pose;
        res.Twp.translation() += derivative.head<3>()*dt;
        const Sophus::SO3d Rv2v1(Sophus::SO3d::exp(derivative.segment<3>(3)*dt));
        res.Twp.so3() = Rv2v1*pose.Twp.so3();

        // do euler integration for now
        res.V += derivative.tail<3>()*dt;
        return res;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    static Eigen::Matrix<double,9,1> GetPoseDerivative(const ImuPose& pose, const Eigen::Vector3d& tG_w, const ImuMeasurement& zStart,
                                                       const ImuMeasurement& zEnd, const Eigen::Vector3d& vBg,
                                                       const Eigen::Vector3d& vBa, const double dt)
    {
        double alpha = (zEnd.Time - (zStart.Time+dt))/(zEnd.Time - zStart.Time);
        Eigen::Vector3d zb = zStart.W*alpha + zEnd.W*(1.0-alpha);
        Eigen::Vector3d za = zStart.A*alpha + zEnd.A*(1.0-alpha);

        Eigen::Matrix<double,9,1> deriv;
        //derivative of position is velocity
        deriv.head<3>() = pose.V;
        //deriv.template segment<3>(3) = Sophus::SO3Group<T>::vee(tTwb.so3().matrix()*Sophus::SO3Group<T>::hat(zb));
        deriv.segment<3>(3) = pose.Twp.so3().Adj()*(zb+vBg);
        deriv.segment<3>(6) = pose.Twp.so3()*(za+vBa) - tG_w;
        return deriv;
    }


    ///////////////////////////////////////////////////////////////////////////////////////////////
    static ImuPose IntegrateImu(const ImuPose& y0, const ImuMeasurement& zStart,
                     const ImuMeasurement& zEnd, const Eigen::Vector3d& vBg,
                     const Eigen::Vector3d& vBa,const Eigen::Vector3d& dG)
    {
        //construct the state matrix
        double h = zEnd.Time - zStart.Time;
        if(h == 0){
            return y0;
        }

        Eigen::Matrix<double,9,1> k1 = GetPoseDerivative(y0,dG,zStart,zEnd,vBg,vBa,0);
        ImuPose y1 = IntegratePose(y0,k1,h*0.5);
        Eigen::Matrix<double,9,1> k2 = GetPoseDerivative(y1,dG,zStart,zEnd,vBg,vBa,h/2);
        ImuPose y2 = IntegratePose(y0,k2,h*0.5);
        Eigen::Matrix<double,9,1> k3 = GetPoseDerivative(y2,dG,zStart,zEnd,vBg,vBa,h/2);
        ImuPose y3 = IntegratePose(y0,k3,h);
        Eigen::Matrix<double,9,1> k4 = GetPoseDerivative(y3,dG,zStart,zEnd,vBg,vBa,h/2);

        Eigen::Matrix<double,9,1> k = (k1+2*k2+2*k3+k4);
        ImuPose res = IntegratePose(y0,k, (1.0/6.0) * h);

        res.W = k.segment<3>(3);
        res.Time = zEnd.Time;
//        pose.m_dW = currentPose.m_dW;
        return res;
    }

    template<int LmSize>
    ///////////////////////////////////////////////////////////////////////////////////////////////
    static ImuPose IntegrateResidual(const PoseT<LmSize>& pose,
                                     const std::vector<ImuMeasurement>& vMeasurements,
                                     const Eigen::Vector3d& bg,
                                     const Eigen::Vector3d& ba,
                                     const Eigen::Vector3d& g,
                                     std::vector<ImuPose>& vPoses)
    {
        ImuPose imuPose(pose.Twp,pose.V,Eigen::Vector3d::Zero(),pose.Time);
        const ImuMeasurement* pPrevMeas = 0;
        vPoses.clear();
        vPoses.reserve(vMeasurements.size()+1);
        vPoses.push_back(imuPose);

        // integrate forward in time, and retain all the poses
        for(const ImuMeasurement& meas : vMeasurements){
            if(pPrevMeas != 0){
//                std::cout << "Integrating from time " << pPrevMeas->Time << " to " << meas.Time << std::endl;
                imuPose = IntegrateImu(imuPose,*pPrevMeas,meas,bg,ba,g);
                vPoses.push_back(imuPose);
            }
            pPrevMeas = &meas;
        }
        return imuPose;
    }


};



}
