#pragma once
#include "Types.h"

namespace ba
{

/// Ceres autodifferentiatable cost function for pose errors.
/// The parameters are error state and should be a 6d pose delta
template< typename Scalar=double >
struct GlobalPoseCostFunction
{
    GlobalPoseCostFunction(const Sophus::SE3Group<Scalar>& dMeasurement, const double dWeight  )
    : m_Tw_c(dMeasurement),
      m_dWeight(dWeight)
    {}

    template <typename T>
    bool operator()(const T* const _tTic,const T* const _tTwi, T* residuals) const
    {
        //const Eigen::Map<const Sophus::SE3Group<T> > T_wx(_t); //the pose delta
        const Eigen::Map<const Sophus::SE3Group<T> > T_i_c(_tTic);
        const Eigen::Map<const Sophus::SE3Group<T> > T_w_i(_tTwi);
       Eigen::Map<Eigen::Matrix<T,6,1> > pose_residuals(residuals); //the pose residuals

       pose_residuals = (T_w_i* T_i_c * m_Tw_c.inverse().template cast<T>()).log()  * (T)m_dWeight;
       pose_residuals.tail(3) * m_dWeight;
       pose_residuals.head(3) * m_dWeight;
        //pose_residuals.head(3) *= (T)0.9;
        //pose_residuals.tail(4) *= (T)0.3;
        return true;
    }


    const Sophus::SE3Group<Scalar> m_Tw_c;
    const double m_dWeight;
};

template< typename Scalar=double >
struct FullImuCostFunction
{
    FullImuCostFunction(const std::vector<ImuMeasurementT<Scalar>>& vMeas)    :
        m_vMeas(vMeas)
  {
  }

    template <typename T>
    bool operator()(const T* const _tx2,const T* const _tx1,
                    const T* const _tVx2,const T* const _tVx1,
                    const T* const _tG,const T* const _tBg,
                    const T* const _tBa, T* residuals) const
    {
        Eigen::IOFormat CleanFmt(3, 0, ", ", "\n" , "[" , "]");

        //the residual vector consists of a 6d pose and a 3d velocity residual
        Eigen::Map<Eigen::Matrix<T,6,1> > pose_residuals(residuals); //the pose residuals
        Eigen::Map<Eigen::Matrix<T,3,1> > vel_residuals(&residuals[6]); //the velocity residuals
        //Eigen::Map<Eigen::Matrix<T,3,1> > vel_prior(&residuals[9]); //the velocity residuals

        //parameter vector consists of a 6d pose delta plus starting velocity and 2d gravity angles
        const Eigen::Map<const Sophus::SE3Group<T> > T_w_x2(_tx2);
        const Eigen::Map<const Sophus::SE3Group<T> > T_w_x1(_tx1);
        const Eigen::Map<const Sophus::SO3Group<T> > R_w_x1(&_tx1[0]);
        const Eigen::Map<const Eigen::Matrix<T,3,1> > v_v1(_tVx1); //the velocity at the starting point
        const Eigen::Map<const Eigen::Matrix<T,3,1> > v_v2(_tVx2); //the velocity at the end point
        const Eigen::Map<const Eigen::Matrix<T,3,1> > v_Bg(_tBg); //gyro bias
        const Eigen::Map<const Eigen::Matrix<T,3,1> > v_Ba(_tBa); //accelerometer bias
        const Eigen::Map<const Eigen::Matrix<T,2,1> > g(_tG); //the 2d gravity vector (angles)

        //get the gravity components in 3d based on the 2 angles of the gravity vector
        const Eigen::Matrix<T,3,1> g_vector = GetGravityVector<T>(g);

        ImuPoseT<T> startPose;
        startPose.Twp = T_w_x1;
        startPose.V = v_v1;
        startPose.W.setZero();
        startPose.Time = m_vMeas.front().Time;
        std::vector<ImuPoseT<T>> vPoses;
        ImuPoseT<T> endPose = ImuResidualT<Scalar>::template IntegrateResidual<T>(startPose,m_vMeas,v_Bg,v_Ba,g_vector,vPoses);

        //and now calculate the error with this pose
        pose_residuals = (endPose.Twp * T_w_x2.inverse()).log();

        //to calculate the velocity error, first augment the IMU integration velocity with gravity and initial velocity
        vel_residuals = (endPose.V - v_v2);
        return true;
    }

    const std::vector<ImuMeasurementT<Scalar>> m_vMeas;
};
}
