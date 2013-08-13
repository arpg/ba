#pragma once
#include "Types.h"

namespace ba
{

///////////////////////////////////////////////////////////////////////////////////////////////
template<typename T, typename Scalar>
static ImuPoseT<T> IntegratePoseJet(const ImuPoseT<T>& pose, const Eigen::Matrix<T,9,1>& k, const Scalar dt)
{
    const Sophus::SO3Group<T> Rv2v1(Sophus::SO3Group<T>::exp(k.template segment<3>(3)*(T)dt));

    ImuPoseT<T> y = pose;
    y.Twp.translation() += k.template head<3>()*(T)dt;
//        y.Twp.so3() = Rv2v1*pose.Twp.so3();
    memcpy(y.Twp.so3().data(),(Rv2v1.unit_quaternion()*pose.Twp.so3().unit_quaternion()).coeffs().data(),sizeof(T)*4);
    // do euler integration for now
    y.V += k.template tail<3>()*(T)dt;

    return y;
}

///////////////////////////////////////////////////////////////////////////////////////////////
template<typename T, typename Scalar>
static Eigen::Matrix<T,9,1> GetPoseDerivativeJet(const ImuPoseT<T>& pose, const Eigen::Matrix<T,3,1>& tG_w, const ImuMeasurementT<Scalar>& zStart,
                                                   const ImuMeasurementT<Scalar>& zEnd, const Eigen::Matrix<T,3,1>& vBg,
                                                   const Eigen::Matrix<T,3,1>& vBa, const Scalar dt)
{
    double alpha = (zEnd.Time - (zStart.Time+dt))/(zEnd.Time - zStart.Time);
    Eigen::Matrix<Scalar,3,1> zg = zStart.W*alpha + zEnd.W*(1.0-alpha);
    Eigen::Matrix<Scalar,3,1> za = zStart.A*alpha + zEnd.A*(1.0-alpha);

    Eigen::Matrix<T,9,1> deriv;
    //derivative of position is velocity
    deriv.template head<3>() = pose.V;                               // v (velocity)
    //deriv.template segment<3>(3) = Sophus::SO3Group<T>::vee(tTwb.so3().matrix()*Sophus::SO3Group<T>::hat(zb));
    deriv.template segment<3>(3) = pose.Twp.so3().Adj()*(zg.template cast<T>()+vBg);    // w (angular rates)
    deriv.template segment<3>(6) = pose.Twp.so3()*(za.template cast<T>()+vBa) - tG_w;   // a (acceleration)

    return deriv;
}

///////////////////////////////////////////////////////////////////////////////////////////////
template<typename T, typename Scalar>
static ImuPoseT<T> IntegrateImuJet(const ImuPoseT<T>& y0, const ImuMeasurementT<Scalar>& zStart,
                 const ImuMeasurementT<Scalar>& zEnd, const Eigen::Matrix<T,3,1>& vBg,
                 const Eigen::Matrix<T,3,1>& vBa,const Eigen::Matrix<T,3,1>& dG,
                 Eigen::Matrix<Scalar,10,6>* pDy_db = 0,
                 Eigen::Matrix<Scalar,10,10>* pDy_dy0 = 0)
{
    //construct the state matrix
    Scalar dt = zEnd.Time - zStart.Time;
    if(dt == 0){
        return y0;
    }
    ImuPoseT<T> res = y0;
    Eigen::Matrix<T,9,1> k;

    const Eigen::Matrix<T,9,1> k1 = GetPoseDerivativeJet<T,Scalar>(y0,dG,zStart,zEnd,vBg,vBa,0);
    const ImuPoseT<T> y1 = IntegratePoseJet<T,Scalar>(y0,k1,dt*0.5);
    const Eigen::Matrix<T,9,1> k2 = GetPoseDerivativeJet<T,Scalar>(y1,dG,zStart,zEnd,vBg,vBa,dt/2);
    const ImuPoseT<T> y2 = IntegratePoseJet<T,Scalar>(y0,k2,dt*0.5);
    const Eigen::Matrix<T,9,1> k3 = GetPoseDerivativeJet<T,Scalar>(y2,dG,zStart,zEnd,vBg,vBa,dt/2);
    const ImuPoseT<T> y3 = IntegratePoseJet<T,Scalar>(y0,k3,dt);
    const Eigen::Matrix<T,9,1> k4 = GetPoseDerivativeJet<T,Scalar>(y3,dG,zStart,zEnd,vBg,vBa,dt);
    k = (k1+(T)2*k2+(T)2*k3+k4);
    res = IntegratePoseJet<T,Scalar>(y0,k, dt/6.0);

    res.W = k.template segment<3>(3);
    res.Time = zEnd.Time;
//        pose.m_dW = currentPose.m_dW;
    return res;
}

///////////////////////////////////////////////////////////////////////////////////////////////
template< typename T, typename Scalar >
static ImuPoseT<T> IntegrateResidualJet(const PoseT<T>& pose,
                                 const std::vector<ImuMeasurementT<Scalar>>& vMeasurements,
                                 const Eigen::Matrix<T,3,1>& bg,
                                 const Eigen::Matrix<T,3,1>& ba,
                                 const Eigen::Matrix<T,3,1>& g,
                                 std::vector<ImuPoseT<T>>& vPoses)
{
    ImuPoseT<T> imuPose(pose.Twp,pose.V,Eigen::Matrix<T,3,1>::Zero(),pose.Time);
    const ImuMeasurementT<Scalar>* pPrevMeas = 0;
    vPoses.clear();
    vPoses.reserve(vMeasurements.size()+1);
    vPoses.push_back(imuPose);

    // integrate forward in time, and retain all the poses
    for(const ImuMeasurementT<Scalar>& meas : vMeasurements){
        if(pPrevMeas != 0){
//                std::cout << "Integrating from time " << pPrevMeas->Time << " to " << meas.Time << std::endl;
            imuPose = IntegrateImuJet<T,Scalar>(imuPose,*pPrevMeas,meas,bg,ba,g);
            vPoses.push_back(imuPose);
        }
        pPrevMeas = &meas;
    }
    return imuPose;
}

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
       pose_residuals.tail(3) * (T)m_dWeight;
       pose_residuals.head(3) * (T)m_dWeight;
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

        PoseT<T> startPose;
        startPose.Twp = T_w_x1;
        startPose.V = v_v1;
        startPose.Time = m_vMeas.front().Time;
        std::vector<ImuPoseT<T>> vPoses;
        ImuPoseT<T> endPose = IntegrateResidualJet<T,Scalar>(startPose,m_vMeas,v_Bg,v_Ba,g_vector,vPoses);

        //and now calculate the error with this pose
        pose_residuals = (endPose.Twp * T_w_x2.inverse()).log();

        //to calculate the velocity error, first augment the IMU integration velocity with gravity and initial velocity
        vel_residuals = (endPose.V - v_v2);
        return true;
    }

    const std::vector<ImuMeasurementT<Scalar>> m_vMeas;
};
}
