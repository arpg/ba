#pragma once

#include <Eigen/Eigen>
#include <sophus/se3.hpp>
#include "Utils.h"
#include <calibu/Calibu.h>

#define IMU_GYRO_UNCERTAINTY 100 //0.00104719755
#define IMU_ACCEL_UNCERTAINTY 100 //0.0392266

namespace ba
{

template<typename Scalar=double>
struct PoseT
{
    Sophus::SE3Group<Scalar> Twp;
    Sophus::SE3Group<Scalar> Tvs;
    Eigen::Matrix<Scalar,3,1> V;
    Eigen::Matrix<Scalar,6,1> B;
    bool IsActive;
    unsigned int Id;
    unsigned int OptId;
    double Time;
    std::vector<int> ProjResiduals;
    std::vector<int> ImuResiduals;
    std::vector<int> BinaryResiduals;
    std::vector<int> UnaryResiduals;
    std::vector<Sophus::SE3Group<Scalar>> Tsw;

    const Sophus::SE3Group<Scalar>& GetTsw(const unsigned int camId, const calibu::CameraRigT<Scalar>& rig, const bool bUseTswParam)
    {
        while(Tsw.size() <= camId ){
            if(bUseTswParam == false){
                Tsw.push_back( (Twp*rig.cameras[Tsw.size()].T_wc).inverse());
            }else{
                // this needs to be modified to work with stereo
                Tsw.push_back( (Twp*Tvs).inverse());
            }
        }
        return Tsw[camId];
    }
};

template<typename Scalar=double,int LmSize=1>
struct LandmarkT
{
    Eigen::Matrix<Scalar,4,1> Xs;
    Eigen::Matrix<Scalar,4,1> Xw;
    std::vector<int> ProjResiduals;
    unsigned int OptId;
    unsigned int RefPoseId;
    unsigned int RefCamId;
};

template<typename Scalar=double>
struct ImuCalibrationT
{
    ImuCalibrationT(const Sophus::SE3Group<Scalar>& tvs, const Eigen::Matrix<Scalar,3,1>& bg, const Eigen::Matrix<Scalar,3,1>& ba, const Eigen::Matrix<Scalar,2,1>& g):
        Tvs(tvs),Bg(bg),Ba(ba),G(g),
        R((Eigen::Vector6d() << IMU_GYRO_UNCERTAINTY, IMU_GYRO_UNCERTAINTY, IMU_GYRO_UNCERTAINTY,
           IMU_ACCEL_UNCERTAINTY, IMU_ACCEL_UNCERTAINTY, IMU_ACCEL_UNCERTAINTY).finished().asDiagonal()){}
    ///
    /// \brief Calibration from vehicle to sensor frame (monocular for now)
    ///
    Sophus::SE3Group<Scalar> Tvs;
    ///
    /// \brief Gyroscope bias vector
    ///
    Eigen::Matrix<Scalar,3,1> Bg;
    ///
    /// \brief Accelerometer bias vector
    ///
    Eigen::Matrix<Scalar,3,1> Ba;
    ///
    /// \brief Gravity vector (2D, parametrized by roll and pitch of the vector wrt the ground plane)
    ///
    Eigen::Matrix<Scalar,2,1> G;
    ///
    /// \brief Sensor uncertainty. The first 3 rows/cols are gyroscope and the last are accel
    ///
    Eigen::Matrix<Scalar,6,6> R;
};

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

template< typename Scalar=double >
struct ImuPoseT
{
    ImuPoseT(const PoseT<Scalar>& pose) :
        Twp(pose.Twp), V(pose.V), W(Eigen::Matrix<Scalar,3,1>::Zero()), Time(pose.Time) {}
    ImuPoseT(const Sophus::SE3Group<Scalar>& twp,
             const Eigen::Matrix<Scalar,3,1>& v,
             const Eigen::Matrix<Scalar,3,1>& w,
             const double time) :
        Twp(twp), V(v), W(w), Time(time) {}
    Sophus::SE3Group<Scalar> Twp;
    Eigen::Matrix<Scalar,3,1> V;
    Eigen::Matrix<Scalar,3,1> W;    
    double Time;
};

template< typename Scalar=double >
struct ImuMeasurementT
{
    ImuMeasurementT() {}
    ImuMeasurementT(const Eigen::Matrix<Scalar,3,1>& w,const Eigen::Matrix<Scalar,3,1>& a, const double time): W(w), A(a), Time(time) {}
    Eigen::Matrix<Scalar,3,1> W;
    Eigen::Matrix<Scalar,3,1> A;
    double Time;
    ImuMeasurementT operator*(const Scalar &rhs) {
        return ImuMeasurementT( W*rhs, A*rhs, Time );
    }
    ImuMeasurementT operator+(const ImuMeasurementT &rhs) {
        return ImuMeasurementT( W+rhs.W, A+rhs.A, Time );
    }
};

template< typename Scalar=double >
struct UnaryResidualT
{
    static const unsigned int ResSize = 6;
    unsigned int PoseId;
    unsigned int ResidualId;
    unsigned int ResidualOffset;
    Scalar       W;
    Sophus::SE3Group<Scalar> Twp;
    Eigen::Matrix<Scalar,ResSize,6> dZ_dX;
    Eigen::Matrix<Scalar,6,1> Residual;
};

template< typename Scalar=double >
struct BinaryResidualT
{
    static const unsigned int ResSize = 6;
    unsigned int PoseAId;
    unsigned int PoseBId;
    unsigned int ResidualId;
    unsigned int ResidualOffset;
    Scalar       W;
    Sophus::SE3Group<Scalar> Tab;
    Eigen::Matrix<Scalar,ResSize,6> dZ_dX1;
    Eigen::Matrix<Scalar,ResSize,6> dZ_dX2;
    Eigen::Matrix<Scalar,6,1> Residual;
};

template<typename Scalar=double, int LmSize = 1>
struct ProjectionResidualT
{
    static const unsigned int ResSize = 2;
    Eigen::Matrix<Scalar,2,1> Z;
    unsigned int MeasPoseId;
    unsigned int RefPoseId;
    unsigned int LandmarkId;
    unsigned int CameraId;
    unsigned int ResidualId;
    unsigned int ResidualOffset;
    Scalar       W;

    Eigen::Matrix<Scalar,ResSize,LmSize> dZ_dX;
    Eigen::Matrix<Scalar,2,6> dZ_dPm;
    Eigen::Matrix<Scalar,2,6> dZ_dPr;
    Eigen::Matrix<Scalar,2,6> dZ_dTvs_m;
    Eigen::Matrix<Scalar,2,6> dZ_dTvs_r;
    // Eigen::Matrix<Scalar,2,6> dZ_dTic;    // derivative with respect to imu/camera parameters
    // Eigen::Matrix<Scalar,2,5> dZ_dK;    // derivative with respect to camera parameters
    Eigen::Matrix<Scalar,2,1> Residual;
};

template< typename Scalar=double, int ResidualSize = 15, int PoseSize = 15 >
struct ImuResidualT
{
    typedef ImuPoseT<Scalar> ImuPose;
    typedef ImuMeasurementT<Scalar> ImuMeasurement;
    static const unsigned int ResSize = ResidualSize;
    unsigned int PoseAId;
    unsigned int PoseBId;
    unsigned int ResidualId;
    unsigned int ResidualOffset;
    Scalar       W;
    // Eigen::Matrix<Scalar,9,9>   SigmanInv;
    std::vector<ImuMeasurement> Measurements;
    std::vector<ImuPose> Poses;
    Eigen::Matrix<Scalar,ResSize,PoseSize> dZ_dX1;
    Eigen::Matrix<Scalar,ResSize,PoseSize> dZ_dX2;
    Eigen::Matrix<Scalar,9,2> dZ_dG;
    Eigen::Matrix<Scalar,ResSize,6> dZ_dB;
    Eigen::Matrix<Scalar,ResSize,1> Residual;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    static ImuPose IntegratePose(const ImuPose& pose, const Eigen::Matrix<Scalar,9,1>& k, const Scalar dt,
                                 Eigen::Matrix<Scalar,10,9>* pdy_dk = 0,Eigen::Matrix<Scalar,10,10>* pdy_dy = 0)
    {
        const Sophus::SO3Group<Scalar> Rv2v1(Sophus::SO3Group<Scalar>::exp(k.template segment<3>(3)*dt));

        ImuPose y = pose;
        y.Twp.translation() += k.template head<3>()*dt;
//        y.Twp.so3() = Rv2v1*pose.Twp.so3();
        memcpy(y.Twp.so3().data(),(Rv2v1.unit_quaternion()*pose.Twp.so3().unit_quaternion()).coeffs().data(),sizeof(Scalar)*4);
        // do euler integration for now
        y.V += k.template tail<3>()*dt;

        // jacobian of output pose relative to the derivative
        if( pdy_dk != 0 ){
            pdy_dk->setZero();
            pdy_dk->template block<3,3>(0,0) = Eigen::Matrix<Scalar,3,3>::Identity()*dt;  // dt/dv
            pdy_dk->template block<4,3>(3,3) = dq1q2_dq1<Scalar>(pose.Twp.so3().unit_quaternion()) * dqExp_dw<Scalar>(k.template segment<3>(3)*dt) *dt;  // dq/dw
            pdy_dk->template block<3,3>(7,6) = Eigen::Matrix<Scalar,3,3>::Identity()*dt;  // dv/da
        }

        if( pdy_dy != 0 )
        {
            pdy_dy->setZero();
            pdy_dy->template block<3,3>(0,0) = Eigen::Matrix<Scalar,3,3>::Identity();
            pdy_dy->template block<4,4>(3,3) = dq1q2_dq2(Rv2v1.unit_quaternion());
            pdy_dy->template block<3,3>(7,7) = Eigen::Matrix<Scalar,3,3>::Identity();


            assert( _Test_IntegratePose_ExpJacobian(k,dt) );
            assert( _Test_IntegratePose_StateKJacobian(pose, k, dt, *pdy_dk));
        }

        return y;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    static Eigen::Matrix<Scalar,9,1> GetPoseDerivative(const ImuPose& pose, const Eigen::Matrix<Scalar,3,1>& tG_w, const ImuMeasurement& zStart,
                                                       const ImuMeasurement& zEnd, const Eigen::Matrix<Scalar,3,1>& vBg,
                                                       const Eigen::Matrix<Scalar,3,1>& vBa, const Scalar dt,
                                                       Eigen::Matrix<Scalar,9,6>* dk_db = 0,Eigen::Matrix<Scalar,9,10>* dk_dx = 0)
    {
        double alpha = (zEnd.Time - (zStart.Time+dt))/(zEnd.Time - zStart.Time);
        Eigen::Matrix<Scalar,3,1> zg = zStart.W*alpha + zEnd.W*(1.0-alpha);
        Eigen::Matrix<Scalar,3,1> za = zStart.A*alpha + zEnd.A*(1.0-alpha);

        Eigen::Matrix<Scalar,9,1> deriv;
        //derivative of position is velocity
        deriv.template head<3>() = pose.V;                               // v (velocity)
        //deriv.template segment<3>(3) = Sophus::SO3Group<T>::vee(tTwb.so3().matrix()*Sophus::SO3Group<T>::hat(zb));
        deriv.template segment<3>(3) = pose.Twp.so3().Adj()*(zg+vBg);    // w (angular rates)
        deriv.template segment<3>(6) = pose.Twp.so3()*(za+vBa) - tG_w;   // a (acceleration)

        if(dk_db != 0) {
            dk_db->setZero();
            dk_db->template block<3,3>(3,0) = pose.Twp.so3().Adj(); // dw/dbg
            dk_db->template block<3,3>(6,3) = pose.Twp.so3().matrix();       // da/dba
        }
        if(dk_dx != 0){
            dk_dx->setZero();
            dk_dx->template block<3,3>(0,7) = Eigen::Matrix<Scalar,3,3>::Identity(); // dv/dv
            dk_dx->template block<3,4>(3,3) = dqx_dq(pose.Twp.so3().unit_quaternion(),zg) + dqx_dq(pose.Twp.so3().unit_quaternion(),vBg); // dw/dq
            dk_dx->template block<3,4>(6,3) = dqx_dq(pose.Twp.so3().unit_quaternion(),za) + dqx_dq(pose.Twp.so3().unit_quaternion(),vBa); // da/dq
        }
        return deriv;
    }


    ///////////////////////////////////////////////////////////////////////////////////////////////
    static ImuPose IntegrateImu(const ImuPose& y0, const ImuMeasurement& zStart,
                     const ImuMeasurement& zEnd, const Eigen::Matrix<Scalar,3,1>& vBg,
                     const Eigen::Matrix<Scalar,3,1>& vBa,const Eigen::Matrix<Scalar,3,1>& dG,
                     Eigen::Matrix<Scalar,10,6>* pDy_db = 0,
                     Eigen::Matrix<Scalar,10,10>* pDy_dy0 = 0)
    {
        //construct the state matrix
        Scalar dt = zEnd.Time - zStart.Time;
        if(dt == 0){
            return y0;
        }


        ImuPose res = y0;
        Eigen::Matrix<Scalar,9,1> k;

        if(pDy_db != 0) {
            Eigen::Matrix<Scalar,10,6>& dy_db = *pDy_db;
            Eigen::Matrix<Scalar,10,10>& dy_dy0 = *pDy_dy0;

            Eigen::Matrix<Scalar,9,6>  dk_db;
            Eigen::Matrix<Scalar,9,10> dk_dy;
            Eigen::Matrix<Scalar,10,9> dy_dk;
            Eigen::Matrix<Scalar,10,10> dy_dy;
            dy_db.setZero();
            dy_dy0.setIdentity();   // dy0_y0 starts at identity

            const Eigen::Matrix<Scalar,9,1> k1 = GetPoseDerivative(y0,dG,zStart,zEnd,vBg,vBa,0,&dk_db,&dk_dy);
            assert( _Test_IntegrateImu_KBiasJacobian( y0, zStart, zEnd, vBg, vBa, dG, dk_db ) );
            assert( _Test_IntegrateImu_KStateJacobian( y0, zStart, zEnd, vBg, vBa, dG, dk_dy ) );

            // total derivative of k1 wrt b: dk1/db = dG/db + dG/dy*dy/db
            const Eigen::Matrix<Scalar,9,6> dk1_db = dk_db + dk_dy*dy_db;
            const Eigen::Matrix<Scalar,9,10> dk1_dy= dk_dy*dy_dy0;
            const ImuPose y1 = IntegratePose(y0,k1,dt*0.5,&dy_dk,&dy_dy);
            //dy1/db = dInt/db + dInt/dy*dy0/db + dInt/dk*dk/db
            // however dy0/db = 0 (only for y0), therefore we don't need the second term, just dInt/dk and dInt/db.
            // but dInt/db is also 0, as the integration doesn't directly depend on b
            dy_db = dy_dk*dk1_db;
            dy_dy0 = dy_dy + dy_dk*dk1_dy; // this is dy1_dy0

            assert( _Test_IntegrateImu_StateStateJacobian( y0, k1, dy_dy, dt ) );

            const Eigen::Matrix<Scalar,9,1> k2 = GetPoseDerivative(y1,dG,zStart,zEnd,vBg,vBa,dt/2,&dk_db,&dk_dy);
            const Eigen::Matrix<Scalar,9,6> dk2_db = dk_db + dk_dy*dy_db;
            const Eigen::Matrix<Scalar,9,10> dk2_dy= dk_dy*dy_dy0;
            const ImuPose y2 = IntegratePose(y0,k2,dt*0.5,&dy_dk,&dy_dy);
            dy_db = dy_dk*dk2_db;
            dy_dy0 = dy_dy + dy_dk*dk2_dy; // this is dy2_dy0

            const Eigen::Matrix<Scalar,9,1> k3 = GetPoseDerivative(y2,dG,zStart,zEnd,vBg,vBa,dt/2,&dk_db,&dk_dy);
            const Eigen::Matrix<Scalar,9,6> dk3_db = dk_db + dk_dy*dy_db;
            const Eigen::Matrix<Scalar,9,10> dk3_dy = dk_dy*dy_dy0;
            const ImuPose y3 = IntegratePose(y0,k3,dt,&dy_dk,&dy_dy);
            dy_db = dy_dk*dk3_db;
            dy_dy0 = dy_dy + dy_dk*dk3_dy; // this is dy3_dy0

            const Eigen::Matrix<Scalar,9,1> k4 = GetPoseDerivative(y3,dG,zStart,zEnd,vBg,vBa,dt,&dk_db,&dk_dy);
            const Eigen::Matrix<Scalar,9,6> dk4_db = dk_db + dk_dy*dy_db;
            const Eigen::Matrix<Scalar,9,10> dk4_dy = dk_dy*dy_dy0;

            k = (k1+2*k2+2*k3+k4);
            const Eigen::Matrix<Scalar,9,6> dk_total_db = dk1_db + 2*dk2_db + 2*dk3_db + dk4_db;
            const Eigen::Matrix<Scalar,9,10> dk_total_dy = dk1_dy + 2*dk2_dy + 2*dk3_dy + dk4_dy;
            res = IntegratePose(y0,k, dt/6.0,&dy_dk,&dy_dy);
            dy_db = dy_dk*dk_total_db;
            dy_dy0 = dy_dy + dy_dk*dk_total_dy;

        }else{
            const Eigen::Matrix<Scalar,9,1> k1 = GetPoseDerivative(y0,dG,zStart,zEnd,vBg,vBa,0);
            const ImuPose y1 = IntegratePose(y0,k1,dt*0.5);
            const Eigen::Matrix<Scalar,9,1> k2 = GetPoseDerivative(y1,dG,zStart,zEnd,vBg,vBa,dt/2);
            const ImuPose y2 = IntegratePose(y0,k2,dt*0.5);
            const Eigen::Matrix<Scalar,9,1> k3 = GetPoseDerivative(y2,dG,zStart,zEnd,vBg,vBa,dt/2);
            const ImuPose y3 = IntegratePose(y0,k3,dt);
            const Eigen::Matrix<Scalar,9,1> k4 = GetPoseDerivative(y3,dG,zStart,zEnd,vBg,vBa,dt);
            k = (k1+2*k2+2*k3+k4);
            res = IntegratePose(y0,k, dt/6.0);
        }

        res.W = k.template segment<3>(3);
        res.Time = zEnd.Time;
//        pose.m_dW = currentPose.m_dW;
        return res;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    static ImuPose IntegrateResidual(const PoseT<Scalar>& imuPose,
                                     const std::vector<ImuMeasurement>& vMeasurements,
                                     const Eigen::Matrix<Scalar,3,1>& bg,
                                     const Eigen::Matrix<Scalar,3,1>& ba,
                                     const Eigen::Matrix<Scalar,3,1>& g,
                                     std::vector<ImuPose>& vPoses,
                                     Eigen::Matrix<Scalar,10,6>* pJb = 0,
                                     Eigen::Matrix<Scalar,10,10>* pJy = 0)
    {
        return IntegrateResidual(ImuPose(imuPose),vMeasurements,bg,ba,g,vPoses,pJb,pJy);
    }


    ///////////////////////////////////////////////////////////////////////////////////////////////
    static ImuPose IntegrateResidual(ImuPose imuPose,
                                     const std::vector<ImuMeasurement>& vMeasurements,
                                     const Eigen::Matrix<Scalar,3,1>& bg,
                                     const Eigen::Matrix<Scalar,3,1>& ba,
                                     const Eigen::Matrix<Scalar,3,1>& g,
                                     std::vector<ImuPose>& vPoses,
                                     Eigen::Matrix<Scalar,10,6>* pJb = 0,
                                     Eigen::Matrix<Scalar,10,10>* pJy = 0)
    {
        Eigen::IOFormat cleanFmt(4, 0, ", ", ";\n" , "" , "");
        // ImuPose imuPose(pose.Twp,pose.V,Eigen::Matrix<Scalar,3,1>::Zero(),pose.Time);
        const ImuPose origPose = imuPose;
        const ImuMeasurement* pPrevMeas = 0;
        vPoses.clear();
        vPoses.reserve(vMeasurements.size()+1);
        vPoses.push_back(imuPose);

        if(pJb != 0){
            pJb->setZero();
        }

        if(pJy != 0){
            pJy->setIdentity();
        }

        // integrate forward in time, and retain all the poses
        for(const ImuMeasurement& meas : vMeasurements){
            if(pPrevMeas != 0){
//                std::cout << "Integrating from time " << pPrevMeas->Time << " to " << meas.Time << std::endl;
                if( pJb != 0 ){
                    //double dt = meas.Time - pPrevMeas->Time;
                    Eigen::Matrix<Scalar,10,6> dy_db;
                    Eigen::Matrix<Scalar,10,10> dy_dy;
                    const ImuPose y0 = imuPose;
                    imuPose = IntegrateImu(imuPose,*pPrevMeas,meas,bg,ba,g,&dy_db,&dy_dy);

                    assert( _Test_IntegrateImu_BiasJacobian( y0,*pPrevMeas,meas,bg,ba,g,dy_db ) );
                    assert( _Test_IntegrateImu_StateJacobian( y0,*pPrevMeas,meas,bg,ba,g,dy_dy ) );

                    // now push forward the jacobian. This calculates the total derivative Jb = dG/dB + dG/dX * dX/dB
                    // where dG/dB is the jacobian of the return values of IntegrateImu with respect to the bias
                    // values (which is returned in dq_dBg and dv_dBa, as the other values are 0). dG/dX is the
                    // jacobian of the IntegrateImu function with respect to its inputs (a 10x10 matrix, but only
                    // dq2_dq1 is complex, and is returned by IntegrateImu). dX/dB yis the jacobian from the previous
                    // step, which is stored in Jb. The following is the addition and multiplication unrolled into
                    // sparse operations.
                    if(pJb != 0){
                        (*pJb) = dy_db + dy_dy*(*pJb);
                    }

                    if(pJy != 0){
                        *pJy = dy_dy*(*pJy);
                    }
                }else{
                    imuPose = IntegrateImu(imuPose,*pPrevMeas,meas,bg,ba,g);
                }
                vPoses.push_back(imuPose);
            }
            pPrevMeas = &meas;
        }

        if( pJb != 0 ){
            assert(_Test_IntegrateResidual_BiasJacobian(origPose, vMeasurements, bg, ba, g, *pJb ) );
        }

        if(pJy != 0){
            assert(_Test_IntegrateResidual_StateJacobian( origPose, vMeasurements, bg, ba, g, *pJy ) );
        }


        return imuPose;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    static bool _Test_IntegrateImu_BiasJacobian(const ImuPose& y0,
                                                const ImuMeasurement& pPrevMeas,
                                                const ImuMeasurement& meas,
                                                const Eigen::Matrix<Scalar,3,1>& bg,
                                                const Eigen::Matrix<Scalar,3,1>& ba,
                                                const Eigen::Matrix<Scalar,3,1>& g,
                                                const Eigen::Matrix<Scalar,10,6>& dy_db)
    {
        Eigen::Matrix<Scalar, 10,6> J_fd;
        const Scalar dEps = TESTING_EPS;
        for(int ii = 0; ii < 6 ; ii++){
            Eigen::Matrix<Scalar,6,1> biasVec;
            biasVec.template head<3>() = bg;
            biasVec.template tail<3>() = ba;

            Eigen::Matrix<Scalar,6,1> eps = Eigen::Matrix<Scalar,6,1>::Zero();
            eps[ii] += dEps;
            biasVec += eps;
            ImuPose posePlus = IntegrateImu(y0,pPrevMeas,meas,biasVec.template head<3>(),biasVec.template tail<3>(),g);

            Eigen::Matrix<Scalar,10,1> yPlus;
            yPlus.template head<3>() = posePlus.Twp.translation();
            yPlus.template segment<4>(3) = posePlus.Twp.so3().unit_quaternion().coeffs();
            // do euler integration for now
            yPlus.template tail<3>() = posePlus.V;

            biasVec.template head<3>() = bg;
            biasVec.template tail<3>() = ba;

            eps[ii] -= 2*dEps;
            biasVec += eps;
            ImuPose poseMinus = IntegrateImu(y0,pPrevMeas,meas,biasVec.template head<3>(),biasVec.template tail<3>(),g);

            Eigen::Matrix<Scalar,10,1> yMinus;
            yMinus.template head<3>() = poseMinus.Twp.translation();
            yMinus.template segment<4>(3) = poseMinus.Twp.so3().unit_quaternion().coeffs();
            // do euler integration for now
            yMinus.template tail<3>() = poseMinus.V;

            J_fd.col(ii) = (yPlus-yMinus)/(2*dEps);
        }
        std::cout << "dInt_db = " << std::endl << dy_db.format(cleanFmt) << std::endl;
        std::cout << "dInt_db_fd = " << std::endl << J_fd.format(cleanFmt) << std::endl;
        std::cout << "dInt_db-dInt_db_fd = " << std::endl << (dy_db-J_fd).format(cleanFmt) << " norm: " << (dy_db-J_fd).norm() <<  std::endl;
        return (dy_db-J_fd).norm() < NORM_THRESHOLD;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    static bool _Test_IntegrateImu_StateJacobian(const ImuPose& y0,
                                                const ImuMeasurement& prevMeas,
                                                const ImuMeasurement& meas,
                                                const Eigen::Matrix<Scalar,3,1>& bg,
                                                const Eigen::Matrix<Scalar,3,1>& ba,
                                                const Eigen::Matrix<Scalar,3,1>& g,
                                                const Eigen::Matrix<Scalar,10,10>& dy_dy)
    {
        const Scalar dEps = TESTING_EPS;
        Eigen::Matrix<Scalar,10,10>  dy_dy_fd;
        for(int ii = 0; ii < 10 ; ii++){
            Eigen::Matrix<Scalar,10,1> epsVec = Eigen::Matrix<Scalar,10,1>::Zero();
            epsVec[ii] += dEps;
            ImuPose y0_eps = y0;
            y0_eps.Twp.translation() += epsVec.template head<3>();
            Eigen::Quaternion<Scalar> qPlus = y0_eps.Twp.so3().unit_quaternion();
            qPlus.coeffs() += epsVec.template segment<4>(3);
            memcpy(y0_eps.Twp.so3().data(),qPlus.coeffs().data(),sizeof(Scalar)*4);
            y0_eps.V += epsVec.template tail<3>();

            ImuPose posePlus = IntegrateImu(y0_eps,prevMeas,meas,bg,ba,g);

            Eigen::Matrix<Scalar,10,1> yPlus;
            yPlus.template head<3>() = posePlus.Twp.translation();
            yPlus.template segment<4>(3) = posePlus.Twp.so3().unit_quaternion().coeffs();
            // do euler integration for now
            yPlus.template tail<3>() = posePlus.V;

            epsVec[ii] -= 2*dEps;
            y0_eps = y0;
            y0_eps.Twp.translation() += epsVec.template head<3>();
            Eigen::Quaternion<Scalar> qMinus = y0_eps.Twp.so3().unit_quaternion();
            qMinus.coeffs() += epsVec.template segment<4>(3);
            y0_eps.Twp.so3() = Sophus::SO3Group<Scalar>(qMinus);
            memcpy(y0_eps.Twp.so3().data(),qMinus.coeffs().data(),sizeof(Scalar)*4);
            y0_eps.V += epsVec.template tail<3>();

            ImuPose poseMinus = IntegrateImu(y0_eps,prevMeas,meas,bg,ba,g);

            Eigen::Matrix<Scalar,10,1> yMinus;
            yMinus.template head<3>()= poseMinus.Twp.translation();
            yMinus.template segment<4>(3) = poseMinus.Twp.so3().unit_quaternion().coeffs();
            // do euler integration for now
            yMinus.template tail<3>() = poseMinus.V;

            dy_dy_fd.col(ii) = (yPlus-yMinus)/(2*dEps);
        }
        std::cout << "dInt_dy= " << std::endl << dy_dy.format(cleanFmt) << std::endl;
        std::cout << "dInt_dy_fd = " << std::endl << dy_dy_fd.format(cleanFmt) << std::endl;
        std::cout << "dInt_dy-dInt_dy_fd = " << std::endl << (dy_dy-dy_dy_fd).format(cleanFmt) << " norm: " << (dy_dy-dy_dy_fd).norm() << std::endl;

        return (dy_dy-dy_dy_fd).norm() < NORM_THRESHOLD;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    static bool _Test_IntegrateResidual_BiasJacobian(const ImuPose& y0,
                                                     const std::vector<ImuMeasurement>& vMeasurements,
                                                     const Eigen::Matrix<Scalar,3,1>& bg,
                                                     const Eigen::Matrix<Scalar,3,1>& ba,
                                                     const Eigen::Matrix<Scalar,3,1>& g,
                                                     const Eigen::Matrix<Scalar,10,6>& dy_db)
    {
        const ImuMeasurement* pPrevMeas = 0;
        Eigen::Matrix<Scalar,10,6> Jb_fd;
        const Scalar dEps = TESTING_EPS;
        Eigen::Matrix<Scalar,6,1> biasVec;
        biasVec.template head<3>() = bg;
        biasVec.template tail<3>() = ba;
        for(int ii = 0 ; ii < 6 ; ii++){
            Eigen::Matrix<Scalar,6,1> eps = Eigen::Matrix<Scalar,6,1>::Zero();
            eps[ii] += dEps;
            std::vector<ImuPose> poses;
            const Eigen::Matrix<Scalar,6,1> plusBiases = biasVec + eps;
            ImuPose imuPosePlus = y0;
            pPrevMeas = 0;
            for(const ImuMeasurement& meas : vMeasurements){
                if(pPrevMeas != 0){
                    imuPosePlus = IntegrateImu(imuPosePlus,*pPrevMeas,meas,plusBiases.template head<3>(),plusBiases.template tail<3>(),g);
                }
                pPrevMeas = &meas;
            }
            Eigen::Matrix<Scalar,10,1> plusVec;
            plusVec.template head<3>() = imuPosePlus.Twp.translation();
            plusVec.template segment<4>(3) = imuPosePlus.Twp.so3().unit_quaternion().coeffs();
            plusVec.template tail<3>() = imuPosePlus.V;

            eps[ii] -= 2*dEps;
            const Eigen::Matrix<Scalar,6,1> minusBiases = biasVec + eps;
            poses.clear();
            ImuPose imuPoseMinus = y0;
            pPrevMeas = 0;
            for(const ImuMeasurement& meas : vMeasurements){
                if(pPrevMeas != 0){
                    imuPoseMinus = IntegrateImu(imuPoseMinus,*pPrevMeas,meas,minusBiases.template head<3>(),minusBiases.template tail<3>(),g);
                }
                pPrevMeas = &meas;
            }
            Eigen::Matrix<Scalar,10,1> minusVec;
            minusVec.template head<3>() = imuPoseMinus.Twp.translation();
            minusVec.template segment<4>(3) = imuPoseMinus.Twp.so3().unit_quaternion().coeffs();
            minusVec.template tail<3>() = imuPoseMinus.V;

            Jb_fd.col(ii) = (plusVec-minusVec)/(2*dEps);
        }
        std::cout << "Jres = " << std::endl << (dy_db).format(cleanFmt) << std::endl;
        std::cout << "Jres_fd = " << std::endl << Jb_fd.format(cleanFmt) << std::endl;
        std::cout << "Jres-Jres_fd = " << std::endl << (dy_db-Jb_fd).format(cleanFmt) << " norm: " << (dy_db-Jb_fd).norm() <<  std::endl;
        return (dy_db-Jb_fd).norm() < NORM_THRESHOLD;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    static bool _Test_IntegrateResidual_StateJacobian(const ImuPose& y0,
                                                      const std::vector<ImuMeasurement>& vMeasurements,
                                                      const Eigen::Matrix<Scalar,3,1>& bg,
                                                      const Eigen::Matrix<Scalar,3,1>& ba,
                                                      const Eigen::Matrix<Scalar,3,1>& g,
                                                      const Eigen::Matrix<Scalar,10,10>& dy_dy)
    {
        const ImuMeasurement* pPrevMeas = 0;
        Eigen::Matrix<Scalar,10,10> Jy_fd;
        const Scalar dEps = TESTING_EPS;
        for(int ii = 0; ii < 10 ; ii++){
            Eigen::Matrix<Scalar,10,1> epsVec = Eigen::Matrix<Scalar,10,1>::Zero();
            epsVec[ii] += dEps;
            ImuPose posePlus = y0;
            posePlus.Twp.translation() += epsVec.template head<3>();
            Eigen::Quaternion<Scalar> qPlus = posePlus.Twp.so3().unit_quaternion();
            qPlus.coeffs() += epsVec.template segment<4>(3);
            memcpy(posePlus.Twp.so3().data(),qPlus.coeffs().data(),sizeof(Scalar)*4);
            posePlus.V += epsVec.template tail<3>();

            pPrevMeas = 0;
            for(const ImuMeasurement& meas : vMeasurements){
                if(pPrevMeas != 0){
                    posePlus = IntegrateImu(posePlus,*pPrevMeas,meas,bg,ba,g);
                }
                pPrevMeas = &meas;
            }

            Eigen::Matrix<Scalar,10,1> yPlus;
            yPlus.template head<3>() = posePlus.Twp.translation();
            yPlus.template segment<4>(3) = posePlus.Twp.so3().unit_quaternion().coeffs();
            // do euler integration for now
            yPlus.template tail<3>() = posePlus.V;

            epsVec[ii] -= 2*dEps;
            ImuPose poseMinus = y0;
            poseMinus.Twp.translation() += epsVec.template head<3>();
            Eigen::Quaternion<Scalar> qMinus = poseMinus.Twp.so3().unit_quaternion();
            qMinus.coeffs() += epsVec.template segment<4>(3);
            poseMinus.Twp.so3() = Sophus::SO3Group<Scalar>(qMinus);
            memcpy(poseMinus.Twp.so3().data(),qMinus.coeffs().data(),sizeof(Scalar)*4);
            poseMinus.V += epsVec.template tail<3>();

            pPrevMeas = 0;
            for(const ImuMeasurement& meas : vMeasurements){
                if(pPrevMeas != 0){
                    poseMinus = IntegrateImu(poseMinus,*pPrevMeas,meas,bg,ba,g);
                }
                pPrevMeas = &meas;
            }

            Eigen::Matrix<Scalar,10,1> yMinus;
            yMinus.template head<3>()= poseMinus.Twp.translation();
            yMinus.template segment<4>(3) = poseMinus.Twp.so3().unit_quaternion().coeffs();
            // do euler integration for now
            yMinus.template tail<3>() = poseMinus.V;

            Jy_fd.col(ii) = (yPlus-yMinus)/(2*dEps);
        }
        std::cout << "dRes_dy = " << std::endl << (dy_dy).format(cleanFmt) << std::endl;
        std::cout << "dRes_dy_fd = " << std::endl << Jy_fd.format(cleanFmt) << std::endl;
        std::cout << "dRes_dy-dRes_dy_fd = " << std::endl << (dy_dy-Jy_fd).format(cleanFmt) << " norm: " << (dy_dy-Jy_fd).norm() <<  std::endl;

        return (dy_dy-Jy_fd).norm() < NORM_THRESHOLD;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    static bool _Test_IntegrateImu_KBiasJacobian(const ImuPose&           y0,
                                                 const ImuMeasurement&    zStart,
                                                 const ImuMeasurement&    zEnd,
                                                 const Eigen::Matrix<Scalar,3,1>&  bg,
                                                 const Eigen::Matrix<Scalar,3,1>&  ba,
                                                 const Eigen::Matrix<Scalar,3,1>&  g,
                                                 const Eigen::Matrix<Scalar,9,6>&  dk_db)
    {
        const Scalar dEps = TESTING_EPS;
        Eigen::Matrix<Scalar,9,6>  dk_db_fd;
        for(int ii = 0; ii < 6 ; ii++){
            Eigen::Matrix<Scalar,6,1> biasVec;
            biasVec.template head<3>() = bg;
            biasVec.template tail<3>() = ba;

            Eigen::Matrix<Scalar,6,1> eps = Eigen::Matrix<Scalar,6,1>::Zero();
            eps[ii] += dEps;
            biasVec += eps;
            Eigen::Matrix<Scalar,9,1> k1_plus = GetPoseDerivative(y0,g,zStart,zEnd,biasVec.template head<3>(),biasVec.template tail<3>(),0);

            biasVec.template head<3>() = bg;
            biasVec.template tail<3>() = ba;

            eps[ii] -= 2*dEps;
            biasVec += eps;
            Eigen::Matrix<Scalar,9,1> k1_minus = GetPoseDerivative(y0,g,zStart,zEnd,biasVec.template head<3>(),biasVec.template tail<3>(),0);
            dk_db_fd.col(ii) = (k1_plus-k1_minus)/(2*dEps);
        }
        std::cout << "dk_db= " << std::endl << dk_db.format(cleanFmt) << std::endl;
        std::cout << "dk_db_fd = " << std::endl << dk_db_fd.format(cleanFmt) << std::endl;
        std::cout << "dk_db-dk_db_fd = " << std::endl << (dk_db-dk_db_fd).format(cleanFmt) << "norm: " << (dk_db-dk_db_fd).norm() << std::endl;

        return (dk_db-dk_db_fd).norm() < NORM_THRESHOLD;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    static bool _Test_IntegrateImu_KStateJacobian(const ImuPose&           y0,
                                                  const ImuMeasurement&    zStart,
                                                  const ImuMeasurement&    zEnd,
                                                  const Eigen::Matrix<Scalar,3,1>&  bg,
                                                  const Eigen::Matrix<Scalar,3,1>&  ba,
                                                  const Eigen::Matrix<Scalar,3,1>&  g,
                                                  const Eigen::Matrix<Scalar,9,10>&  dk_dy)
    {
        const Scalar dEps = TESTING_EPS;
        Eigen::Matrix<Scalar,9,10>  dk_dy_fd;
        for(int ii = 0; ii < 10 ; ii++){
            Eigen::Matrix<Scalar,10,1> epsVec = Eigen::Matrix<Scalar,10,1>::Zero();
            epsVec[ii] += dEps;
            ImuPose y0_eps = y0;
            y0_eps.Twp.translation() += epsVec.template head<3>();
            Eigen::Quaternion<Scalar> qPlus = y0_eps.Twp.so3().unit_quaternion();
            qPlus.coeffs() += epsVec.template segment<4>(3);
            memcpy(y0_eps.Twp.so3().data(),qPlus.coeffs().data(),sizeof(Scalar)*4);
            y0_eps.V += epsVec.template tail<3>();
            Eigen::Matrix<Scalar,9,1> k1_plus = GetPoseDerivative(y0_eps,g,zStart,zEnd,bg,ba,0);

            epsVec[ii] -= 2*dEps;
            y0_eps = y0;
            y0_eps.Twp.translation() += epsVec.template head<3>();
            Eigen::Quaternion<Scalar> qMinus = y0_eps.Twp.so3().unit_quaternion();
            qMinus.coeffs() += epsVec.template segment<4>(3);
            y0_eps.Twp.so3() = Sophus::SO3Group<Scalar>(qMinus);
            memcpy(y0_eps.Twp.so3().data(),qMinus.coeffs().data(),sizeof(Scalar)*4);
            y0_eps.V += epsVec.template tail<3>();
            Eigen::Matrix<Scalar,9,1> k1_minus = GetPoseDerivative(y0_eps,g,zStart,zEnd,bg,ba,0);

            dk_dy_fd.col(ii) = (k1_plus-k1_minus)/(2*dEps);
        }
        std::cout << "dk_dy= " << std::endl << dk_dy.format(cleanFmt) << std::endl;
        std::cout << "dk_dy_fd = " << std::endl << dk_dy_fd.format(cleanFmt) << std::endl;
        std::cout << "dk_dy-dk_dy_fd = " << std::endl << (dk_dy-dk_dy_fd).format(cleanFmt) << "norm: " << (dk_dy-dk_dy_fd).norm() <<  std::endl;

        return (dk_dy-dk_dy_fd).norm() < NORM_THRESHOLD;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    static bool _Test_IntegrateImu_StateStateJacobian(const ImuPose&           y0,
                                                      const Eigen::Matrix<Scalar,9,1>&  k,
                                                      const Eigen::Matrix<Scalar,10,10>&  dy_dy,
                                                      const Scalar dt)
    {
        const Scalar dEps = TESTING_EPS;
        Eigen::Matrix<Scalar,10,10>  dy_dy_fd;
        for(int ii = 0; ii < 10 ; ii++){
            Eigen::Matrix<Scalar,10,1> epsVec = Eigen::Matrix<Scalar,10,1>::Zero();
            epsVec[ii] += dEps;
            ImuPose y0_eps = y0;
            y0_eps.Twp.translation() += epsVec.template head<3>();
            Eigen::Quaternion<Scalar> qPlus = y0_eps.Twp.so3().unit_quaternion();
            qPlus.coeffs() += epsVec.template segment<4>(3);
            memcpy(y0_eps.Twp.so3().data(),qPlus.coeffs().data(),sizeof(Scalar)*4);
            y0_eps.V += epsVec.template tail<3>();

            Eigen::Matrix<Scalar,10,1> yPlus;
            ImuPose posePlus = IntegratePose(y0_eps,k,dt*0.5);
            yPlus.template head<3>() = posePlus.Twp.translation();
            yPlus.template segment<4>(3) = posePlus.Twp.so3().unit_quaternion().coeffs();
            yPlus.template tail<3>() = posePlus.V;

            epsVec[ii] -= 2*dEps;
            y0_eps = y0;
            y0_eps.Twp.translation() += epsVec.template head<3>();
            Eigen::Quaternion<Scalar> qMinus = y0_eps.Twp.so3().unit_quaternion();
            qMinus.coeffs() += epsVec.template segment<4>(3);
            y0_eps.Twp.so3() = Sophus::SO3Group<Scalar>(qMinus);
            memcpy(y0_eps.Twp.so3().data(),qMinus.coeffs().data(),sizeof(Scalar)*4);
            y0_eps.V += epsVec.template tail<3>();

            Eigen::Matrix<Scalar,10,1> yMinus;
            ImuPose poseMinus = IntegratePose(y0_eps,k,dt*0.5);
            yMinus.template head<3>() = poseMinus.Twp.translation();
            yMinus.template segment<4>(3) = poseMinus.Twp.so3().unit_quaternion().coeffs();
            yMinus.template tail<3>() = poseMinus.V;

            dy_dy_fd.col(ii) = (yPlus-yMinus)/(2*dEps);
        }
        std::cout << "dy_dy= " << std::endl << dy_dy.format(cleanFmt) << std::endl;
        std::cout << "dy_dy_fd = " << std::endl << dy_dy_fd.format(cleanFmt) << std::endl;
        std::cout << "dy_dy-dy_dy_fd = " << std::endl << (dy_dy-dy_dy_fd).format(cleanFmt) << "norm: " << (dy_dy-dy_dy_fd).norm() << std::endl;

        return (dy_dy-dy_dy_fd).norm() < NORM_THRESHOLD;
    }



    ///////////////////////////////////////////////////////////////////////////////////////////////
    static bool _Test_IntegratePose_ExpJacobian(const Eigen::Matrix<Scalar,9,1>&    k,
                                                const Scalar                        dt)
    {
        const Scalar dEps = TESTING_EPS;
        Eigen::Matrix<Scalar,4,3> Jexp_fd;
        for(int ii = 0; ii < 3 ; ii++){
            Eigen::Matrix<Scalar,3,1> eps = Eigen::Matrix<Scalar,3,1>::Zero();
            eps[ii] += dEps;
            Eigen::Matrix<Scalar,3,1> kPlus = k.template segment<3>(3)*dt;
            kPlus += eps;
            Eigen::Matrix<Scalar,4,1> res_Plus = Sophus::SO3Group<Scalar>::exp(kPlus).unit_quaternion().coeffs();

            eps[ii] -= 2*dEps;
            Eigen::Matrix<Scalar,3,1> kMinus = k.template segment<3>(3)*dt;
            kMinus += eps;
            Eigen::Matrix<Scalar,4,1> res_Minus = Sophus::SO3Group<Scalar>::exp(kMinus).unit_quaternion().coeffs();

            Jexp_fd.col(ii) = (res_Plus - res_Minus)/(2*dEps);
        }

        std::cout << "dExp_dw= " << std::endl << dqExp_dw<Scalar>(k.template segment<3>(3)*dt).format(cleanFmt) <<  std::endl;
        std::cout << "dExp_dw_fd=" << std::endl << Jexp_fd.format(cleanFmt) << std::endl;
        std::cout << "dExp_dw-dExp_dw_fd=" << std::endl << (dqExp_dw<Scalar>(k.template segment<3>(3)*dt)-Jexp_fd).format(cleanFmt) <<
                     "norm: " << (dqExp_dw<Scalar>(k.template segment<3>(3)*dt)-Jexp_fd).norm() << std::endl;

        return (dqExp_dw<Scalar>(k.template segment<3>(3)*dt)-Jexp_fd).norm() < NORM_THRESHOLD;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    static bool _Test_IntegratePose_StateKJacobian(const ImuPose&             pose,
                                                   const Eigen::Matrix<Scalar,9,1>&    k,
                                                   const Scalar                        dt,
                                                   const Eigen::Matrix<Scalar,10,9>&   dy_dk)
    {
        const Scalar dEps = TESTING_EPS;
        Eigen::Matrix<Scalar,10,9> Jfd;
        for(int ii = 0; ii < 9 ; ii++){
            Eigen::Matrix<Scalar,9,1> eps = Eigen::Matrix<Scalar,9,1>::Zero();
            eps[ii] += dEps;
            Eigen::Matrix<Scalar,9,1> kPlus = k;
            kPlus += eps;
            Eigen::Matrix<Scalar,10,1> res_Plus;
            res_Plus.template head<3>() = pose.Twp.translation() + kPlus.template head<3>()*dt;
            res_Plus.template segment<4>(3) = (Sophus::SO3Group<Scalar>::exp(kPlus.template segment<3>(3)*dt).unit_quaternion() * pose.Twp.so3().unit_quaternion()).coeffs();
            res_Plus.template tail<3>() = pose.V + kPlus.template tail<3>()*dt;

            eps[ii] -= 2*dEps;
            Eigen::Matrix<Scalar,9,1> kMinus = k;
            kMinus += eps;
            Eigen::Matrix<Scalar,10,1> res_Minus;
            res_Minus.template head<3>() = pose.Twp.translation() + kMinus.template head<3>()*dt;
            res_Minus.template segment<4>(3) = (Sophus::SO3Group<Scalar>::exp(kMinus.template segment<3>(3)*dt) * pose.Twp.so3()).unit_quaternion().coeffs();
            res_Minus.template tail<3>() = pose.V + kMinus.template tail<3>()*dt;

            Jfd.col(ii) = (res_Plus - res_Minus)/(2*dEps);
        }

        std::cout << "dy_dk=" << std::endl << dy_dk.format(cleanFmt) <<  std::endl;
        std::cout << "dy_dk_fd=" << std::endl << Jfd.format(cleanFmt) << std::endl;
        std::cout << "dy_dk-dy_dk_fd=" << std::endl << (dy_dk-Jfd).format(cleanFmt) << "norm: " << (dy_dk-Jfd).norm() << std::endl;

        return (dy_dk-Jfd).norm() < NORM_THRESHOLD;
    }

};



}
