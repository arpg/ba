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
    static ImuPose IntegratePose(const ImuPose& pose, const Eigen::Matrix<double,9,1>& k, const double dt,
                                 Eigen::Matrix<double,10,9>* pdy_dk = 0,Eigen::Matrix<double,10,10>* pdy_dy = 0)
    {
        const Sophus::SO3d Rv2v1(Sophus::SO3d::exp(k.segment<3>(3)*dt));

        ImuPose y = pose;
        y.Twp.translation() += k.head<3>()*dt;
//        y.Twp.so3() = Rv2v1*pose.Twp.so3();
        memcpy(y.Twp.so3().data(),(Rv2v1.unit_quaternion()*pose.Twp.so3().unit_quaternion()).coeffs().data(),sizeof(double)*4);
        // do euler integration for now
        y.V += k.tail<3>()*dt;

        // jacobian of output pose relative to the derivative
        if( pdy_dk != 0 ){
            pdy_dk->setZero();
            pdy_dk->block<3,3>(0,0) = Eigen::Matrix3d::Identity()*dt;  // dt/dv
            pdy_dk->block<4,3>(3,3) = dq1q2_dq1(pose.Twp.so3().unit_quaternion()) * dqExp_dw(k.segment<3>(3)*dt) *dt;  // dq/dw
            pdy_dk->block<3,3>(7,6) = Eigen::Matrix3d::Identity()*dt;  // dv/da
        }

        if( pdy_dy != 0 )
        {
            pdy_dy->setZero();
            pdy_dy->block<3,3>(0,0) = Eigen::Matrix3d::Identity();
            pdy_dy->block<4,4>(3,3) = dq1q2_dq2(Rv2v1.unit_quaternion());
            pdy_dy->block<3,3>(7,7) = Eigen::Matrix3d::Identity();


            const double dEps = 1e-9;
            Eigen::Matrix<double,4,3> Jexp_fd;
            for(int ii = 0; ii < 3 ; ii++){
                Eigen::Vector3d eps = Eigen::Vector3d::Zero();
                eps[ii] += dEps;
                Eigen::Vector3d kPlus = k.segment<3>(3)*dt;
                kPlus += eps;
                Eigen::Vector4d res_Plus = Sophus::SO3d::exp(kPlus).unit_quaternion().coeffs();

                eps[ii] -= 2*dEps;
                Eigen::Vector3d kMinus = k.segment<3>(3)*dt;
                kMinus += eps;
                Eigen::Vector4d res_Minus = Sophus::SO3d::exp(kMinus).unit_quaternion().coeffs();

                Jexp_fd.col(ii) = (res_Plus - res_Minus)/(2*dEps);
            }

            std::cout << "Jexp= [" << std::endl << dqExp_dw(k.segment<3>(3)*dt) <<  "]" << std::endl;
            std::cout << "Jexpfd=[" << std::endl << Jexp_fd << "]" << std::endl;
            std::cout << "Jexp-Jexpfd=[" << std::endl << dqExp_dw(k.segment<3>(3)*dt)-Jexp_fd << "]" << std::endl;


            Eigen::Matrix<double,10,9> Jfd;
            for(int ii = 0; ii < 9 ; ii++){
                Eigen::Matrix<double,9,1> eps = Eigen::Matrix<double,9,1>::Zero();
                eps[ii] += dEps;
                Eigen::Matrix<double,9,1> kPlus = k;
                kPlus += eps;
                Eigen::Matrix<double,10,1> res_Plus;
                res_Plus.head<3>() = pose.Twp.translation() + kPlus.head<3>()*dt;
                res_Plus.segment<4>(3) = (Sophus::SO3d::exp(kPlus.segment<3>(3)*dt).unit_quaternion() * pose.Twp.so3().unit_quaternion()).coeffs();
                res_Plus.tail<3>() = pose.V + kPlus.tail<3>()*dt;

                eps[ii] -= 2*dEps;
                Eigen::Matrix<double,9,1> kMinus = k;
                kMinus += eps;
                Eigen::Matrix<double,10,1> res_Minus;
                res_Minus.head<3>() = pose.Twp.translation() + kMinus.head<3>()*dt;
                res_Minus.segment<4>(3) = (Sophus::SO3d::exp(kMinus.segment<3>(3)*dt) * pose.Twp.so3()).unit_quaternion().coeffs();
                res_Minus.tail<3>() = pose.V + kMinus.tail<3>()*dt;

                Jfd.col(ii) = (res_Plus - res_Minus)/(2*dEps);
            }

            std::cout << "Jexptimes= [" << std::endl << *pdy_dk <<  "]" << std::endl;
            std::cout << "Jexptimesfd=[" << std::endl << Jfd << "]" << std::endl;
            std::cout << "Jexptimes-Jexptimesfd=[" << std::endl << *pdy_dk-Jfd << "]" << std::endl;
        }

        return y;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    static Eigen::Matrix<double,9,1> GetPoseDerivative(const ImuPose& pose, const Eigen::Vector3d& tG_w, const ImuMeasurement& zStart,
                                                       const ImuMeasurement& zEnd, const Eigen::Vector3d& vBg,
                                                       const Eigen::Vector3d& vBa, const double dt,
                                                       Eigen::Matrix<double,9,6>* dk_db = 0,Eigen::Matrix<double,9,10>* dk_dx = 0)
    {
        double alpha = (zEnd.Time - (zStart.Time+dt))/(zEnd.Time - zStart.Time);
        Eigen::Vector3d zg = zStart.W*alpha + zEnd.W*(1.0-alpha);
        Eigen::Vector3d za = zStart.A*alpha + zEnd.A*(1.0-alpha);

        Eigen::Matrix<double,9,1> deriv;
        //derivative of position is velocity
        deriv.head<3>() = pose.V;                               // v (velocity)
        //deriv.template segment<3>(3) = Sophus::SO3Group<T>::vee(tTwb.so3().matrix()*Sophus::SO3Group<T>::hat(zb));
        deriv.segment<3>(3) = pose.Twp.so3().Adj()*(zg+vBg);    // w (angular rates)
        deriv.segment<3>(6) = pose.Twp.so3()*(za+vBa) - tG_w;   // a (acceleration)

//        Eigen::IOFormat cleanFmt(2, 0, ", ", ";\n" , "" , "");
//        const double dEps = 1e-9;
//        Eigen::Matrix<double,3,4> Jfd;
//        for(int ii = 0; ii < 4 ; ii++){
//            Eigen::Vector4d eps = Eigen::Vector4d::Zero();
//            eps[ii] += dEps;
//            Eigen::Quaterniond q_plus = pose.Twp.so3().unit_quaternion();
//            q_plus.coeffs() += eps;
//            const Eigen::Vector3d plus = q_plus._transformVector(zg) + q_plus._transformVector(vBg);

//            eps[ii] -= 2*dEps;
//            Eigen::Quaterniond q_minus = pose.Twp.so3().unit_quaternion();
//            q_minus.coeffs() += eps;
//            const Eigen::Vector3d minus = q_minus._transformVector(zg) + q_plus._transformVector(vBg);
//            Jfd.col(ii) = (plus-minus)/(2*dEps);
//        }
//        std::cout << "J=[" << (dqx_dq(pose.Twp.so3().unit_quaternion(),zg) + dqx_dq(pose.Twp.so3().unit_quaternion(),vBg)).format(cleanFmt) << "]" << std::endl;
//        std::cout << "Jfd=[" << Jfd.format(cleanFmt) << "]" << std::endl;

        if(dk_db != 0) {
            dk_db->setZero();
            dk_db->block<3,3>(3,0) = pose.Twp.so3().Adj(); // dw/dbg
            dk_db->block<3,3>(6,3) = pose.Twp.so3().matrix();       // da/dba
        }
        if(dk_dx != 0){
            dk_dx->setZero();
            dk_dx->block<3,3>(0,7) = Eigen::Matrix3d::Identity(); // dv/dv
            dk_dx->block<3,4>(3,3) = dqx_dq(pose.Twp.so3().unit_quaternion(),zg) + dqx_dq(pose.Twp.so3().unit_quaternion(),vBg); // dw/dq
            dk_dx->block<3,4>(6,3) = dqx_dq(pose.Twp.so3().unit_quaternion(),za) + dqx_dq(pose.Twp.so3().unit_quaternion(),vBa); // da/dq
        }
        return deriv;
    }


    ///////////////////////////////////////////////////////////////////////////////////////////////
    static ImuPose IntegrateImu(const ImuPose& y0, const ImuMeasurement& zStart,
                     const ImuMeasurement& zEnd, const Eigen::Vector3d& vBg,
                     const Eigen::Vector3d& vBa,const Eigen::Vector3d& dG,
                     Eigen::Matrix<double,10,6>* pDy_db = 0,
                     Eigen::Matrix<double,10,10>* pDy_dy0 = 0)
    {
        //construct the state matrix
        double dt = zEnd.Time - zStart.Time;
        if(dt == 0){
            return y0;
        }


        ImuPose res = y0;
        Eigen::Matrix<double,9,1> k;

        if(pDy_db != 0) {
            Eigen::Matrix<double,10,6>& dy_db = *pDy_db;
            Eigen::Matrix<double,10,10>& dy_dy0 = *pDy_dy0;

            Eigen::Matrix<double,9,6>  dk_db;
            Eigen::Matrix<double,9,10> dk_dy;
            Eigen::Matrix<double,10,9> dy_dk;
            Eigen::Matrix<double,10,10> dy_dy;
            dy_db.setZero();
            dy_dy0.setIdentity();   // dy0_y0 starts at identity

            const Eigen::Matrix<double,9,1> k1 = GetPoseDerivative(y0,dG,zStart,zEnd,vBg,vBa,0,&dk_db,&dk_dy);
            const double dEps = 1e-9;
            Eigen::Matrix<double,9,6>  dk_db_fd;
            for(int ii = 0; ii < 6 ; ii++){
                Eigen::Vector6d biasVec;
                biasVec.head<3>() = vBg;
                biasVec.tail<3>() = vBa;

                Eigen::Vector6d eps = Eigen::Vector6d::Zero();
                eps[ii] += dEps;
                biasVec += eps;
                Eigen::Matrix<double,9,1> k1_plus = GetPoseDerivative(y0,dG,zStart,zEnd,biasVec.head<3>(),biasVec.tail<3>(),0);

                biasVec.head<3>() = vBg;
                biasVec.tail<3>() = vBa;

                eps[ii] -= 2*dEps;
                biasVec += eps;
                Eigen::Matrix<double,9,1> k1_minus = GetPoseDerivative(y0,dG,zStart,zEnd,biasVec.head<3>(),biasVec.tail<3>(),0);
                dk_db_fd.col(ii) = (k1_plus-k1_minus)/(2*dEps);
            }
            std::cout << "dk_db= " << std::endl << dk_db << std::endl;
            std::cout << "dk_db_fd = " << std::endl << dk_db_fd << std::endl;
            std::cout << "dk_db-dk_db_fd = " << std::endl << dk_db-dk_db_fd << "norm: " << (dk_db-dk_db_fd).norm() << std::endl;


            Eigen::Matrix<double,9,10>  dk_dy_fd;
            for(int ii = 0; ii < 10 ; ii++){
                Eigen::Matrix<double,10,1> epsVec = Eigen::Matrix<double,10,1>::Zero();
                epsVec[ii] += dEps;
                ImuPose y0_eps = y0;
                y0_eps.Twp.translation() += epsVec.head<3>();
                Eigen::Quaterniond qPlus = y0_eps.Twp.so3().unit_quaternion();
                qPlus.coeffs() += epsVec.segment<4>(3);
                memcpy(y0_eps.Twp.so3().data(),qPlus.coeffs().data(),sizeof(double)*4);
                y0_eps.V += epsVec.tail<3>();
                Eigen::Matrix<double,9,1> k1_plus = GetPoseDerivative(y0_eps,dG,zStart,zEnd,vBg,vBa,0);

                epsVec[ii] -= 2*dEps;
                y0_eps = y0;
                y0_eps.Twp.translation() += epsVec.head<3>();
                Eigen::Quaterniond qMinus = y0_eps.Twp.so3().unit_quaternion();
                qMinus.coeffs() += epsVec.segment<4>(3);
                y0_eps.Twp.so3() = Sophus::SO3d(qMinus);
                memcpy(y0_eps.Twp.so3().data(),qMinus.coeffs().data(),sizeof(double)*4);
                y0_eps.V += epsVec.tail<3>();
                Eigen::Matrix<double,9,1> k1_minus = GetPoseDerivative(y0_eps,dG,zStart,zEnd,vBg,vBa,0);

                dk_dy_fd.col(ii) = (k1_plus-k1_minus)/(2*dEps);
            }
            std::cout << "dk_dy= " << std::endl << dk_dy << std::endl;
            std::cout << "dk_dy_fd = " << std::endl << dk_dy_fd << std::endl;
            std::cout << "dk_dy-dk_dy_fd = " << std::endl << dk_dy-dk_dy_fd << "norm: " << (dk_dy-dk_dy_fd).norm() <<  std::endl;


            // total derivative of k1 wrt b: dk1/db = dG/db + dG/dy*dy/db
            const Eigen::Matrix<double,9,6> dk1_db = dk_db + dk_dy*dy_db;
            const Eigen::Matrix<double,9,10> dk1_dy= dk_dy*dy_dy0;
            const ImuPose y1 = IntegratePose(y0,k1,dt*0.5,&dy_dk,&dy_dy);
            //dy1/db = dInt/db + dInt/dy*dy0/db + dInt/dk*dk/db
            // however dy0/db = 0 (only for y0), therefore we don't need the second term, just dInt/dk and dInt/db.
            // but dInt/db is also 0, as the integration doesn't directly depend on b
            dy_db = dy_dk*dk1_db;
            dy_dy0 = dy_dy + dy_dk*dk1_dy; // this is dy1_dy0

            Eigen::Matrix<double,10,10>  dy_dy_fd;
            for(int ii = 0; ii < 10 ; ii++){
                Eigen::Matrix<double,10,1> epsVec = Eigen::Matrix<double,10,1>::Zero();
                epsVec[ii] += dEps;
                ImuPose y0_eps = y0;
                y0_eps.Twp.translation() += epsVec.head<3>();
                Eigen::Quaterniond qPlus = y0_eps.Twp.so3().unit_quaternion();
                qPlus.coeffs() += epsVec.segment<4>(3);
                memcpy(y0_eps.Twp.so3().data(),qPlus.coeffs().data(),sizeof(double)*4);
                y0_eps.V += epsVec.tail<3>();

                Eigen::Matrix<double,10,1> yPlus;
                ImuPose posePlus = IntegratePose(y0_eps,k1,dt*0.5);
                yPlus.head<3>() = posePlus.Twp.translation();
                yPlus.segment<4>(3) = posePlus.Twp.so3().unit_quaternion().coeffs();
                yPlus.tail<3>() = posePlus.V;

                epsVec[ii] -= 2*dEps;
                y0_eps = y0;
                y0_eps.Twp.translation() += epsVec.head<3>();
                Eigen::Quaterniond qMinus = y0_eps.Twp.so3().unit_quaternion();
                qMinus.coeffs() += epsVec.segment<4>(3);
                y0_eps.Twp.so3() = Sophus::SO3d(qMinus);
                memcpy(y0_eps.Twp.so3().data(),qMinus.coeffs().data(),sizeof(double)*4);
                y0_eps.V += epsVec.tail<3>();

                Eigen::Matrix<double,10,1> yMinus;
                ImuPose poseMinus = IntegratePose(y0_eps,k1,dt*0.5);
                yMinus.head<3>() = poseMinus.Twp.translation();
                yMinus.segment<4>(3) = poseMinus.Twp.so3().unit_quaternion().coeffs();
                yMinus.tail<3>() = poseMinus.V;

                dy_dy_fd.col(ii) = (yPlus-yMinus)/(2*dEps);
            }
            std::cout << "dy_dy= " << std::endl << dy_dy << std::endl;
            std::cout << "dy_dy_fd = " << std::endl << dy_dy_fd << std::endl;
            std::cout << "dy_dy-dy_dy_fd = " << std::endl << dy_dy-dy_dy_fd << "norm: " << (dy_dy-dy_dy_fd).norm() << std::endl;


            const Eigen::Matrix<double,9,1> k2 = GetPoseDerivative(y1,dG,zStart,zEnd,vBg,vBa,dt/2,&dk_db,&dk_dy);
            const Eigen::Matrix<double,9,6> dk2_db = dk_db + dk_dy*dy_db;
            const Eigen::Matrix<double,9,10> dk2_dy= dk_dy*dy_dy0;
            const ImuPose y2 = IntegratePose(y0,k2,dt*0.5,&dy_dk,&dy_dy);
            dy_db = dy_dk*dk2_db;
            dy_dy0 = dy_dy + dy_dk*dk2_dy; // this is dy2_dy0

            const Eigen::Matrix<double,9,1> k3 = GetPoseDerivative(y2,dG,zStart,zEnd,vBg,vBa,dt/2,&dk_db,&dk_dy);
            const Eigen::Matrix<double,9,6> dk3_db = dk_db + dk_dy*dy_db;
            const Eigen::Matrix<double,9,10> dk3_dy = dk_dy*dy_dy0;
            const ImuPose y3 = IntegratePose(y0,k3,dt,&dy_dk,&dy_dy);
            dy_db = dy_dk*dk3_db;
            dy_dy0 = dy_dy + dy_dk*dk3_dy; // this is dy3_dy0

            const Eigen::Matrix<double,9,1> k4 = GetPoseDerivative(y3,dG,zStart,zEnd,vBg,vBa,dt,&dk_db,&dk_dy);
            const Eigen::Matrix<double,9,6> dk4_db = dk_db + dk_dy*dy_db;
            const Eigen::Matrix<double,9,10> dk4_dy = dk_dy*dy_dy0;

            k = (k1+2*k2+2*k3+k4);
            const Eigen::Matrix<double,9,6> dk_total_db = dk1_db + 2*dk2_db + 2*dk3_db + dk4_db;
            const Eigen::Matrix<double,9,10> dk_total_dy = dk1_dy + 2*dk2_dy + 2*dk3_dy + dk4_dy;
            res = IntegratePose(y0,k, dt/6.0,&dy_dk,&dy_dy);
            dy_db = dy_dk*dk_total_db;
            dy_dy0 = dy_dy + dy_dk*dk_total_dy;

        }else{
            const Eigen::Matrix<double,9,1> k1 = GetPoseDerivative(y0,dG,zStart,zEnd,vBg,vBa,0);
            const ImuPose y1 = IntegratePose(y0,k1,dt*0.5);
            const Eigen::Matrix<double,9,1> k2 = GetPoseDerivative(y1,dG,zStart,zEnd,vBg,vBa,dt/2);
            const ImuPose y2 = IntegratePose(y0,k2,dt*0.5);
            const Eigen::Matrix<double,9,1> k3 = GetPoseDerivative(y2,dG,zStart,zEnd,vBg,vBa,dt/2);
            const ImuPose y3 = IntegratePose(y0,k3,dt);
            const Eigen::Matrix<double,9,1> k4 = GetPoseDerivative(y3,dG,zStart,zEnd,vBg,vBa,dt);
            k = (k1+2*k2+2*k3+k4);
            res = IntegratePose(y0,k, dt/6.0);
        }

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
                                     std::vector<ImuPose>& vPoses,
                                     Eigen::Matrix<double,10,6>* pJb = 0)
    {
        ImuPose imuPose(pose.Twp,pose.V,Eigen::Vector3d::Zero(),pose.Time);
        const ImuPose origPose = imuPose;
        const ImuMeasurement* pPrevMeas = 0;
        vPoses.clear();
        vPoses.reserve(vMeasurements.size()+1);
        vPoses.push_back(imuPose);

        if(pJb != 0){
            pJb->setZero();
        }

        // integrate forward in time, and retain all the poses
        for(const ImuMeasurement& meas : vMeasurements){
            if(pPrevMeas != 0){
//                std::cout << "Integrating from time " << pPrevMeas->Time << " to " << meas.Time << std::endl;
                if( pJb != 0 ){
                    double dt = meas.Time - pPrevMeas->Time;
                    Eigen::Matrix<double,10,6> dy_db;
                    Eigen::Matrix<double,10,10> dy_dy;
                    const ImuPose y0 = imuPose;
                    imuPose = IntegrateImu(imuPose,*pPrevMeas,meas,bg,ba,g,&dy_db,&dy_dy);

                    Eigen::Matrix<double, 10,6> J_fd;
                    const double dEps = 1e-9;
                    for(int ii = 0; ii < 6 ; ii++){
                        Eigen::Vector6d biasVec;
                        biasVec.head<3>() = bg;
                        biasVec.tail<3>() = ba;

                        Eigen::Vector6d eps = Eigen::Vector6d::Zero();
                        eps[ii] += dEps;
                        biasVec += eps;
                        ImuPose posePlus = IntegrateImu(y0,*pPrevMeas,meas,biasVec.head<3>(),biasVec.tail<3>(),g);

                        Eigen::Matrix<double,10,1> yPlus;
                        yPlus.head<3>() = posePlus.Twp.translation();
                        yPlus.segment<4>(3) = posePlus.Twp.so3().unit_quaternion().coeffs();
                        // do euler integration for now
                        yPlus.tail<3>() = posePlus.V;

                        biasVec.head<3>() = bg;
                        biasVec.tail<3>() = ba;

                        eps[ii] -= 2*dEps;
                        biasVec += eps;
                        ImuPose poseMinus = IntegrateImu(y0,*pPrevMeas,meas,biasVec.head<3>(),biasVec.tail<3>(),g);

                        Eigen::Matrix<double,10,1> yMinus;
                        yMinus.head<3>() = poseMinus.Twp.translation();
                        yMinus.segment<4>(3) = poseMinus.Twp.so3().unit_quaternion().coeffs();
                        // do euler integration for now
                        yMinus.tail<3>() = poseMinus.V;

                        J_fd.col(ii) = (yPlus-yMinus)/(2*dEps);
                    }
                    std::cout << "dInt_db = " << std::endl << dy_db << std::endl;
                    std::cout << "dInt_db_fd = " << std::endl << J_fd << std::endl;
                    std::cout << "dInt_db-dInt_db_fd = " << std::endl << (dy_db-J_fd) << " norm: " << (dy_db-J_fd).norm() <<  std::endl;


                    Eigen::Matrix<double,10,10>  dx_dx_fd;
                    for(int ii = 0; ii < 10 ; ii++){
                        Eigen::Matrix<double,10,1> epsVec = Eigen::Matrix<double,10,1>::Zero();
                        epsVec[ii] += dEps;
                        ImuPose y0_eps = y0;
                        y0_eps.Twp.translation() += epsVec.head<3>();
                        Eigen::Quaterniond qPlus = y0_eps.Twp.so3().unit_quaternion();
                        qPlus.coeffs() += epsVec.segment<4>(3);
                        memcpy(y0_eps.Twp.so3().data(),qPlus.coeffs().data(),sizeof(double)*4);
                        y0_eps.V += epsVec.tail<3>();

                        ImuPose posePlus = IntegrateImu(y0_eps,*pPrevMeas,meas,bg,ba,g);

                        Eigen::Matrix<double,10,1> yPlus;
                        yPlus.head<3>() = posePlus.Twp.translation();
                        yPlus.segment<4>(3) = posePlus.Twp.so3().unit_quaternion().coeffs();
                        // do euler integration for now
                        yPlus.tail<3>() = posePlus.V;

                        epsVec[ii] -= 2*dEps;
                        y0_eps = y0;
                        y0_eps.Twp.translation() += epsVec.head<3>();
                        Eigen::Quaterniond qMinus = y0_eps.Twp.so3().unit_quaternion();
                        qMinus.coeffs() += epsVec.segment<4>(3);
                        y0_eps.Twp.so3() = Sophus::SO3d(qMinus);
                        memcpy(y0_eps.Twp.so3().data(),qMinus.coeffs().data(),sizeof(double)*4);
                        y0_eps.V += epsVec.tail<3>();

                        ImuPose poseMinus = IntegrateImu(y0_eps,*pPrevMeas,meas,bg,ba,g);

                        Eigen::Matrix<double,10,1> yMinus;
                        yMinus.head<3>()= poseMinus.Twp.translation();
                        yMinus.segment<4>(3) = poseMinus.Twp.so3().unit_quaternion().coeffs();
                        // do euler integration for now
                        yMinus.tail<3>() = poseMinus.V;

                        dx_dx_fd.col(ii) = (yPlus-yMinus)/(2*dEps);
                    }
                    std::cout << "dInt_dy= " << std::endl << dy_dy << std::endl;
                    std::cout << "dInt_dy_fd = " << std::endl << dx_dx_fd << std::endl;
                    std::cout << "dInt_dy-dInt_dy_fd = " << std::endl << dy_dy-dx_dx_fd << " norm: " << (dy_dy-dx_dx_fd).norm() << std::endl;

                    // now push forward the jacobian. This calculates the total derivative Jb = dG/dB + dG/dX * dX/dB
                    // where dG/dB is the jacobian of the return values of IntegrateImu with respect to the bias
                    // values (which is returned in dq_dBg and dv_dBa, as the other values are 0). dG/dX is the
                    // jacobian of the IntegrateImu function with respect to its inputs (a 10x10 matrix, but only
                    // dq2_dq1 is complex, and is returned by IntegrateImu). dX/dB yis the jacobian from the previous
                    // step, which is stored in Jb. The following is the addition and multiplication unrolled into
                    // sparse operations.
                    (*pJb) = dy_db + dy_dy*(*pJb);
                }else{
                    imuPose = IntegrateImu(imuPose,*pPrevMeas,meas,bg,ba,g);
                }
                vPoses.push_back(imuPose);
            }
            pPrevMeas = &meas;
        }

        if( pJb != 0 ){
            Eigen::Matrix<double,10,6> Jb_fd;
            const double dEps = 1e-9;
            Eigen::Vector6d biasVec;
            biasVec.head<3>() = bg;
            biasVec.tail<3>() = ba;
            for(int ii = 0 ; ii < 6 ; ii++){
                Eigen::Vector6d eps = Eigen::Vector6d::Zero();
                eps[ii] += dEps;
                std::vector<ImuPose> poses;
                const Eigen::Vector6d plusBiases = biasVec + eps;
                ImuPose imuPosePlus = origPose;
                pPrevMeas = 0;
                for(const ImuMeasurement& meas : vMeasurements){
                    if(pPrevMeas != 0){
                        imuPosePlus = IntegrateImu(imuPosePlus,*pPrevMeas,meas,plusBiases.head<3>(),plusBiases.tail<3>(),g);
                    }
                    pPrevMeas = &meas;
                }
                Eigen::Matrix<double,10,1> plusVec;
                plusVec.head<3>() = imuPosePlus.Twp.translation();
                plusVec.segment<4>(3) = imuPosePlus.Twp.so3().unit_quaternion().coeffs();
                plusVec.tail<3>() = imuPosePlus.V;

                eps[ii] -= 2*dEps;
                const Eigen::Vector6d minusBiases = biasVec + eps;
                poses.clear();
                ImuPose imuPoseMinus = origPose;
                pPrevMeas = 0;
                for(const ImuMeasurement& meas : vMeasurements){
                    if(pPrevMeas != 0){
                        imuPoseMinus = IntegrateImu(imuPoseMinus,*pPrevMeas,meas,minusBiases.head<3>(),minusBiases.tail<3>(),g);
                    }
                    pPrevMeas = &meas;
                }
                Eigen::Matrix<double,10,1> minusVec;
                minusVec.head<3>() = imuPoseMinus.Twp.translation();
                minusVec.segment<4>(3) = imuPoseMinus.Twp.so3().unit_quaternion().coeffs();
                minusVec.tail<3>() = imuPoseMinus.V;

                Jb_fd.col(ii) = (plusVec-minusVec)/(2*dEps);
            }
            std::cout << "Jres = " << std::endl << *pJb << std::endl;
            std::cout << "Jres_fd = " << std::endl << Jb_fd << std::endl;
            std::cout << "Jres-Jres_fd = " << std::endl << (*pJb-Jb_fd) << " norm: " << (*pJb-Jb_fd).norm() <<  std::endl;
        }
        return imuPose;
    }


};



}
