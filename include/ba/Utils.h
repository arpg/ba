#pragma once
#include <Eigen/Eigen>
#include <sophus/se3.hpp>
#include <sys/time.h>
#include <time.h>

namespace Eigen
{
    typedef Matrix<double,6,1> Vector6d;
    typedef Matrix<double,9,1> Vector9d;
}

///////////////////////////////////////////////////////////////////////////////
namespace ba
{
    static Eigen::IOFormat cleanFmt(4, 0, ", ", ";\n" , "" , "");
    #define NORM_THRESHOLD 1e-4
    #define TESTING_EPS 1e-9

    template<typename Scalar=double>
    inline Eigen::Matrix<Scalar,4,1> MultHomogeneous( const Sophus::SE3Group<Scalar>& lhs, const Eigen::Matrix<Scalar,4,1>& rhs )
    {
        Eigen::Matrix<Scalar,4,1> out;
        out.template head<3>() = lhs.so3() * (Eigen::Matrix<Scalar,3,1>)rhs.template head<3>() + lhs.translation()*rhs[3];
        out[3] = rhs[3];
        return out;
    }

    ///////////////////////////////////////////////////////////////////////////////
//    template<class T>
//    inline constexpr T powi(const T base, unsigned const exponent)
//    {
//        // (parentheses not required in next line)
//        return (exponent == 0) ? 1 : (base * pow(base, exponent-1));
//    }

    ///////////////////////////////////////////////////////////////////////////////
    template<typename Scalar=double>
    inline Scalar powi(const Scalar x, const int y)
    {
        if(y == 0){
            return 1.0;
        }else if(y < 0 ){
            return 1.0/powi(x,-y);
        }else if(y == 0){
            return 1.0;
        }else{
            double ret = x;
            for(int ii = 1; ii <  y ; ii++){
                ret *= x;
            }
            return ret;
        }
    }

    inline double Tic() {
        struct timeval tv;
        gettimeofday(&tv, 0);
        return tv.tv_sec  + 1e-6 * (tv.tv_usec);
    }

    inline double Toc(double dTic) {
        return Tic() - dTic;
    }

    ///////////////////////////////////////////////////////////////////////////////
    // this function implements d vee(log(A * exp(x) * B) ) / dx , which is in R^{6x6}
    template<typename Scalar=double>
    inline Eigen::Matrix<Scalar,6,6> dLog_dX(const Sophus::SE3Group<Scalar>& A,const Sophus::SE3Group<Scalar>& B)
    {
        const Eigen::Matrix<Scalar,6,1> d_2 = Sophus::SE3Group<Scalar>::log(A*B)/2;
        const Scalar d1 = d_2[3], d2 = d_2[4], d3 = d_2[5], dx = d_2[0], dy = d_2[1], dz = d_2[2];
        // this is using the 2nd order cbh expansion, to evaluate (I + 0.5 [Adj*x, log(AB)])*Adj
        // refer to the thesis by Hauke Strasdat, Appendix 3.
        return (Eigen::Matrix<Scalar, 6, 6>() <<
                  1,  d3, -d2,   0,  dz, -dy,
                -d3,   1,  d1, -dz,   0,  dx,
                 d2, -d1,   1,  dy, -dx,   0,
                  0,   0,   0,   1,  d3, -d2,
                  0,   0,   0, -d3,   1,  d1,
                  0,   0,   0,  d2, -d1,   1
               ).finished() * A.Adj();
    }

    ///////////////////////////////////////////////////////////////////////////////
    // this function implements the derivative of log with respect to the input quaternion
    template<typename Scalar=double>
    inline Eigen::Matrix<Scalar,3,4> dLog_dq(const Eigen::Quaternion<Scalar>& q)
    {
        const Scalar vec_squarednorm = powi(q.x(),2) + powi(q.y(),2) + powi(q.z(),2);
        const Scalar vec_norm = sqrt(vec_squarednorm);
        // std::cout << "vec norm = " << vec_norm << std::endl;
        if( vec_norm < 1e-9 ){
            const Scalar s1 = 2*vec_squarednorm;
            const Scalar s2 = 1.0/powi(q.w(),3);
            const Scalar s3 = (3*s1)/powi(q.w(),4) - 2/powi(q.w(),2);
            const Scalar s4 = 2/q.w();

            // std::cout << " s1 " << s1 << " s2 " << s2 << " s3 " << s3 << " s4 " << s4 << std::endl;

            return (Eigen::Matrix<Scalar, 3, 4>() <<
                    -4*s2*powi(q.x(),2)+s4-s1*s2,                 -4*q.x()*q.y()*s2,              -4*q.x()*q.z()*s2, q.x()*s3,
                               -4*q.x()*q.y()*s2,  -4*s2*powi(q.y(),2) + s4 - s1*s2,              -4*q.y()*q.z()*s2, q.y()*s3,
                               -4*q.x()*q.z()*s2,                 -4*q.y()*q.z()*s2,   -4*s2*powi(q.z(),2)+s4-s1*s2, q.z()*s3
                   ).finished();

        }else{
            const Scalar s1 = vec_squarednorm;
            const Scalar s2 = 1.0/pow(s1,(3.0/2.0));
            const Scalar s3 = 1.0/sqrt(s1);
            const Scalar s4 = 1.0/sqrt(1.0 - powi(q.w(),2));
            const Scalar s5 = acos(q.w());
            const Scalar s6 = 2.0*s3*s5;
    //        const Scalar s7 = s3*s4;
    //        const Scalar s8 = s2*s5;

            // std::cout << " s1 " << s1 << " s2 " << s2 << " s3 " << s3 << " s4 " << s4 << " s5 " << s5 << " s6 " << s6 << std::endl;

            return (Eigen::Matrix<Scalar, 3, 4>() <<
                     - 2*s2*s5*powi(q.x(),2) + s6,      -2*q.x()*q.y()*s2*s5,      -2*q.x()*q.z()*s2*s5, -2*q.x()*s3*s4,
                          -2*q.x()*q.y()*s2*s5, - 2*s2*s5*powi(q.y(),2) + s6,      -2*q.y()*q.z()*s2*s5, -2*q.y()*s3*s4,
                          -2*q.x()*q.z()*s2*s5,      -2*q.y()*q.z()*s2*s5, - 2*s2*s5*powi(q.z(),2) + s6, -2*q.z()*s3*s4
                   ).finished();
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    template<typename Scalar=double>
    inline std::vector<Eigen::Matrix<Scalar,3,3> > dLog_dR(const Eigen::Matrix<Scalar,3,3> R)
    {
        std::vector<Eigen::Matrix<Scalar,3,3> > vRes(3);
        const Scalar s1 = R(0)/2 + R(4)/2 + R(8)/2 - 0.5;
        const Scalar s2 = - (R(5) - R(7))/(4*(powi(s1,2) - 1)) - (s1*acos(s1)*(R(5) - R(7)))/(4*pow(1 - powi(s1,2),3.0/2.0));
        const Scalar s3 = acos(s1)/(2*sqrt(1 - powi(s1,2)));
        vRes[0] << s2, 0, 0, 0, s2, s3, 0, -s3, s2;

        const Scalar s4 = s1; // r0/2 + R(4)/2 + R(8)/2 - 1/2
        const Scalar s5 = (R(2) - R(6))/(4*(powi(s4,2) - 1)) + (s4*acos(s4)*(R(2) - R(6)))/(4*pow(1 - powi(s4,2),3.0/2.0));
        const Scalar s6 = ( 1/sqrt(1 - powi(s4,2)) )*acos(s4)*0.5;
        vRes[1] << s5, 0, -s6, 0, s5, 0, s6, 0, s5;

        const Scalar s7 = s1; // r0/2 + R(4)/2 + R(8)/2 - 1/2;
        const Scalar s8 = -(R(1) - R(3))/(4*(powi(s7,2) - 1)) - (s7*acos(s7)*(R(1) - R(3)))/(4*pow(1 - powi(s7,2),3.0/2.0));
        const Scalar s9 = acos(s7)/(2*sqrt(1 - powi(s7,2)));
        vRes[2] << s8, s9, 0, -s9, s8, 0, 0, 0, s8;

        return vRes;
    }

    ///////////////////////////////////////////////////////////////////////////////
    template<typename Scalar=double>
    inline Eigen::Matrix<Scalar,4,3> dqExp_dw(const Eigen::Matrix<Scalar,3,1>& w)
    {
        const Scalar t = w.norm();
        const Scalar s1 = t/20 - 1;
        const Scalar s2 = powi(t,2)/48 - 0.5;
        const Scalar s3 = (s1*w[1]*w[2])/24;
        const Scalar s4 = (s1*w[0]*w[2])/24;
        const Scalar s5 = (s1*w[0]*w[1])/24;
        const Scalar s6 = powi(t,2);
        return (Eigen::Matrix<Scalar, 4, 3>() <<
        (s1*powi(w[0],2))/24 - s6/48 + 0.5,                                 s5,                                  s4,
                                        s5, (s1*powi(w[1],2))/24 - s6/48 + 0.5,                                  s3,
                                        s4,                                 s3,  (s1*powi(w[2],2))/24 - s6/48 + 0.5,
                                 (s2*w[0])/2,                      (s2*w[1])/2,                         (s2*w[2])/2
             ).finished();

    }

    ///////////////////////////////////////////////////////////////////////////////
    template<typename Scalar=double>
    inline Eigen::Matrix<Scalar,4,4> dq1q2_dq2(const Eigen::Quaternion<Scalar>& q1)
    {
        return (Eigen::Matrix<Scalar, 4, 4>() <<
                 q1.w(), -q1.z(),  q1.y(), q1.x(),
                 q1.z(),  q1.w(), -q1.x(), q1.y(),
                -q1.y(),  q1.x(),  q1.w(), q1.z(),
                -q1.x(), -q1.y(), -q1.z(), q1.w()
             ).finished();
    }

    ///////////////////////////////////////////////////////////////////////////////
    template<typename Scalar=double>
    inline Eigen::Matrix<Scalar,4,4> dq1q2_dq1(const Eigen::Quaternion<Scalar>& q2)
    {
        return (Eigen::Matrix<Scalar, 4, 4>() <<
                 q2.w(),  q2.z(), -q2.y(), q2.x(),
                -q2.z(),  q2.w(),  q2.x(), q2.y(),
                 q2.y(), -q2.x(),  q2.w(), q2.z(),
                -q2.x(), -q2.y(), -q2.z(), q2.w()
             ).finished();
    }

    ///////////////////////////////////////////////////////////////////////////////
    template<typename Scalar=double>
    inline Eigen::Matrix<Scalar,3,4> dqx_dq(const Eigen::Quaternion<Scalar>& q, const Eigen::Matrix<Scalar,3,1>& vec)
    {
        const Scalar x = vec[0], y = vec[1], z = vec[2];
        const Scalar s1 = 2*q.x()*y;
        const Scalar s2 = 2*q.y()*y;
        const Scalar s3 = 2*q.x()*x;
        const Scalar s4 = 2*q.z()*x;
        const Scalar s5 = 2*q.y()*z;
        const Scalar s6 = 2*q.z()*z;

        return (Eigen::Matrix<Scalar, 3, 4>() <<
                                  s2 + s6,     s1 - 4*q.y()*x + 2*q.w()*z, 2*q.x()*z - 2*q.w()*y - 4*q.z()*x, s5 - 2*q.z()*y,
                 2*q.y()*x - 4*q.x()*y - 2*q.w()*z,                  s3 + s6,     s5 + 2*q.w()*x - 4*q.z()*y, s4 - 2*q.x()*z,
                     s4 + 2*q.w()*y - 4*q.x()*z, 2*q.z()*y - 2*q.w()*x - 4*q.y()*z,                  s2 + s3, s1 - 2*q.y()*x
             ).finished();
    }

    ///////////////////////////////////////////////////////////////////////////////
    template<typename Scalar=double>
    inline Eigen::Matrix<Scalar,6,1> log_decoupled(const Sophus::SE3Group<Scalar>& A,const Sophus::SE3Group<Scalar>& B)
    {
        Eigen::Matrix<Scalar,6,1> res;
        res.template head<3>() = A.translation()-B.translation();
        res.template tail<3>() = (A.so3()*B.so3().inverse()).log();
        return res;
    }

    ///////////////////////////////////////////////////////////////////////////////
    template<typename Scalar=double>
    inline Sophus::SE3Group<Scalar> exp_decoupled(const Sophus::SE3Group<Scalar>& A,const Eigen::Matrix<Scalar,6,1> x)
    {
        // Sophus::SO3Group<Scalar> Aso3 = A.so3();
        // Aso3.fastMultiply(Sophus::SO3Group<Scalar>::exp(x.template tail<3>()));
        // return Sophus::SE3Group<Scalar>(Aso3,A.translation() + x.template head<3>());
        return Sophus::SE3Group<Scalar>(A.so3()*Sophus::SO3Group<Scalar>::exp(x.template tail<3>()),A.translation() + x.template head<3>());
    }

    ///////////////////////////////////////////////////////////////////////////////
    // this function implements d vee(log(A * exp(x) * B) ) / dx , which is in R^{6x6}
    template<typename Scalar=double>
    inline Eigen::Matrix<Scalar,6,6> dLog_decoupled_dX(const Sophus::SE3Group<Scalar>& A,const Sophus::SE3Group<Scalar>& B)
    {
        Sophus::SO3Group<Scalar> Bso3inv = B.so3().inverse();

        const Eigen::Matrix<Scalar,6,1> d_2 = Sophus::SE3Group<Scalar>::log(A*B.inverse())/2;
        const Scalar d1 = d_2[3], d2 = d_2[4], d3 = d_2[5], dx = d_2[0], dy = d_2[1], dz = d_2[2];
        // this is using the 2nd order cbh expansion, to evaluate (I + 0.5 [Adj*x, log(AB)])*Adj
        // refer to the thesis by Hauke Strasdat, Appendix 3.
        const Eigen::Matrix<Scalar, 6, 6> dLog ((Eigen::Matrix<Scalar, 6, 6>() <<
                  1,  d3, -d2,   0,  dz, -dy,
                -d3,   1,  d1, -dz,   0,  dx,
                 d2, -d1,   1,  dy, -dx,   0,
                  0,   0,   0,   1,  d3, -d2,
                  0,   0,   0, -d3,   1,  d1,
                  0,   0,   0,  d2, -d1,   1
               ).finished() * A.Adj());
        Eigen::Matrix<Scalar,6,6> dLog_decoupled = Eigen::Matrix<Scalar,6,6>::Identity();
        dLog_decoupled.template block<3,3>(3,3) = dLog.template block<3,3>(3,3);
        // dLog_decoupled.template block<3,3>(3,3) = dLog_dq(A.unit_quaternion()*Bso3inv.unit_quaternion()) * dq1q2_dq1(Bso3inv.unit_quaternion()) * dqExp_dw<Scalar>(Eigen::Matrix<Scalar,3,1>::Zero());
        return dLog_decoupled;
    }

//    ///////////////////////////////////////////////////////////////////////////////
//    inline Eigen::Matrix<double,3,4> dqx_dq(const Eigen::Quaterniond& q, const Eigen::Vector3d& vec)
//    {
//        const double x = vec[0], y = vec[1], z = vec[2];
//        const double s1 = 2*q.y()*y;
//        const double s2 = 2*q.x()*x;
//        const double s3 = 2*q.z()*y;
//        const double s4 = 2*q.y()*x;
//        const double s5 = 2*q.z()*z;
//        const double s6 = 2*q.x()*z;

//        return (Eigen::Matrix<double, 3, 4>() <<
//                s3 - 2*q.y()*z,                         s1 + s5,    2*q.x()*y - 4*q.y()*x - 2*q.w()*z,          s6 - 4*q.z()*x + 2*q.w()*y,
//                s6 - 2*q.z()*x,        s4-4*q.x()*y + 2*q.w()*z,                              s2 + s5,   2*q.y()*z - 4*q.z()*y - 2*q.w()*x,
//                s4 - 2*q.x()*y, 2*q.z()*x-2*q.w()*y - 4*q.x()*z,           s3 + 2*q.w()*x - 4*q.y()*z,                              s1 + s2
//             ).finished();
//    }
}
