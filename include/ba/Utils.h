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
    inline Eigen::Vector4d MultHomogeneous( const Sophus::SE3d& lhs, const Eigen::Vector4d& rhs )
    {
        Eigen::Vector4d out;
        out.head<3>() = lhs.so3() * (Eigen::Vector3d)rhs.head<3>() + lhs.translation()*rhs[3];
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
    inline double powi(const double x, const int y)
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
    inline Eigen::Matrix<double,6,6> dLog_dX(const Sophus::SE3d& A,const Sophus::SE3d& B)
    {
        const Eigen::Vector6d d_2 = Sophus::SE3d::log(A*B)/2;
        const double d1 = d_2[3], d2 = d_2[4], d3 = d_2[5], dx = d_2[0], dy = d_2[1], dz = d_2[2];
        // this is using the 2nd order cbh expansion, to evaluate (I + 0.5 [Adj*x, log(AB)])*Adj
        // refer to the thesis by Hauke Strasdat, Appendix 3.
        return (Eigen::Matrix<double, 6, 6>() <<
                  1,  d3, -d2,   0,  dz, -dy,
                -d3,   1,  d1, -dz,   0,  dx,
                 d2, -d1,   1,  dy, -dx,   0,
                  0,   0,   0,   1,  d3, -d2,
                  0,   0,   0, -d3,   1,  d1,
                  0,   0,   0,  d2, -d1,   1
               ).finished() * A.Adj();
    }

    ///////////////////////////////////////////////////////////////////////////////
    // this function implements the derivative of log with respect to the input jacobian
    inline Eigen::Matrix<double,3,4> dLog_dq(const Eigen::Quaterniond& q)
    {
        const double s1 = powi(q.x(),2) + powi(q.y(),2) + powi(q.z(),2);
        const double s2 = 1/pow(s1,(3.0/2.0));
        const double s3 = 1/sqrt(s1);
        const double s4 = 1.0/sqrt(1 - powi(q.w(),2));
        const double s5 = acos(q.w());
        const double s6 = 2*s3*s5;
//        const double s7 = s3*s4;
//        const double s8 = s2*s5;

        // this is using the 2nd order cbh expansion, to evaluate (I + 0.5 [Adj*x, log(AB)])*Adj
        // refer to the thesis by Hauke Strasdat, Appendix 3.
        return (Eigen::Matrix<double, 3, 4>() <<
                 - 2*s2*s5*powi(q.x(),2) + s6,      -2*q.x()*q.y()*s2*s5,      -2*q.x()*q.z()*s2*s5, -2*q.x()*s3*s4,
                      -2*q.x()*q.y()*s2*s5, - 2*s2*s5*powi(q.y(),2) + s6,      -2*q.y()*q.z()*s2*s5, -2*q.y()*s3*s4,
                      -2*q.x()*q.z()*s2*s5,      -2*q.y()*q.z()*s2*s5, - 2*s2*s5*powi(q.z(),2) + s6, -2*q.z()*s3*s4
               ).finished();
    }

    ///////////////////////////////////////////////////////////////////////////////
    inline std::vector<Eigen::Matrix<double,3,3> > dLog_dR(const Eigen::Matrix3d R)
    {
        std::vector<Eigen::Matrix<double,3,3> > vRes(3);
        const double s1 = R(0)/2 + R(4)/2 + R(8)/2 - 0.5;
        const double s2 = - (R(5) - R(7))/(4*(powi(s1,2) - 1)) - (s1*acos(s1)*(R(5) - R(7)))/(4*pow(1 - powi(s1,2),3.0/2.0));
        const double s3 = acos(s1)/(2*sqrt(1 - powi(s1,2)));
        vRes[0] << s2, 0, 0, 0, s2, s3, 0, -s3, s2;

        const double s4 = s1; // r0/2 + R(4)/2 + R(8)/2 - 1/2
        const double s5 = (R(2) - R(6))/(4*(powi(s4,2) - 1)) + (s4*acos(s4)*(R(2) - R(6)))/(4*pow(1 - powi(s4,2),3.0/2.0));
        const double s6 = ( 1/sqrt(1 - powi(s4,2)) )*acos(s4)*0.5;
        vRes[1] << s5, 0, -s6, 0, s5, 0, s6, 0, s5;

        const double s7 = s1; // r0/2 + R(4)/2 + R(8)/2 - 1/2;
        const double s8 = -(R(1) - R(3))/(4*(powi(s7,2) - 1)) - (s7*acos(s7)*(R(1) - R(3)))/(4*pow(1 - powi(s7,2),3.0/2.0));
        const double s9 = acos(s7)/(2*sqrt(1 - powi(s7,2)));
        vRes[2] << s8, s9, 0, -s9, s8, 0, 0, 0, s8;

        return vRes;
    }

    ///////////////////////////////////////////////////////////////////////////////
    inline Eigen::Matrix<double,4,3> dqExp_dw(const Eigen::Vector3d& w)
    {
        const double t = w.norm();
        const double s1 = t/20 - 1;
        const double s2 = powi(t,2)/48 - 0.5;
        const double s3 = (s1*w[1]*w[2])/24;
        const double s4 = (s1*w[0]*w[2])/24;
        const double s5 = (s1*w[0]*w[1])/24;
        const double s6 = powi(t,2);
        return (Eigen::Matrix<double, 4, 3>() <<
        (s1*powi(w[0],2))/24 - s6/48 + 0.5,                                 s5,                                  s4,
                                        s5, (s1*powi(w[1],2))/24 - s6/48 + 0.5,                                  s3,
                                        s4,                                 s3,  (s1*powi(w[2],2))/24 - s6/48 + 0.5,
                                 (s2*w[0])/2,                      (s2*w[1])/2,                         (s2*w[2])/2
             ).finished();

    }

    ///////////////////////////////////////////////////////////////////////////////
    inline Eigen::Matrix<double,4,4> dq1q2_dq2(const Eigen::Quaterniond& q1)
    {
        return (Eigen::Matrix<double, 4, 4>() <<
                 q1.w(), -q1.z(),  q1.y(), q1.x(),
                 q1.z(),  q1.w(), -q1.x(), q1.y(),
                -q1.y(),  q1.x(),  q1.w(), q1.z(),
                -q1.x(), -q1.y(), -q1.z(), q1.w()
             ).finished();
    }

    ///////////////////////////////////////////////////////////////////////////////
    inline Eigen::Matrix<double,4,4> dq1q2_dq1(const Eigen::Quaterniond& q2)
    {
        return (Eigen::Matrix<double, 4, 4>() <<
                 q2.w(),  q2.z(), -q2.y(), q2.x(),
                -q2.z(),  q2.w(),  q2.x(), q2.y(),
                 q2.y(), -q2.x(),  q2.w(), q2.z(),
                -q2.x(), -q2.y(), -q2.z(), q2.w()
             ).finished();
    }

    ///////////////////////////////////////////////////////////////////////////////
    inline Eigen::Matrix<double,3,4> dqx_dq(const Eigen::Quaterniond& q, const Eigen::Vector3d& vec)
    {
        const double x = vec[0], y = vec[1], z = vec[2];
        const double s1 = 2*q.x()*y;
        const double s2 = 2*q.y()*y;
        const double s3 = 2*q.x()*x;
        const double s4 = 2*q.z()*x;
        const double s5 = 2*q.y()*z;
        const double s6 = 2*q.z()*z;

        return (Eigen::Matrix<double, 3, 4>() <<
                                  s2 + s6,     s1 - 4*q.y()*x + 2*q.w()*z, 2*q.x()*z - 2*q.w()*y - 4*q.z()*x, s5 - 2*q.z()*y,
                 2*q.y()*x - 4*q.x()*y - 2*q.w()*z,                  s3 + s6,     s5 + 2*q.w()*x - 4*q.z()*y, s4 - 2*q.x()*z,
                     s4 + 2*q.w()*y - 4*q.x()*z, 2*q.z()*y - 2*q.w()*x - 4*q.y()*z,                  s2 + s3, s1 - 2*q.y()*x
             ).finished();
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
