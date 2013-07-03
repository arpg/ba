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
}
