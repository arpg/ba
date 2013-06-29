#pragma once

///////////////////////////////////////////////////////////////////////////////
namespace Sophus
{
    inline Eigen::Vector4d MultHomogeneous( const Sophus::SE3d& lhs, const Eigen::Vector4d& rhs )
    {
        Eigen::Vector4d out;
        out.head<3>() = lhs.so3() * (Eigen::Vector3d)rhs.head<3>() + lhs.translation()*rhs[3];
        out[3] = rhs[3];
        return out;
    }
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
