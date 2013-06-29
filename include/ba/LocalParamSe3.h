// Copyright (c) George Washington University, all rights reserved.  See the
// accompanying LICENSE file for more information.

#ifndef LOCAL_PARAM_SE3_H
#define LOCAL_PARAM_SE3_H

#include <sophus/se3.hpp>

class LocalParamSe3 : public ceres::LocalParameterization {
 public:
  virtual ~LocalParamSe3() {}
  virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const
  {
        const Eigen::Map<const Sophus::SE3d> T(x);
        const Eigen::Map<const Eigen::Matrix<double,6,1> > dx(delta);
        Eigen::Map<Sophus::SE3d> Tdx(x_plus_delta);
        Tdx = T * Sophus::SE3d::exp(dx);
        return true;
  }

  virtual bool ComputeJacobian(const double* x, double* jacobian) const
  {

        // Largely zeroes.
        memset(jacobian,0, sizeof(double)*7*6);

        /* Explicit formulation. Needs to be optimized */
        const double q1	   = x[0];
        const double q2	   = x[1];
        const double q3	   = x[2];
        const double q0	   = x[3]; //w

        const double half_q0 = 0.5*q0;
        const double half_q1 = 0.5*q1;
        const double half_q2 = 0.5*q2;
        const double half_q3 = 0.5*q3;

        const double q1_sq = q1*q1;
        const double q2_sq = q2*q2;
        const double q3_sq = q3*q3;

        // d output_quaternion / d update
        jacobian[3] = half_q0;
        jacobian[4] = -half_q3;
        jacobian[5] = half_q2;

        jacobian[9] = half_q3;
        jacobian[10] = half_q0;
        jacobian[11] = -half_q1;

        jacobian[15] = -half_q2;
        jacobian[16] = half_q1;
        jacobian[17] = half_q0;

        jacobian[21] = -half_q1;
        jacobian[22] = -half_q2;
        jacobian[23] = -half_q3;

        // d output_translation / d update
        jacobian[24]  = 1.0 - 2.0 * (q2_sq + q3_sq);
        jacobian[25]  = 2.0 * (q1*q2 - q0*q3);
        jacobian[26]  = 2.0 * (q1*q3 + q0*q2);

        jacobian[30]  = 2.0 * (q1*q2 + q0*q3);
        jacobian[31]  = 1.0 - 2.0 * (q1_sq + q3_sq);
        jacobian[32]  = 2.0 * (q2*q3 - q0*q1);

        jacobian[36] = 2.0 * (q1*q3 - q0*q2) ;
        jacobian[37] = 2.0 * (q2*q3 + q0*q1);
        jacobian[38] = 1.0 - 2.0 * (q1_sq + q2_sq);

        return true;
  }

  virtual int GlobalSize() const { return 7; }
  virtual int LocalSize() const { return 6; }

};
  
#endif
