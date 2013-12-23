/*
 This file is part of the BA Project.

 Copyright (C) 2013 George Washington University,
 Nima Keivan,
 Gabe Sibley

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

#ifndef BA_UTILS_H
#define BA_UTILS_H

#include <Eigen/Eigen>
#include <sophus/se3.hpp>
#include <sys/time.h>
#include <time.h>

namespace Eigen {
typedef Matrix<double, 6, 1> Vector6d;
typedef Matrix<double, 9, 1> Vector9d;
}

///////////////////////////////////////////////////////////////////////////////
namespace ba {
// #define ENABLE_TIMING
extern int debug_level_threshold;
extern int debug_level;

/** @todo Add stream message to Android logging */
#ifndef StreamMessage
#  define StreamMessage(error_level)                                    \
  if (((int)error_level) < ba::debug_level_threshold) std::cerr
#endif

#ifdef ENABLE_TESTING
#define BA_TEST(x)  assert(x)
#else
#define BA_TEST(x)
#endif

#define ForceStartTimer(x) double x = ba::Tic()
#define ForcePrintTimer(x) std::cout << ba::Toc(x) << " seconds -> " <<  \
                                        #x << std::endl
#ifdef ENABLE_TIMING
#define StartTimer(x) double x = ba::Tic()
#define PrintTimer(x) std::cout << ba::Toc(x) << " seconds -> " <<  \
  #x << std::endl
#else
#define StartTimer(x)
#define PrintTimer(x)
#endif

static Eigen::IOFormat kCleanFmt(4, 0, ", ", ";\n", "", "");
static Eigen::IOFormat kLongFmt(Eigen::FullPrecision, 0, ", ", ";\n", "", "");
static Eigen::IOFormat kLongCsvFmt(Eigen::FullPrecision, 0, ", ", "\n", "", "");

#define NORM_THRESHOLD 1e-3
#define TESTING_EPS 1e-9

template<typename Scalar = double>
inline Eigen::Matrix<Scalar, 4, 1> MultHomogeneous(
    const Sophus::SE3Group<Scalar>& lhs,
    const Eigen::Matrix<Scalar, 4, 1>& rhs) {
  Eigen::Matrix<Scalar, 4, 1> out;
  out.template head<3>() = lhs.so3()
      * (Eigen::Matrix<Scalar, 3, 1>) rhs.template head<3>()
      + lhs.translation() * rhs[3];

  out[3] = rhs[3];
  return out;
}

///////////////////////////////////////////////////////////////////////////////
template<typename Scalar = double>
inline Scalar powi(const Scalar x, const int y) {
  if (y == 0) {
    return 1.0;
  } else if (y < 0) {
    return 1.0 / powi(x, -y);
  } else if (y == 0) {
    return 1.0;
  } else {
    Scalar ret = x;
    for (int ii = 1; ii < y; ii++) {
      ret *= x;
    }
    return ret;
  }
}

inline double Tic() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec + 1e-6 * (tv.tv_usec);
}

inline double Toc(double tic) {
  return Tic() - tic;
}

///////////////////////////////////////////////////////////////////////////////
// this function implements d vee(log(A * exp(x) * B) ) / dx,which is in R^{6x6}
template<typename Scalar = double>
inline Eigen::Matrix<Scalar, 6, 6> dlog_dx(const Sophus::SE3Group<Scalar>& a,
                                           const Sophus::SE3Group<Scalar>& b) {
  const Eigen::Matrix<Scalar, 6, 1> d_2 = Sophus::SE3Group < Scalar
      > ::log(a * b) / 2;
  const Scalar d1 = d_2[3], d2 = d_2[4], d3 = d_2[5], dx = d_2[0], dy = d_2[1],
      dz = d_2[2];
  // this is using the 2nd order cbh expansion, to evaluate
  // (I + 0.5 [Adj*x, log(AB)])*Adj
  // refer to the thesis by Hauke Strasdat, Appendix 3.
  return (Eigen::Matrix<Scalar, 6, 6>() <<
          1, d3, -d2, 0, dz, -dy,
          -d3, 1, d1, -dz, 0, dx,
          d2, -d1, 1, dy, -dx, 0,
          0, 0, 0, 1, d3, -d2,
          0, 0, 0, -d3, 1, d1,
          0, 0, 0, d2, -d1, 1)
      .finished() * a.Adj();
}

///////////////////////////////////////////////////////////////////////////////
// this function implements the derivative of log with respect to the input q
template<typename Scalar = double>
inline Eigen::Matrix<Scalar, 3, 4> dlog_dq(const Eigen::Quaternion<Scalar>& q) {
  const Scalar x = q.x();
  const Scalar y = q.y();
  const Scalar z = q.z();
  const Scalar w = q.w();
  const Scalar vec_squarednorm = powi(x, 2) + powi(y, 2) + powi(z, 2);
  const Scalar vec_norm = sqrt(vec_squarednorm);
  if (vec_norm < 1e-9) {
    StreamMessage(debug_level) << "Vec norm less than 1e-9: " <<
                                  q.coeffs().transpose() << std::endl;
    const Scalar s1 = 2 * vec_squarednorm;
    const Scalar s2 = 1.0 / powi(w, 3);
    const Scalar s3 = (3 * s1) / powi(w, 4) - 2 / powi(w, 2);
    const Scalar s4 = 2 / w;

    return (Eigen::Matrix<Scalar, 3, 4>()
        << -4 * s2 * powi(x, 2) + s4 - s1 * s2,
           -4 * x * y * s2,
           -4 * x * z * s2,
            x * s3,
           -4 * x * y * s2,
           -4 * s2 * powi(y, 2) + s4 - s1 * s2,
           -4 * y * z * s2,
            y * s3,
           -4 * x * z * s2,
           -4 * y * z * s2,
           -4 * s2 * powi(z, 2) + s4 - s1 * s2,
            z * s3).finished();

  } else {
    const Scalar s1 = vec_squarednorm;
    const Scalar s2 = 1 / (s1 / powi(w, 2) + 1);
    const Scalar s3 = atan(sqrt(s1) / w);
    const Scalar s4 = 1 / pow(s1, (3.0 / 2.0));
    const Scalar s5 = 1 / s1;
    const Scalar s6 = 1 / w;
    const Scalar s7 = (2 * s3) / sqrt(s1);
    const Scalar s8 = 2 * y * z * s2 * s5 * s6 - 2 * y * z * s3 * s4;
    const Scalar s9 = 2 * x * z * s2 * s5 * s6 - 2 * x * z * s3 * s4;
    const Scalar s10 = 2 * x * y * s2 * s5 * s6 - 2 * x * y * s3 * s4;

    return (Eigen::Matrix<Scalar, 3, 4>()
        << s7 - 2 * powi(x, 2) * s3 * s4 + 2 * powi(x, 2) * s2 * s5 * s6, s10,
            s9, -(2* x * s2) / powi(w, 2), s10, s7 - 2 * powi(y, 2) * s3 * s4
        + 2 * powi(y, 2) * s2 * s5 * s6, s8, -(2 * y * s2) / powi(w, 2), s9, s8,
            s7 - 2 * powi(z, 2) * s3 * s4 + 2 * powi(z, 2) * s2 * s5 * s6,
            -(2 * z * s2) / powi(w, 2)).finished();
  }
}

//////////////////////////////////////////////////////////////////////////////
template<typename Scalar>
static bool _Test_dlog_dq(const Eigen::Quaternion<Scalar>& q) {
  Scalar dEps = 1e-9;
  std::cout << "q:" << q.coeffs().transpose() << std::endl;
  Eigen::Matrix<Scalar, 3, 4> dLog_dq_fd;
  for (int ii = 0; ii < 4; ii++) {
    Eigen::Matrix<Scalar, 4, 1> eps;
    eps.setZero();
    eps[ii] += dEps;
    Eigen::Quaternion<Scalar> q_plus = q;
    q_plus.coeffs() += eps;
    Sophus::SO3Group<Scalar> so3_plus;
    memcpy(so3_plus.data(), q_plus.coeffs().data(), sizeof(Scalar) * 4);
    Eigen::Matrix<Scalar, 3, 1> res_plus = so3_plus.log();

    eps[ii] -= 2 * dEps;
    Eigen::Quaternion<Scalar> q_minus = q;
    q_minus.coeffs() += eps;
    Sophus::SO3Group<Scalar> so3_minus;
    memcpy(so3_minus.data(), q_minus.coeffs().data(), sizeof(Scalar) * 4);
    Eigen::Matrix<Scalar, 3, 1> res_minus = so3_minus.log();

    dLog_dq_fd.col(ii) = (res_plus - res_minus) / (2 * dEps);
  }
  const Eigen::Matrix<Scalar, 3, 4> dlog = dlog_dq(q);
  std::cout << "dlog_dq = [" << dlog.format(kCleanFmt) << "]" << std::endl;
  std::cout << "dlog_dqf = [" << dLog_dq_fd.format(kCleanFmt) << "]"
      << std::endl;
  std::cout << "dlog_dq - dlog_dqf = [" << (dlog - dLog_dq_fd).format(kCleanFmt)
      << "]" << std::endl;
  return (dlog - dLog_dq_fd).norm() < NORM_THRESHOLD;
}

///////////////////////////////////////////////////////////////////////////////
template<typename Scalar = double>
inline std::vector<Eigen::Matrix<Scalar, 3, 3> > dlog_dr(
    const Eigen::Matrix<Scalar, 3, 3> r) {
  std::vector < Eigen::Matrix<Scalar, 3, 3> > res(3);
  const Scalar s1 = r(0) / 2 + r(4) / 2 + r(8) / 2 - 0.5;
  const Scalar s2 = -(r(5) - r(7)) / (4 * (powi(s1, 2) - 1))
      - (s1 * acos(s1) * (r(5) - r(7))) / (4 * pow(1 - powi(s1, 2), 3.0 / 2.0));

  const Scalar s3 = acos(s1) / (2 * sqrt(1 - powi(s1, 2)));
  res[0] << s2, 0, 0, 0, s2, s3, 0, -s3, s2;

  const Scalar s4 = s1;  // r0/2 + R(4)/2 + R(8)/2 - 1/2
  const Scalar s5 = (r(2) - r(6)) / (4 * (powi(s4, 2) - 1))
      + (s4 * acos(s4) * (r(2) - r(6))) / (4 * pow(1 - powi(s4, 2), 3.0 / 2.0));

  const Scalar s6 = (1 / sqrt(1 - powi(s4, 2))) * acos(s4) * 0.5;
  res[1] << s5, 0, -s6, 0, s5, 0, s6, 0, s5;

  const Scalar s7 = s1;  // r0/2 + R(4)/2 + R(8)/2 - 1/2;
  const Scalar s8 = -(r(1) - r(3)) / (4 * (powi(s7, 2) - 1))
      - (s7 * acos(s7) * (r(1) - r(3))) / (4 * pow(1 - powi(s7, 2), 3.0 / 2.0));

  const Scalar s9 = acos(s7) / (2 * sqrt(1 - powi(s7, 2)));
  res[2] << s8, s9, 0, -s9, s8, 0, 0, 0, s8;

  return res;
}

///////////////////////////////////////////////////////////////////////////////
template<typename Scalar = double>
inline Eigen::Matrix<Scalar, 4, 3> dq_exp_dw(
    const Eigen::Matrix<Scalar, 3, 1>& w) {
  const Scalar t = w.norm();
  const Scalar s1 = t / 20 - 1;
  const Scalar s2 = powi(t, 2) / 48 - 0.5;
  const Scalar s3 = (s1 * w[1] * w[2]) / 24;
  const Scalar s4 = (s1 * w[0] * w[2]) / 24;
  const Scalar s5 = (s1 * w[0] * w[1]) / 24;
  const Scalar s6 = powi(t, 2);
  return (Eigen::Matrix<Scalar, 4, 3>()
      << (s1 * powi(w[0], 2)) / 24 - s6 / 48 + 0.5, s5, s4, s5, (s1
      * powi(w[1], 2)) / 24 - s6 / 48 + 0.5, s3, s4, s3, (s1 * powi(w[2], 2))
      / 24 - s6 / 48 + 0.5, (s2 * w[0]) / 2, (s2 * w[1]) / 2, (s2 * w[2]) / 2)
      .finished();
}

///////////////////////////////////////////////////////////////////////////////
template<typename Scalar = double>
inline Eigen::Matrix<Scalar, 4, 4> dqinv_dq() {
  return ((Eigen::Matrix<Scalar, 4, 1>() << -1, -1, -1, 1).finished())
      .asDiagonal();
}

///////////////////////////////////////////////////////////////////////////////
template<typename Scalar = double>
inline Eigen::Matrix<Scalar, 4, 4> dq1q2_dq2(
    const Eigen::Quaternion<Scalar>& q1) {
  return (Eigen::Matrix<Scalar, 4, 4>() << q1.w(), -q1.z(), q1.y(), q1.x(),
          q1.z(), q1.w(), -q1.x(), q1.y(), -q1.y(), q1.x(), q1.w(), q1.z(),
          -q1.x(), -q1.y(), -q1.z(), q1.w()).finished();
}

///////////////////////////////////////////////////////////////////////////////
template<typename Scalar = double>
inline Eigen::Matrix<Scalar, 4, 4> dq1q2_dq1(
    const Eigen::Quaternion<Scalar>& q2) {
  return (Eigen::Matrix<Scalar, 4, 4>() << q2.w(), q2.z(), -q2.y(), q2.x(),
          -q2.z(), q2.w(), q2.x(), q2.y(), q2.y(), -q2.x(), q2.w(), q2.z(),
          -q2.x(), -q2.y(), -q2.z(), q2.w()).finished();
}

///////////////////////////////////////////////////////////////////////////////
template<typename Scalar = double>
inline Eigen::Matrix<Scalar, 3, 4> dqx_dq(
    const Eigen::Quaternion<Scalar>& q,
    const Eigen::Matrix<Scalar, 3, 1>& vec) {
  const Scalar x = vec[0], y = vec[1], z = vec[2];
  const Scalar s1 = 2 * q.x() * y;
  const Scalar s2 = 2 * q.y() * y;
  const Scalar s3 = 2 * q.x() * x;
  const Scalar s4 = 2 * q.z() * x;
  const Scalar s5 = 2 * q.y() * z;
  const Scalar s6 = 2 * q.z() * z;

  return (Eigen::Matrix<Scalar, 3, 4>() << s2 + s6, s1 - 4 * q.y() * x
      + 2 * q.w() * z, 2 * q.x() * z - 2 * q.w() * y - 4 * q.z() * x, s5
      - 2 * q.z() * y, 2 * q.y() * x - 4 * q.x() * y - 2 * q.w() * z, s3 + s6,
      s5 + 2 * q.w() * x - 4 * q.z() * y, s4 - 2 * q.x() * z, s4 + 2 * q.w() * y
      - 4 * q.x() * z, 2 * q.z() * y - 2 * q.w() * x - 4 * q.y() * z, s2 + s3,
      s1 - 2 * q.y() * x).finished();
}

///////////////////////////////////////////////////////////////////////////////
template<typename Scalar = double>
inline Eigen::Matrix<Scalar, 3, 3> dqx_dx(
    const Eigen::Quaternion<Scalar>& q,
    const Eigen::Matrix<Scalar, 3, 1>& vec) {
  const Scalar x = vec[0], y = vec[1], z = vec[2];
  const Scalar s1 = 2 * q.y() * q.z();
  const Scalar s2 = 2 * q.x() * q.z();
  const Scalar s3 = 2 * q.x() * q.y();
  const Scalar s4 = powi(q.y(), 2);
  const Scalar s5 = powi(q.x(), 2);
  const Scalar s6 = powi(q.z(), 2);
  const Scalar s7 = 2 * q.w() * q.y();

  return (Eigen::Matrix<Scalar, 3, 3>() <<
             1 - 2*s6 - 2*s4,    s3 - 2*q.w()*q.z(),               s2 + s7,
          s3 + 2*q.w()*q.z(),       1 - 2*s6 - 2*s5,    s1 - 2*q.w()*q.x(),
                     s2 - s7,    s1 + 2*q.w()*q.x(),       1 - 2*s5 - 2*s4)
      .finished();
}

///////////////////////////////////////////////////////////////////////////////
template<typename Scalar = double>
inline Eigen::Matrix<Scalar, 7, 7> dt1t2_dt1(
    const Sophus::SE3Group<Scalar>& t1,
    const Sophus::SE3Group<Scalar>& t2) {

  Eigen::Matrix<Scalar,7,7> dt1t2_dt2;
  dt1t2_dt2.setZero();
  dt1t2_dt2.template block<3,3>(0,0).setIdentity();
  dt1t2_dt2.template block<3,4>(0,3) =
      dqx_dq(t1.unit_quaternion(), t2.translation());
  dt1t2_dt2.template block<4,4>(3,3) =
      dq1q2_dq1(t2.unit_quaternion());

  return dt1t2_dt2;
}

///////////////////////////////////////////////////////////////////////////////
template<typename Scalar = double>
inline Eigen::Matrix<Scalar, 6, 1> log_decoupled(
    const Sophus::SE3Group<Scalar>& a, const Sophus::SE3Group<Scalar>& b) {
  Eigen::Matrix<Scalar, 6, 1> res;
  res.template head<3>() = a.translation() - b.translation();
  res.template tail<3>() = (a.so3() * b.so3().inverse()).log();
  return res;
}

///////////////////////////////////////////////////////////////////////////////
template<typename Scalar = double>
inline Sophus::SE3Group<Scalar> exp_decoupled(
    const Sophus::SE3Group<Scalar>& a, const Eigen::Matrix<Scalar, 6, 1> x) {
  return Sophus::SE3Group < Scalar
      > (a.so3() * Sophus::SO3Group < Scalar > ::exp(x.template tail<3>()), a
          .translation() + x.template head<3>());
}

///////////////////////////////////////////////////////////////////////////////
// this function implements d vee(log(A * exp(x) * B) ) / dx,which is in R^{6x6}
template<typename Scalar = double>
inline Eigen::Matrix<Scalar, 6, 6> dlog_decoupled_dx(
    const Sophus::SE3Group<Scalar>& a, const Sophus::SE3Group<Scalar>& b) {
  Eigen::Matrix<Scalar, 6, 6> dLog_decoupled =
      Eigen::Matrix<Scalar, 6, 6>::Identity();
  dLog_decoupled.template block<3, 3>(3, 3) =
      dlog_dq((a*b.inverse()).unit_quaternion()) *
      dq1q2_dq2(a.unit_quaternion()) *
      dq1q2_dq1(b.inverse().unit_quaternion()) *
      dq_exp_dw<Scalar>(Eigen::Matrix<Scalar, 3, 1>::Zero());
  return dLog_decoupled;
}

///////////////////////////////////////////////////////////////////////////////
template<typename Scalar = double>
inline Eigen::Matrix<Scalar, 6, 7> dLog_decoupled_dt1(
    const Sophus::SE3Group<Scalar>& t1, const Sophus::SE3Group<Scalar>& t2) {
  Eigen::Matrix<Scalar, 6, 7> dLog_decoupled;
  dLog_decoupled.setZero();
  dLog_decoupled.template block<3, 3>(0, 0).setIdentity();
  dLog_decoupled.template block<3, 4>(3, 3) =
      dlog_dq((t1*t2.inverse()).unit_quaternion()) *
      dq1q2_dq1(t2.inverse().unit_quaternion());
  return dLog_decoupled;
}

///////////////////////////////////////////////////////////////////////////////
template<typename Scalar = double>
inline Eigen::Matrix<Scalar, 6, 7> dlog_decoupled_dt2(
    const Sophus::SE3Group<Scalar>& t1, const Sophus::SE3Group<Scalar>& t2) {
  Eigen::Matrix<Scalar, 6, 7> dlog_dt2;
  Eigen::Quaternion<Scalar> qlog =
      (t1.so3() * t2.so3().inverse()).unit_quaternion();
  dlog_dt2.setZero();
  dlog_dt2.template block<3,3>(0,0) = -Eigen::Matrix<Scalar, 3, 3>::Identity();
  dlog_dt2.template block<3,4>(3,3) =
      dlog_dq<Scalar>(qlog) *
      dq1q2_dq2<Scalar>(t1.unit_quaternion()) *
      dqinv_dq<Scalar>();

  // Check the dlog_db
  //{
  //  Eigen::Matrix<double, 6, 7>  dlog_dt2_fd;
  //  Scalar deps = 1e-6;
  //  for(int ii = 0; ii < 7 ; ++ii){
  //      Eigen::Matrix<double, 7, 1> eps_vec;
  //      eps_vec.setZero();
  //      eps_vec[ii] = deps;

  //      Sophus::SE3d t_plus = t2;
  //      t_plus.translation() += eps_vec.head<3>();
  //      Eigen::Quaterniond q_plus = t_plus.so3().unit_quaternion();
  //      q_plus.coeffs() += eps_vec.tail<4>();
  //      memcpy(t_plus.so3().data(),q_plus.coeffs().data(),4*sizeof(Scalar));

  //      Eigen::Matrix<double, 6, 1> y_plus =
  //          log_decoupled(t1, t_plus);

  //      eps_vec[ii] = -deps;
  //      Sophus::SE3d t_minus = t2;
  //      t_minus.translation() += eps_vec.head<3>();
  //      Eigen::Quaterniond q_minus = t_minus.so3().unit_quaternion();
  //      q_minus.coeffs() += eps_vec.tail<4>();
  //      memcpy(t_minus.so3().data(),q_minus.coeffs().data(),4*sizeof(Scalar));

  //      Eigen::Matrix<double, 6, 1> y_minus =
  //          log_decoupled(t1, t_minus);

  //      dlog_dt2_fd.col(ii) = (y_plus - y_minus) / (2 * deps);
  //  }
  //  std::cerr << "dlog_dt2:" << dlog_dt2 << std::endl;
  //  std::cerr << "dlog_dt2_fd:" << dlog_dt2_fd << std::endl;
  //}
  return dlog_dt2;
}

///////////////////////////////////////////////////////////////////////////////
template<typename Scalar = double>
inline Eigen::Matrix<Scalar, 7, 6> dexp_decoupled_dx(
    const Sophus::SE3Group<Scalar>& t) {
  Eigen::Matrix<Scalar, 7, 6> dexp;
  dexp.setZero();
  dexp.template block<3, 3>(0, 0).setIdentity();
  dexp.template block<4, 3>(3, 3) =
      dq1q2_dq2(t.unit_quaternion()) *
      dq_exp_dw<Scalar>(Eigen::Matrix<Scalar, 3, 1>::Zero());

  // Check the dExp
  /*{
    Eigen::Matrix<Scalar,7,6> dz_exp_fd;
    Scalar deps = 1e-6;
    for (int ii = 0; ii < 6 ; ii++) {
      Eigen::Matrix<Scalar,6,1> delta;
      delta.setZero();
      delta[ii] = deps;
      const SE3t se3_plus = exp_decoupled(t_w2, delta).inverse();
      Vector7t p_plus;
      p_plus.template head<3>() = se3_plus.translation();
      p_plus.template tail<4>() = se3_plus.unit_quaternion().coeffs();

      delta[ii] = -deps;
      const SE3t se3_minus = exp_decoupled(t_w2, delta).inverse();
      Vector7t p_minus;
      p_minus.template head<3>() = se3_minus.translation();
      p_minus.template tail<4>() = se3_minus.unit_quaternion().coeffs();

      dz_exp_fd.col(ii) = (p_plus - p_minus)/(2*deps);
    }

    std::cerr << "dz_exp:" <<
                 dInvExp_decoupled_dX<Scalar>(t_w2) <<
                 std::endl;
    std::cerr << "dz_exp_fd:" << dz_exp_fd << std::endl;
  }*/

  return dexp;
}

///////////////////////////////////////////////////////////////////////////////
template<typename Scalar = double>
inline Eigen::Matrix<Scalar, 7, 6> dinv_exp_decoupled_dx(
    const Sophus::SE3Group<Scalar>& t) {
  const Eigen::Matrix<Scalar, 4, 3> dq_exp =
      dq_exp_dw<Scalar>(Eigen::Matrix<Scalar, 3, 1>::Zero());
  const Eigen::Quaternion<Scalar> qt_inv = t.so3().inverse().unit_quaternion();
  const Eigen::Matrix<Scalar, 4, 4> dq1q2_dq_exp = dq1q2_dq1(qt_inv);

  Eigen::Matrix<Scalar, 7, 6> dexp;
  dexp.setZero();
  dexp.template block<3, 3>(0, 0) = -t.so3().inverse().matrix();
  dexp.template block<3, 3>(0, 3) =
      dqx_dq(qt_inv, t.translation()) * dq1q2_dq_exp * dq_exp;
  dexp.template block<4, 3>(3, 3) = dq1q2_dq_exp * -dq_exp;

  // Check the dExp
  //{
  //  Eigen::Matrix<Scalar,7,6> dz_exp_fd;
  //  Scalar deps = 1e-6;
  //  for (int ii = 0; ii < 6 ; ii++) {
  //    Eigen::Matrix<Scalar,6,1> delta;
  //    delta.setZero();
  //    delta[ii] = deps;
  //    const Sophus::SE3Group<Scalar> se3_plus =
  //        exp_decoupled(t, delta).inverse();
  //    Eigen::Matrix<Scalar, 7, 1> p_plus;
  //    p_plus.template head<3>() = se3_plus.translation();
  //    p_plus.template tail<4>() = se3_plus.unit_quaternion().coeffs();

  //    delta[ii] = -deps;
  //    const Sophus::SE3Group<Scalar> se3_minus =
  //        exp_decoupled(t, delta).inverse();
  //    Eigen::Matrix<Scalar, 7, 1> p_minus;
  //    p_minus.template head<3>() = se3_minus.translation();
  //    p_minus.template tail<4>() = se3_minus.unit_quaternion().coeffs();

  //    dz_exp_fd.col(ii) = (p_plus - p_minus)/(2*deps);
  //  }

  //  std::cerr << "dz_invexp:" << dexp << std::endl;
  //  std::cerr << "dz_invexp_fd:" << dz_exp_fd << std::endl;
  //}

  return dexp;
}

///////////////////////////////////////////////////////////////////////////////
template<typename Scalar = double>
inline Eigen::Matrix<Scalar, 4, 7> dt_x_dt(
    const Sophus::SE3Group<Scalar>& t, const Eigen::Matrix<Scalar, 4, 1>& x)
{
  Eigen::Matrix<Scalar, 4, 7> dt_x;
  dt_x.setZero();
  dt_x.template topLeftCorner<3, 3>() =
      Eigen::Matrix<Scalar, 3, 3>::Identity() * x[3];
  dt_x.template topRightCorner<3, 4>() = dqx_dq<Scalar>(t.unit_quaternion(),
                                                       x.template head<3>());

  // Check the dt1_t2_dt1
  //{
  //  Eigen::Matrix<double, 4, 7>  dt_x_fd;
  //  Scalar deps = 1e-6;
  //  for(int ii = 0; ii < 7 ; ++ii){
  //      Eigen::Matrix<double, 7, 1> eps_vec;
  //      eps_vec.setZero();
  //      eps_vec[ii] = deps;

  //      Sophus::SE3d t_plus = t;
  //      t_plus.translation() += eps_vec.head<3>();
  //      Eigen::Quaterniond q_plus = t_plus.so3().unit_quaternion();
  //      q_plus.coeffs() += eps_vec.tail<4>();
  //      memcpy(t_plus.so3().data(),q_plus.coeffs().data(),4*sizeof(Scalar));

  //      Eigen::Matrix<double, 4, 1> y_plus = t_plus.matrix() * x;

  //      eps_vec[ii] = -deps;
  //      Sophus::SE3d t_minus = t;
  //      t_minus.translation() += eps_vec.head<3>();
  //      Eigen::Quaterniond q_minus = t_minus.so3().unit_quaternion();
  //      q_minus.coeffs() += eps_vec.tail<4>();
  //      memcpy(t_minus.so3().data(),q_minus.coeffs().data(),4*sizeof(Scalar));

  //      Eigen::Matrix<double, 4, 1> y_minus = t_minus.matrix() * x;

  //      dt_x_fd.col(ii) = (y_plus - y_minus) / (2 * deps);
  //  }
  //  std::cerr << "dt_x:" << dt_x << std::endl;
  //  std::cerr << "dt_x_fd:" << dt_x_fd << std::endl;
  //}

  return dt_x;
}

///////////////////////////////////////////////////////////////////////////////
template<typename Scalar = double>
inline Eigen::Matrix<Scalar, 7, 7> dt1_t2_dt1(
    const Sophus::SE3Group<Scalar>& t1, const Sophus::SE3Group<Scalar>& t2)
{
  Eigen::Matrix<Scalar, 7, 7> dt1t2;
  dt1t2.setZero();
  // dt/dt1
  dt1t2.template topLeftCorner<3, 3>().setIdentity();
  // dt/dR1
  dt1t2.template topRightCorner<3, 4>() = dqx_dq(t1.unit_quaternion(),
                                                 t2.translation());
  // dR/dR1
  dt1t2.template bottomRightCorner<4, 4>() = dq1q2_dq1(t2.unit_quaternion());

  // Check the dt1_t2_dt1
  /*{
    Eigen::Matrix<double, 7, 7>  dt1_t2_dt1_fd;
    Scalar deps = 1e-6;
    for(int ii = 0; ii < 7 ; ++ii){
        Eigen::Matrix<double, 7, 1> eps_vec;
        eps_vec.setZero();
        eps_vec[ii] = deps;

        Sophus::SE3d t_plus = t_1w;
        t_plus.translation() += eps_vec.head<3>();
        Eigen::Quaterniond q_plus = t_plus.so3().unit_quaternion();
        q_plus.coeffs() += eps_vec.tail<4>();
        memcpy(t_plus.so3().data(),q_plus.coeffs().data(),4*sizeof(Scalar));

        Eigen::Matrix<double, 7, 1> y_plus;
        y_plus.template head<3>() = (t_plus * t_w2).translation();
        y_plus.template tail<4>() =
            (t_1w.unit_quaternion() * t_plus.unit_quaternion()).coeffs();

        eps_vec[ii] = -deps;
        Sophus::SE3d t_minus = t_1w;
        t_minus.translation() += eps_vec.head<3>();
        Eigen::Quaterniond q_minus = t_minus.so3().unit_quaternion();
        q_minus.coeffs() += eps_vec.tail<4>();
        memcpy(t_minus.so3().data(),q_minus.coeffs().data(),4*sizeof(Scalar));

        Eigen::Matrix<double, 7, 1> y_minus;
        y_minus.template head<3>() = (t_minus * t_w2).translation();
        y_minus.template tail<4>() =
            (t_1w.unit_quaternion() * t_minus.unit_quaternion()).coeffs();

        dt1_t2_dt1_fd.col(ii) = (y_plus - y_minus) / (2 * deps);
    }
    std::cerr << "dt1_t2_dt1:" << dt1_t2_dt1(t_1w, t_w2) << std::endl;
    std::cerr << "dt1_t2_dt1_fd:" << dt1_t2_dt1_fd << std::endl;
  }*/

  return dt1t2;
}

///////////////////////////////////////////////////////////////////////////////
template<typename Scalar = double>
inline Eigen::Matrix<Scalar, 7, 7> dt1_t2_dt2(
    const Sophus::SE3Group<Scalar>& t1, const Sophus::SE3Group<Scalar>& t2)
{
  Eigen::Matrix<Scalar, 7, 7> dt1t2;
  dt1t2.setZero();
  // dt/dt2
  dt1t2.template topLeftCorner<3, 3>() = t1.rotationMatrix();
  // dR/dR2
  dt1t2.template bottomRightCorner<4, 4>() = dq1q2_dq2(t1.unit_quaternion());

  /*
  // Check the dt1_t2_dt2
  {
    Eigen::Matrix<double, 7, 7>  dt1_t2_dt2_fd;
    Scalar deps = 1e-6;
    for(int ii = 0; ii < 7 ; ++ii){
        Eigen::Matrix<double, 7, 1> eps_vec;
        eps_vec.setZero();
        eps_vec[ii] = deps;

        Sophus::SE3d t_plus = t_w2;
        t_plus.translation() += eps_vec.head<3>();
        Eigen::Quaterniond q_plus = t_plus.so3().unit_quaternion();
        q_plus.coeffs() += eps_vec.tail<4>();
        memcpy(t_plus.so3().data(),q_plus.coeffs().data(),4*sizeof(Scalar));

        Eigen::Matrix<double, 7, 1> y_plus;
        y_plus.template head<3>() = (t_1w * t_plus).translation();
        y_plus.template tail<4>() =
            (t_1w.unit_quaternion() * t_plus.unit_quaternion()).coeffs();

        eps_vec[ii] = -deps;
        Sophus::SE3d t_minus = t_w2;
        t_minus.translation() += eps_vec.head<3>();
        Eigen::Quaterniond q_minus = t_minus.so3().unit_quaternion();
        q_minus.coeffs() += eps_vec.tail<4>();
        memcpy(t_minus.so3().data(),q_minus.coeffs().data(),4*sizeof(Scalar));

        Eigen::Matrix<double, 7, 1> y_minus;
        y_minus.template head<3>() = (t_1w * t_minus).translation();
        y_minus.template tail<4>() =
            (t_1w.unit_quaternion() * t_minus.unit_quaternion()).coeffs();

        dt1_t2_dt2_fd.col(ii) = (y_plus - y_minus) / (2 * deps);
    }
    std::cerr << "dt1_t2_dt2:" << dt1_t2_dt2(t_1w, t_w2) << std::endl;
    std::cerr << "dt1_t2_dt2_fd:" << dt1_t2_dt2_fd << std::endl;
  }
  */

  return dt1t2;
}


template<typename Scalar = double>
inline Eigen::Matrix<Scalar, 6, 7> dlog_dse3(Sophus::SE3Group<Scalar> t) {

  const Eigen::Matrix<Scalar, 3, 4> dw_dq = dlog_dq(t.unit_quaternion());

  const Scalar x = t.translation()[0];
  const Scalar y = t.translation()[1];
  const Scalar z = t.translation()[2];
  const Eigen::Matrix<Scalar, 3, 1> w = t.so3().log();
  const Scalar wx = w[0];
  const Scalar wy = w[1];
  const Scalar wz = w[2];

  Eigen::Matrix<Scalar, 3, 1> upsilon_omega;
  Scalar theta;
  upsilon_omega.template tail<3>() = Sophus::SO3Group < Scalar
      > ::logAndTheta(t.so3(), &theta);
  const Eigen::Matrix<Scalar, 3, 3> & omega = Sophus::SO3Group < Scalar
      > ::hat(upsilon_omega.template tail<3>());

  // avoid division by 0
  bool close_to_zero = abs(theta) < Sophus::SophusConstants<Scalar>::epsilon();

  const Eigen::Matrix<Scalar,3,3> & v_inv = close_to_zero?
      Eigen::Matrix<Scalar,3,3>::Identity() -
      static_cast<Scalar>(0.5) * omega
      + static_cast<Scalar>(1./12.) * (omega * omega)
     :
      (Eigen::Matrix<Scalar, 3, 3>::Identity()
      - static_cast<Scalar>(0.5) * omega
      + (static_cast<Scalar>(1)
      - theta / (static_cast<Scalar>(2) * tan(theta / Scalar(2))))
      / (theta * theta) * (omega * omega));


  Eigen::Matrix<Scalar, 6, 7> dlog;
  dlog.setZero();
  dlog.template block<3, 3>(0, 0) = v_inv;
  dlog.template block<3, 4>(3, 3) = dw_dq;

  if ( close_to_zero ) {
    const Scalar div_12 = 1./12;
    const Scalar div_6  = 1./6.;
    const Scalar wx_x   = wx*x;
    const Scalar wy_x   = wy*x;
    const Scalar wz_x   = wz*x;
    const Scalar wx_y   = wx*y;
    const Scalar wy_y   = wy*y;
    const Scalar wz_y   = wz*y;
    const Scalar wx_z   = wx*z;
    const Scalar wy_z   = wy*z;
    const Scalar wz_z   = wz*z;

    const Eigen::Matrix<Scalar, 3, 3> dlog_dw(
          (Eigen::Matrix<Scalar, 3, 3>()
           << div_12*(wy_y + wz_z),
              div_12*wx_y - div_6*wy_x - 0.5*z,
              0.5*y - div_6*wz_x + div_12*wx_z,
              0.5*z + div_12*wy_x - div_6*wx_y,
              div_12*(wx_x + wz_z),
              div_12*wy_z - div_6*wz_y - 0.5*x,
              div_12*wz_x - div_6*wx_z - 0.5*y,
              0.5*x + div_12*wz_y - div_6*wy_z,
              div_12*(wx_x*wy_y))
          .finished());

    dlog.template block<3, 4>(0, 3) = dlog_dw * dw_dq;

  } else {

    const Scalar s1 = powi(wx, 2) + powi(wy, 2) + powi(wz, 2);
    const Scalar s2 = tan(sqrt(s1) / 2);
    const Scalar s3 = sqrt(s1) / (2 * s2) - 1;
    const Scalar s4 = wz / (2 * sqrt(s1) * s2)
        - (wz * (powi(s2, 2) + 1)) / (4 * powi(s2, 2));
    const Scalar s5 = wy / (2 * sqrt(s1) * s2)
        - (wy * (powi(s2, 2) + 1)) / (4 * powi(s2, 2));
    const Scalar s6 = wx / (2 * sqrt(s1) * s2)
        - (wx * (powi(s2, 2) + 1)) / (4 * powi(s2, 2));
    const Scalar s7 = 1 / s1;
    const Scalar s8 = 1 / powi(s1, 2);
    const Scalar s9 = powi(wx, 2) + powi(wy, 2);
    const Scalar s10 = powi(wx, 2) + powi(wz, 2);
    const Scalar s11 = powi(wy, 2) + powi(wz, 2);
    const Scalar s12 = 2 * s3 * s8 * wx * wy * wz;
    const Scalar s13 = -2 * s3 * s8 * wy * powi(wz, 2) + s4 * s7 * wy * wz
        + s3 * s7 * wy;
    const Scalar s14 = -2 * s3 * s8 * wx * powi(wz, 2) + s4 * s7 * wx * wz
        + s3 * s7 * wx;
    const Scalar s15 = -2 * s3 * s8 * wz * powi(wy, 2) + s5 * s7 * wz * wy
        + s3 * s7 * wz;
    const Scalar s16 = -2 * s3 * s8 * wz * powi(wx, 2) + s6 * s7 * wz * wx
        + s3 * s7 * wz;
    const Scalar s17 = -2 * s3 * s8 * wx * powi(wy, 2) + s5 * s7 * wx * wy
        + s3 * s7 * wx;
    const Scalar s18 = -2 * s3 * s8 * wy * powi(wx, 2) + s6 * s7 * wy * wx
        + s3 * s7 * wy;
    const Scalar s19 = 2 * s3 * s7 * wy;
    const Scalar s20 = 2 * s3 * s7 * wx;

    const Eigen::Matrix<Scalar, 3, 3> dlog_dw(
        (Eigen::Matrix<Scalar, 3, 3>()
        << x * (s6 * s7 * s11 - 2 * s3 * s8 * s11 * wx) - s18 * y - s16 * z, x
            * (s19 + s5 * s7 * s11 - 2 * s3 * s8 * s11 * wy) - s17 * y
            - z * (s5 * s7 * wx * wz - 2 * s3 * s8 * wx * wy * wz + 0.5),

       x * (s4 * s7 * s11 + 2 * s3 * s7 * wz - 2 * s3 * s8 * s11 * wz) - s14 * z
            + y * (s12 - s4 * s7 * wx * wy + 0.5),

        y * (s20 + s6 * s7 * s10 - 2 * s3 * s8 * s10 * wx) - s18 * x
            + z * (s12 - s6 * s7 * wy * wz + 0.5),

        y * (s5 * s7 * s10 - 2 * s3 * s8 * s10 * wy) - s17 * x - s15 * z, y
            * (s4 * s7 * s10 + 2 * s3 * s7 * wz - 2 * s3 * s8 * s10 * wz)
            - s13 * z - x * (s4 * s7 * wx * wy - s12 + 0.5),

        z * (s20 + s6 * s7 * s9 - 2 * s3 * s8 * s9 * wx) - s16 * x
            - y * (s6 * s7 * wy * wz - s12 + 0.5), z
            * (s19 + s5 * s7 * s9 - 2 * s3 * s8 * s9 * wy) - s15 * y
            + x * (s12 - s5 * s7 * wx * wz + 0.5), z
            * (s4 * s7 * s9 - 2 * s3 * s8 * s9 * wz) - s14 * x - s13 * y)
          .finished());

    dlog.template block<3, 4>(0, 3) = dlog_dw * dw_dq;

  }

  return dlog;
}

}

#endif
