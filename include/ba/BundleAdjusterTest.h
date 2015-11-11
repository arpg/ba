
#include <Eigen/Sparse>
#include <calibu/Calibu.h>
#include "SparseBlockMatrix.h"
#include "SparseBlockMatrixOps.h"
#include "CeresCostFunctions.h"
#include "Utils.h"
#include "Types.h"

namespace ba {

////////////////////////////////////////////////////////////////////////////////
template<typename Scalar, int kLmDim>
bool _Test_dProjectionResidual_dX(
    const ProjectionResidualT<Scalar, kLmDim>& res,
    const PoseT<Scalar>& pose, const PoseT<Scalar>& ref_pose,
    const LandmarkT<Scalar, kLmDim>& lm,
    std::shared_ptr<calibu::Rig<Scalar>> rig)
{
  Eigen::Matrix<Scalar,2,1> dl_diff;
  Eigen::Matrix<Scalar,2,6> dx_meas_diff;
  Eigen::Matrix<Scalar,2,6> dx_ref_diff;

  dl_diff.setZero();
  dx_meas_diff.setZero();
  dx_ref_diff.setZero();

  double eps = 1e-9;
  if (lm.is_active) {
    Eigen::Matrix<Scalar,2,1> dz_dl_fd;

    Sophus::SE3Group<Scalar> Tss =
        (pose.t_wp * rig->cameras_[res.cam_id]->Pose()).inverse() *
        ref_pose.t_wp * rig->cameras_[lm.ref_cam_id]->Pose();

    const Eigen::Matrix<Scalar,2,1> pPlus =
    rig->cameras_[res.cam_id]->Transfer3D(
          Tss,lm.x_s.template head(3),lm.x_s[3]+eps);

    const Eigen::Matrix<Scalar,2,1> pMinus =
    rig->cameras_[res.cam_id]->Transfer3D(
          Tss,lm.x_s.template head(3),lm.x_s[3]-eps);

    dz_dl_fd = -(pPlus-pMinus)/(2*eps);
    dl_diff.template block<2, kLmDim>(0,0) =
        res.dz_dlm.template block<2, kLmDim>(0,0) -
        dz_dl_fd.template block<2, kLmDim>(0,0);

    std::cout << "dz_dl   :" << std::endl << res.dz_dlm << std::endl;
    std::cout << "dz_dl_fd:" << std::endl << dz_dl_fd << " norm: " <<
                 dl_diff.norm() <<  std::endl;
  }

  if (pose.is_active || ref_pose.is_active) {
    Eigen::Matrix<Scalar,2,6> dz_dx_fd;
    for(int ii = 0; ii < 6 ; ii++) {
        Eigen::Matrix<Scalar,6,1> delta;
        delta.setZero();
        delta[ii] = eps;
        Sophus::SE3Group<Scalar> Tss = (exp_decoupled(pose.t_wp, delta) *
            //(pose.t_wp * SE3t::exp(delta)*
                    rig->cameras_[res.cam_id]->Pose()).inverse() *
                    ref_pose.t_wp * rig->cameras_[lm.ref_cam_id]->Pose();

        const Eigen::Matrix<Scalar,2,1> pPlus =
        rig->cameras_[res.cam_id]->Transfer3D(
              Tss,lm.x_s.template head(3),lm.x_s[3]);

        delta[ii] = -eps;
        // Tss = (pose.t_wp *SE3t::exp(delta) *
        Tss = (exp_decoupled(pose.t_wp, delta) *
               rig->cameras_[res.cam_id]->Pose()).inverse() *
               ref_pose.t_wp * rig->cameras_[lm.ref_cam_id]->Pose();

        const Eigen::Matrix<Scalar,2,1> pMinus =
        rig->cameras_[res.cam_id]->Transfer3D(
              Tss,lm.x_s.template head(3),lm.x_s[3]);

        dz_dx_fd.col(ii) = -(pPlus-pMinus)/(2*eps);
    }
    dx_meas_diff = res.dz_dx_meas - dz_dx_fd;
    std::cerr << "dz_dx   :" << std::endl << res.dz_dx_meas << std::endl;
    std::cerr << "dz_dx_fd:" << std::endl <<
                 dz_dx_fd << " norm: " <<
                 (dx_meas_diff).norm() <<  std::endl;

    if (kLmDim == 1) {
      Eigen::Matrix<Scalar,2,6> dz_dx_ref_fd;
      for(int ii = 0; ii < 6 ; ii++) {
          Eigen::Matrix<Scalar,6,1> delta;
          delta.setZero();
          delta[ii] = eps;
          Sophus::SE3Group<Scalar> Tss =
              (pose.t_wp * rig->cameras_[res.cam_id]->Pose()).inverse() *
              exp_decoupled(ref_pose.t_wp, delta) *
              // (ref_pose.t_wp*SE3t::exp(delta)) *
              rig->cameras_[lm.ref_cam_id]->Pose();

          const Eigen::Matrix<Scalar,2,1> pPlus =
          rig->cameras_[res.cam_id]->Transfer3D(
                Tss,lm.x_s.template head(3),lm.x_s[3]);

          delta[ii] = -eps;
          Tss = (pose.t_wp * rig->cameras_[res.cam_id]->Pose()).inverse() *
              exp_decoupled(ref_pose.t_wp, delta) *
              // (ref_pose.t_wp*SE3t::exp(delta)) *
              rig->cameras_[lm.ref_cam_id]->Pose();

          const Eigen::Matrix<Scalar,2,1> pMinus =
          rig->cameras_[res.cam_id]->Transfer3D(
                Tss,lm.x_s.template head(3),lm.x_s[3]);

          dz_dx_ref_fd.col(ii) = -(pPlus-pMinus)/(2*eps);
      }

      dx_ref_diff = res.dz_dx_ref - dz_dx_ref_fd;
      std::cerr << "dz_dx_ref   :" << std::endl << res.dz_dx_ref << std::endl;
      std::cerr << "dz_dx_ref_fd:" << std::endl <<
                   dz_dx_ref_fd << " norm: " <<
                   (dx_ref_diff).norm() <<  std::endl;
    }
  }

  return ((dl_diff).norm() < NORM_THRESHOLD &&
         (dx_meas_diff).norm() < NORM_THRESHOLD &&
         (dx_ref_diff).norm() < NORM_THRESHOLD);
}

////////////////////////////////////////////////////////////////////////////////
template<typename Scalar>
bool _Test_dBinaryResidual_dX(const BinaryResidualT<Scalar>& res,
                              const Sophus::SE3Group<Scalar>& t_w1,
                              const Sophus::SE3Group<Scalar>& t_w2)
{
  // finite difference checking
  Eigen::Matrix<Scalar,6,6> dz_dx1_fd;
  Scalar deps = 1e-6;
  for (int ii = 0; ii < 6 ; ii++) {
    Eigen::Matrix<Scalar,6,1> delta;
    delta.setZero();
    delta[ii] = deps;
    const Eigen::Matrix<Scalar, 6, 1> pPlus =
        log_decoupled(exp_decoupled(t_w1, delta).inverse() * t_w2,
                      res.t_12);
    delta[ii] = -deps;
    const Eigen::Matrix<Scalar, 6, 1> pMinus =
        log_decoupled(exp_decoupled(t_w1, delta).inverse() * t_w2,
                      res.t_12);
    dz_dx1_fd.col(ii) = (pPlus-pMinus)/(2*deps);
  }
  std::cerr << "dbinary_dx1:" << res.dz_dx1 << std::endl;
  std::cerr << "dbinary_dx1_fd:" << dz_dx1_fd << std::endl;
  std::cerr << "diff:" << res.dz_dx1 - dz_dx1_fd << std::endl << " norm: " <<
               (res.dz_dx1 - dz_dx1_fd).norm() << std::endl;

  Eigen::Matrix<Scalar,6,6> dz_dx2_fd;
  for (int ii = 0; ii < 6 ; ii++) {
    Eigen::Matrix<Scalar,6,1> delta;
    delta.setZero();
    delta[ii] = deps;
    const Eigen::Matrix<Scalar, 6, 1> pPlus =
        log_decoupled(t_w1.inverse() * exp_decoupled(t_w2, delta), res.t_12);
    delta[ii] = -deps;
    const Eigen::Matrix<Scalar, 6, 1> pMinus =
        log_decoupled(t_w1.inverse() * exp_decoupled(t_w2, delta), res.t_12);
    dz_dx2_fd.col(ii) = (pPlus-pMinus)/(2*deps);
  }
  std::cerr << "dbinary_dx2:" << res.dz_dx2 << std::endl;
  std::cerr << "dbinary_dx2_fd:" << dz_dx2_fd << std::endl;
  std::cerr << "diff:" << res.dz_dx2 - dz_dx2_fd << std::endl << " norm: " <<
               (res.dz_dx2 - dz_dx2_fd).norm() << std::endl;

  return ((res.dz_dx2 - dz_dx2_fd).norm() < NORM_THRESHOLD &&
         (res.dz_dx1 - dz_dx1_fd).norm() < NORM_THRESHOLD);
}

////////////////////////////////////////////////////////////////////////////////
template<typename Scalar>
  bool _Test_dUnaryResidual_dX(const UnaryResidualT<Scalar>& res,
                               const Sophus::SE3Group<Scalar>& t_wp)
{
  Eigen::Matrix<Scalar,6,6> dunary_dx_fd;
  Scalar dEps = 1e-6;
  for (int ii = 0; ii < 6 ; ii++) {
    Eigen::Matrix<Scalar,6,1> delta;
    delta.setZero();
    delta[ii] = dEps;
    const Eigen::Matrix<Scalar, 6, 1> pPlus =
        log_decoupled(exp_decoupled(t_wp, delta), res.t_wp);
        // SE3t::log(res.t_wp.inverse() * t_wp * SE3t::exp(delta));
    delta[ii] = -dEps;
    const Eigen::Matrix<Scalar, 6, 1> pMinus =
        log_decoupled(exp_decoupled(t_wp, delta), res.t_wp);
        // SE3t::log(res.t_wp.inverse() * t_wp * SE3t::exp(delta));
    dunary_dx_fd.col(ii) = (pPlus-pMinus)/(2*dEps);
  }
  std::cerr << "dunary_dx:" << res.dz_dx << std::endl;
  std::cerr << "dunary_dx_fd:" << dunary_dx_fd << std::endl;
  std::cerr << "diff:" << res.dz_dx - dunary_dx_fd << std::endl << " norm: " <<
               (res.dz_dx - dunary_dx_fd).norm() << std::endl;

  return ((res.dz_dx - dunary_dx_fd).norm() < NORM_THRESHOLD);
}

////////////////////////////////////////////////////////////////////////////////
template<typename Scalar, int kResSize, int kPoseSize>
bool _Test_dImuResidual_dX(
    const PoseT<Scalar>& pose1, const PoseT<Scalar>& pose2,
    const ImuPoseT<Scalar>& imu_pose,
    const ImuResidualT<Scalar, kResSize, kPoseSize>& res,
    const Eigen::Matrix<Scalar, 3, 1>& gravity,
    const Eigen::Matrix<Scalar, 7, 6>& dse3_dx1,
    const Eigen::Matrix<Scalar,10,6>& dt_db,
    const ImuCalibrationT<Scalar>& imu)
{
  const Sophus::SE3Group<Scalar> t_12 =
      pose1.t_wp.inverse()*imu_pose.t_wp;
  const Sophus::SE3Group<Scalar>& t_w1 = pose1.t_wp;
  const Sophus::SE3Group<Scalar>& t_w2 = pose2.t_wp;
  const Sophus::SE3Group<Scalar>& t_2w = t_w2.inverse();

  Scalar eps = 1e-9;
  /*
  Eigen::Matrix<Scalar,6,6> Jlog_fd;
  for(int ii = 0 ; ii < 6 ; ii++){
      Eigen::Matrix<Scalar, 6, 1> eps_vec = Eigen::Matrix<Scalar, 6, 1>::Zero();
      eps_vec[ii] += eps;
      Eigen::Matrix<Scalar, 6, 1> res_plus =
          Sophus::SE3Group<Scalar>::log(
            t_w1*Sophus::SE3Group<Scalar>::exp(eps_vec) *
            (t_w2*t_12.inverse()).inverse());
      eps_vec[ii] -= 2*eps;
      Eigen::Matrix<Scalar, 6, 1> res_minus =
          Sophus::SE3Group<Scalar>::log(
            t_w1*Sophus::SE3Group<Scalar>::exp(eps_vec) *
            (t_w2*t_12.inverse()).inverse());
      Jlog_fd.col(ii) = (res_plus-res_minus)/(2*eps);
  }
  const Eigen::Matrix<Scalar,6,6> dlog_dtw1 =
      dlog_dx(t_w1,(t_w2*t_12.inverse()).inverse());
  std::cout << "dlog_dtw1 = [" << dlog_dtw1.format(kCleanFmt) <<
               "]" << std::endl;
  std::cout << "dlog_dtw1_f = [" << Jlog_fd.format(kCleanFmt) <<
               "]" << std::endl;
  std::cout << "dlog_dtw1 - dlog_dtw1_f = [" <<
               (dlog_dtw1- Jlog_fd).format(kCleanFmt) << "]" << std::endl;

  Eigen::Matrix<Scalar,6,6> dlog_dTwbf;
  for(int ii = 0 ; ii < 6 ; ii++){
      Eigen::Matrix<Scalar, 6, 1> eps_vec = Eigen::Matrix<Scalar, 6, 1>::Zero();
      eps_vec[ii] += eps;
      const Eigen::Matrix<Scalar, 6, 1> res_plus =
          Sophus::SE3Group<Scalar>::log(
            t_w1*t_12 *(t_w2*Sophus::SE3Group<Scalar>::exp(eps_vec)).inverse());
      eps_vec[ii] -= 2*eps;
      const Eigen::Matrix<Scalar, 6, 1> res_minus =
          Sophus::SE3Group<Scalar>::log(
            t_w1*t_12 *(t_w2*Sophus::SE3Group<Scalar>::exp(eps_vec)).inverse());
      dlog_dTwbf.col(ii) = (res_plus-res_minus)/(2*eps);
  }

  const auto dlog_dtw2 = -dlog_dx(t_w1*t_12, t_w2.inverse());
  std::cout << "dlog_dtw2 = [" << dlog_dtw2.format(kCleanFmt) <<
               "]" << std::endl;
  std::cout << "dlog_dtw2_f = [" << dlog_dTwbf.format(kCleanFmt) <<
               "]" << std::endl;
  std::cout << "dlog_dTwb - dlog_dTwbf = [" <<
               (dlog_dtw2-dlog_dTwbf).format(kCleanFmt) << "]" << std::endl;
  */


  // verify using finite differences
  Eigen::Matrix<Scalar,9,9> dz_dx1_fd;
  dz_dx1_fd.setZero();
  Eigen::Matrix<Scalar,9,2> dz_dg_fd;
  dz_dg_fd.setZero();
  Eigen::Matrix<Scalar,9,6> dz_db_fd;
  dz_db_fd.setZero();
  Eigen::Matrix<Scalar,7,6> dt_db_fd;
  dt_db_fd.setZero();

  Eigen::Matrix<Scalar,9,9>  dz_dx2_fd;
  for(int ii = 0; ii < 9 ; ii++){
      Eigen::Matrix<Scalar,9,1> eps_vec = Eigen::Matrix<Scalar,9,1>::Zero();
      eps_vec[ii] = eps;
      ImuPoseT<Scalar> y0_eps(pose2.t_wp,pose2.v_w,
                              Eigen::Matrix<Scalar, 3, 1>::Zero(),0);
      y0_eps.t_wp =
          exp_decoupled<Scalar>(y0_eps.t_wp, eps_vec.template head<6>());
          //y0_eps.t_wp *
          //Sophus::SE3Group<Scalar>::exp(eps_vec.template head<6>());

      y0_eps.v_w += eps_vec.template tail<3>();
      Eigen::Matrix<Scalar,9,1> r_plus;
      r_plus.template head<6>() = log_decoupled(imu_pose.t_wp, y0_eps.t_wp);
          //Sophus::SE3Group<Scalar>::log(imu_pose.t_wp * y0_eps.t_wp.inverse());
      r_plus.template tail<3>() = imu_pose.v_w - y0_eps.v_w;

      eps_vec[ii] = -eps;
      y0_eps =
          ImuPoseT<Scalar>(pose2.t_wp,pose2.v_w,
                           Eigen::Matrix<Scalar, 3, 1>::Zero(),0);;
      y0_eps.t_wp =
          exp_decoupled<Scalar>(y0_eps.t_wp, eps_vec.template head<6>());
          //y0_eps.t_wp *
          //Sophus::SE3Group<Scalar>::exp(eps_vec.template head<6>());

      y0_eps.v_w += eps_vec.template tail<3>();
      Eigen::Matrix<Scalar,9,1> r_minus;
      r_minus.template head<6>() = log_decoupled(imu_pose.t_wp, y0_eps.t_wp);
          //Sophus::SE3Group<Scalar>::log(imu_pose.t_wp * y0_eps.t_wp.inverse());
      r_minus.template tail<3>() = imu_pose.v_w - y0_eps.v_w;

      dz_dx2_fd.col(ii) = (r_plus-r_minus)/(2*eps);
  }
  auto dz_dx2 = res.dz_dx2.template block<9,9>(0,0);
  std::cout << "dz_dx2= " << std::endl <<
               dz_dx2.format(kCleanFmt) << std::endl;
  std::cout << "dz_dx2_fd = " << std::endl <<
               dz_dx2_fd.format(kCleanFmt) << std::endl;
  std::cout << "dz_dx2-dz_dx2_fd = " << std::endl <<
               (dz_dx2-dz_dx2_fd).format(kCleanFmt) << "norm: " <<
               (dz_dx2-dz_dx2_fd).norm() << std::endl;

  Eigen::Matrix<Scalar,7,6> dse3_dx1_fd;
  for(int ii = 0 ; ii < 6 ; ii++){
      Eigen::Matrix<Scalar, 6, 1> eps_vec = Eigen::Matrix<Scalar, 6, 1>::Zero();
      eps_vec[ii] = eps;
      PoseT<Scalar> pose_eps = pose1;
      pose_eps.t_wp = pose_eps.t_wp*Sophus::SE3Group<Scalar>::exp(eps_vec);
      // poseEps.t_wp = poseEps.t_wp * Sophus::SE3Group<Scalar>::exp(eps);
      std::vector<ImuPoseT<Scalar>> poses;
      const ImuPoseT<Scalar> imu_pose_plus =
          ImuResidualT<Scalar>::IntegrateResidual(
            pose_eps, res.measurements, imu.b_g, imu.b_a, gravity, poses);

      Eigen::Matrix<Scalar, 7, 1> error_plus;
      error_plus.template head<3>() =
          (imu_pose_plus.t_wp * t_w2.inverse()).translation();
      error_plus.template tail<4>() =
          (imu_pose_plus.t_wp * t_w2.inverse()).unit_quaternion().coeffs();
      eps_vec[ii] = -eps;
      pose_eps = pose1;
      pose_eps.t_wp = pose_eps.t_wp*Sophus::SE3Group<Scalar>::exp(eps_vec);
      // poseEps.t_wp = poseEps.t_wp * Sophus::SE3Group<Scalar>::exp(eps);
      poses.clear();
      const ImuPoseT<Scalar> imu_pose_minus =
          ImuResidualT<Scalar>::IntegrateResidual(
            pose_eps, res.measurements, imu.b_g, imu.b_a, gravity, poses);

      Eigen::Matrix<Scalar, 7, 1> error_minus;
      error_minus.template head<3>() =
          (imu_pose_minus.t_wp * t_w2.inverse()).translation();
      error_minus.template tail<4>() =
          (imu_pose_minus.t_wp * t_w2.inverse()).unit_quaternion().coeffs();

      dse3_dx1_fd.col(ii).template head<7>() =
          (error_plus - error_minus)/(2*eps);
  }

  std::cout << "dse3_dx1 = [" << std::endl <<
               dse3_dx1.format(kCleanFmt) << "]" << std::endl;
  std::cout << "dse3_dx1_Fd = [" << std::endl <<
               dse3_dx1_fd.format(kCleanFmt) << "]" << std::endl;
  std::cout << "dse3_dx1-dse3_dx1_fd = [" << std::endl <<
               (dse3_dx1-dse3_dx1_fd).format(kCleanFmt) << "] norm = " <<
               (dse3_dx1-dse3_dx1_fd).norm() << std::endl;


  for(int ii = 0 ; ii < 9 ; ii++){
      Eigen::Matrix<Scalar, 9, 1> eps_vec = Eigen::Matrix<Scalar, 9, 1>::Zero();
      eps_vec[ii] = eps;
      PoseT<Scalar> pose_eps = pose1;
      pose_eps.t_wp = exp_decoupled<Scalar>(pose_eps.t_wp,
                                            eps_vec.template head<6>());
          //pose_eps.t_wp*Sophus::SE3Group<Scalar>::exp(
          //  eps_vec.template head<6>());

      pose_eps.v_w += eps_vec.template tail<3>();
      std::vector<ImuPoseT<Scalar>> poses;
      const ImuPoseT<Scalar> imu_pose_plus =
          ImuResidualT<Scalar>::IntegrateResidual(pose_eps,res.measurements,
                                         imu.b_g,imu.b_a,gravity,poses);

      const Eigen::Matrix<Scalar, 6, 1> error_plus =
          // Sophus::SE3Group<Scalar>::log(imu_pose_plus.t_wp * t_w2.inverse());
          log_decoupled(imu_pose_plus.t_wp, t_w2);
      const Eigen::Matrix<Scalar, 3, 1> v_error_plus = imu_pose_plus.v_w - pose2.v_w;
      eps_vec[ii] = -eps;
      pose_eps = pose1;
      pose_eps.t_wp =
          exp_decoupled<Scalar>(pose_eps.t_wp,
                                eps_vec.template head<6>());
          //pose_eps.t_wp*Sophus::SE3Group<Scalar>::exp(
          //  eps_vec.template head<6>());

      pose_eps.v_w += eps_vec.template tail<3>();

      poses.clear();
      const ImuPoseT<Scalar> imu_pose_minus =
          ImuResidualT<Scalar>::IntegrateResidual(
            pose_eps,res.measurements,imu.b_g, imu.b_a,gravity,poses);
      const Eigen::Matrix<Scalar, 6, 1> error_minus =
          // Sophus::SE3Group<Scalar>::log(imu_pose_minus.t_wp * t_w2.inverse());
          log_decoupled(imu_pose_minus.t_wp, t_w2);
      const Eigen::Matrix<Scalar, 3, 1> v_error_minus =
          imu_pose_minus.v_w - pose2.v_w;

      dz_dx1_fd.col(ii).template head<6>() =
          (error_plus - error_minus)/(2*eps);
      dz_dx1_fd.col(ii).template tail<3>() =
          (v_error_plus - v_error_minus)/(2*eps);
  }


  for(int ii = 0 ; ii < 2 ; ii++){
      Eigen::Matrix<Scalar, 2, 1> eps_vec = Eigen::Matrix<Scalar, 2, 1>::Zero();
      eps_vec[ii] += eps;
      std::vector<ImuPoseT<Scalar>> poses;
      const Eigen::Matrix<Scalar, 2, 1> g_plus = imu.g+eps_vec;
      const ImuPoseT<Scalar> imu_pose_plus =
          ImuResidualT<Scalar>::IntegrateResidual(
            pose1,res.measurements,imu.b_g, imu.b_a,
            GetGravityVector(g_plus),poses);

      const Eigen::Matrix<Scalar, 6, 1> error_plus =
          Sophus::SE3Group<Scalar>::log(imu_pose_plus.t_wp*t_w2.inverse());

      const Eigen::Matrix<Scalar, 3, 1> v_error_plus =
          imu_pose_plus.v_w - pose2.v_w;

      eps_vec[ii] -= 2*eps;
      poses.clear();
      const Eigen::Matrix<Scalar, 2, 1> g_minus = imu.g+eps_vec;
      const ImuPoseT<Scalar> imu_pose_minus =
          ImuResidualT<Scalar>::IntegrateResidual(
            pose1,res.measurements,imu.b_g, imu.b_a,
            GetGravityVector(g_minus),poses);

      const Eigen::Matrix<Scalar, 6, 1> error_minus =
          Sophus::SE3Group<Scalar>::log(imu_pose_minus.t_wp*t_w2.inverse());

      const Eigen::Matrix<Scalar, 3, 1> v_error_minus =
          imu_pose_minus.v_w - pose2.v_w;

      dz_dg_fd.col(ii).template head<6>() =
          (error_plus - error_minus)/(2*eps);
      dz_dg_fd.col(ii).template tail<3>() =
          (v_error_plus - v_error_minus)/(2*eps);
  }

  Eigen::Matrix<Scalar, 6, 1> bias_vec;
  bias_vec.template head<3>() = imu.b_g;
  bias_vec.template tail<3>() = imu.b_a;
  for(int ii = 0 ; ii < 6 ; ii++){
      Eigen::Matrix<Scalar, 6, 1> eps_vec = Eigen::Matrix<Scalar, 6, 1>::Zero();
      eps_vec[ii] += eps;
      std::vector<ImuPoseT<Scalar>> poses;
      const Eigen::Matrix<Scalar, 6, 1> plus_b = bias_vec + eps_vec;
      const ImuPoseT<Scalar> imu_pose_plus =
          ImuResidualT<Scalar>::IntegrateResidual(pose1,res.measurements,
                                         plus_b.template head<3>(),
                                         plus_b.template tail<3>(),
                                         gravity,poses);
      Eigen::Matrix<Scalar, 7, 1> error_plus;
      const Sophus::SE3Group<Scalar> t_plus = imu_pose_plus.t_wp;

      error_plus.template head<3>() = (t_plus.translation());
      error_plus.template tail<4>() = (t_plus.unit_quaternion().coeffs());

      eps_vec[ii] -= 2*eps;
      const Eigen::Matrix<Scalar, 6, 1> minus_b = bias_vec + eps_vec;
      poses.clear();
      const ImuPoseT<Scalar> imu_pose_minus =
          ImuResidualT<Scalar>::IntegrateResidual(pose1,res.measurements,
                                         minus_b.template head<3>(),
                                         minus_b.template tail<3>(),
                                         gravity,poses);
      Eigen::Matrix<Scalar, 7, 1> error_minus;
      const Sophus::SE3Group<Scalar> t_minus = imu_pose_minus.t_wp;

      error_minus.template head<3>() = (t_minus.translation());
      error_minus.template tail<4>() = (t_minus.unit_quaternion().coeffs());
      dt_db_fd.col(ii).template head<7>() =
          (error_plus - error_minus)/(2*eps);
  }


  // Eigen::Matrix<Scalar, 6, 1> bias_vec;
  bias_vec.template head<3>() = imu.b_g;
  bias_vec.template tail<3>() = imu.b_a;
  for(int ii = 0 ; ii < 6 ; ii++){
      Eigen::Matrix<Scalar, 6, 1> eps_vec = Eigen::Matrix<Scalar, 6, 1>::Zero();
      eps_vec[ii] += eps;
      std::vector<ImuPoseT<Scalar>> poses;
      const Eigen::Matrix<Scalar, 6, 1> plus_b = bias_vec + eps_vec;
      const ImuPoseT<Scalar> imu_pose_plus =
          ImuResidualT<Scalar>::IntegrateResidual(pose1,res.measurements,
                                         plus_b.template head<3>(),
                                         plus_b.template tail<3>(),
                                         gravity,poses);
      const Eigen::Matrix<Scalar, 6, 1> error_plus =
          // Sophus::SE3Group<Scalar>::log(imu_pose_plus.t_wp * t_w2.inverse());
          log_decoupled(imu_pose_plus.t_wp, t_w2);
      const Eigen::Matrix<Scalar, 3, 1> v_error_plus =
          imu_pose_plus.v_w - pose2.v_w;

      eps_vec[ii] -= 2*eps;
      const Eigen::Matrix<Scalar, 6, 1> minus_b = bias_vec + eps_vec;
      poses.clear();
      const ImuPoseT<Scalar> imu_pose_minus =
          ImuResidualT<Scalar>::IntegrateResidual(pose1,res.measurements,
                                         minus_b.template head<3>(),
                                         minus_b.template tail<3>(),
                                         gravity,poses);
      const Eigen::Matrix<Scalar, 6, 1> error_minus =
          //Sophus::SE3Group<Scalar>::log(imu_pose_minus.t_wp * t_w2.inverse());
          log_decoupled(imu_pose_minus.t_wp, t_w2);
      const Eigen::Matrix<Scalar, 3, 1> v_error_minus =
          imu_pose_minus.v_w - pose2.v_w;

      dz_db_fd.col(ii).template head<6>() =
          (error_plus - error_minus)/(2*eps);
      dz_db_fd.col(ii).template tail<3>() =
          (v_error_plus - v_error_minus)/(2*eps);
  }


  const auto dz_dx1 = res.dz_dx1.template block<9,9>(0,0);
  const auto dz_db = res.dz_db.template block<9,6>(0,0);
  const auto dt_db_ = dt_db.template block<7,6>(0,0);
  std::cout << "dz_dx1 = [" << std::endl << dz_dx1.format(kCleanFmt) <<
               "]" << std::endl;
  std::cout << "dz_dx1 = [" << std::endl << dz_dx1_fd.format(kCleanFmt) <<
               "]" << std::endl;
  std::cout << "dz_dx1-dz_dx1_fd = [" << std::endl <<
               (dz_dx1-dz_dx1_fd).format(kCleanFmt) << "] norm = " <<
               (dz_dx1-dz_dx1_fd).norm() << std::endl;

  std::cout << "dz_dg = [" << std::endl <<
               res.dz_dg.format(kCleanFmt) << "]" << std::endl;
  std::cout << "dz_dg_fd = [" << std::endl <<
               dz_dg_fd.format(kCleanFmt) << "]" << std::endl;
  std::cout << "dz_dg-dz_dg_fd = [" << std::endl <<
               (res.dz_dg-dz_dg_fd).format(kCleanFmt) << "] norm = " <<
               (res.dz_dg-dz_dg_fd).norm() << std::endl;

  std::cout << "dz_db = [" << std::endl << dz_db.format(kCleanFmt) << "]" <<
                std::endl;
  std::cout << "dz_db_fd = [" << std::endl << dz_db_fd.format(kCleanFmt) <<
               "]" << std::endl;
  std::cout << "dz_db-dz_db_fd = [" << std::endl <<
               (dz_db-dz_db_fd).format(kCleanFmt) << "] norm = " <<
               (dz_db-dz_db_fd).norm() << std::endl;

  std::cout << "dt_db = [" << std::endl << dt_db_.format(kCleanFmt) << "]" <<
                std::endl;
  std::cout << "dt_db_fd = [" << std::endl << dt_db_fd.format(kCleanFmt) <<
               "]" << std::endl;
  std::cout << "dt_db-dz_db_fd = [" << std::endl <<
               (dt_db_-dt_db_fd).format(kCleanFmt) << "] norm = " <<
               (dt_db_-dt_db_fd).norm() << std::endl;

  return((dz_dx1-dz_dx1_fd).norm() < NORM_THRESHOLD &&
         (res.dz_dg-dz_dg_fd).norm() < NORM_THRESHOLD &&
         (dz_db-dz_db_fd).norm() < NORM_THRESHOLD &&
         (dt_db_-dt_db_fd).norm() < NORM_THRESHOLD);
}
}
