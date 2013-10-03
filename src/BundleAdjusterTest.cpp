
#include <ba/BundleAdjuster.h>

namespace ba {
////////////////////////////////////////////////////////////////////////////////
template< typename Scalar,int kLmDim, int kPoseDim, int kCalibDim >
bool BundleAdjuster<Scalar, kLmDim, kPoseDim, kCalibDim>::_Test_dImuResidual_dX(
    const Pose& pose1, const Pose& pose2, const ImuPose& imu_pose,
    const ImuResidual& res, const Vector3t gravity,
    const Eigen::Matrix<Scalar, 7, 6>& dse3_dx1,
    const Eigen::Matrix<Scalar,10,6>& dt_db)
{
  const SE3t t_12 = pose1.t_wp.inverse()*imu_pose.t_wp;
  const SE3t& t_w1 = pose1.t_wp;
  const SE3t& t_w2 = pose2.t_wp;
  const SE3t& t_2w = t_w2.inverse();

  Scalar dEps = 1e-9;
  Eigen::Matrix<Scalar,6,6> Jlog_fd;
  for(int ii = 0 ; ii < 6 ; ii++){
      Vector6t eps_vec = Vector6t::Zero();
      eps_vec[ii] += dEps;
      Vector6t res_plus = SE3t::log(t_w1*SE3t::exp(eps_vec) *
                                   (t_w2*t_12.inverse()).inverse());
      eps_vec[ii] -= 2*dEps;
      Vector6t res_minus = SE3t::log(t_w1*SE3t::exp(eps_vec) *
                                    (t_w2*t_12.inverse()).inverse());
      Jlog_fd.col(ii) = (res_plus-res_minus)/(2*dEps);
  }
  const Eigen::Matrix<Scalar,6,6> dlog_dtw1 =
      dLog_dX(t_w1,(t_w2*t_12.inverse()).inverse());
  std::cout << "dlog_dtw1 = [" << dlog_dtw1.format(kCleanFmt) <<
               "]" << std::endl;
  std::cout << "dlog_dtw1_f = [" << Jlog_fd.format(kCleanFmt) <<
               "]" << std::endl;
  std::cout << "dlog_dtw1 - dlog_dtw1_f = [" <<
               (dlog_dtw1- Jlog_fd).format(kCleanFmt) << "]" << std::endl;

  Eigen::Matrix<Scalar,6,6> dlog_dTwbf;
  for(int ii = 0 ; ii < 6 ; ii++){
      Vector6t eps_vec = Vector6t::Zero();
      eps_vec[ii] += dEps;
      const Vector6t res_plus = SE3t::log(t_w1*t_12 *
                                         (t_w2*SE3t::exp(eps_vec)).inverse());
      eps_vec[ii] -= 2*dEps;
      const Vector6t res_minus = SE3t::log(t_w1*t_12 *
                                          (t_w2*SE3t::exp(eps_vec)).inverse());
      dlog_dTwbf.col(ii) = (res_plus-res_minus)/(2*dEps);
  }

  const auto dlog_dtw2 = -dLog_dX(t_w1*t_12, t_w2.inverse());
  std::cout << "dlog_dtw2 = [" << dlog_dtw2.format(kCleanFmt) <<
               "]" << std::endl;
  std::cout << "dlog_dtw2_f = [" << dlog_dTwbf.format(kCleanFmt) <<
               "]" << std::endl;
  std::cout << "dlog_dTwb - dlog_dTwbf = [" <<
               (dlog_dtw2-dlog_dTwbf).format(kCleanFmt) << "]" << std::endl;



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
      eps_vec[ii] += dEps;
      ImuPose y0_eps(pose2.t_wp,pose2.v_w, Vector3t::Zero(),0);
      y0_eps.t_wp = y0_eps.t_wp * SE3t::exp(eps_vec.template head<6>());
      y0_eps.v_w += eps_vec.template tail<3>();
      Eigen::Matrix<Scalar,9,1> r_plus;
      r_plus.template head<6>() =
          SE3t::log(imu_pose.t_wp * y0_eps.t_wp.inverse());
      r_plus.template tail<3>() = imu_pose.v_w - y0_eps.v_w;



      eps_vec[ii] -= 2*dEps;
      y0_eps = ImuPose(pose2.t_wp,pose2.v_w, Vector3t::Zero(),0);;
      y0_eps.t_wp = y0_eps.t_wp * SE3t::exp(eps_vec.template head<6>());
      y0_eps.v_w += eps_vec.template tail<3>();
      Eigen::Matrix<Scalar,9,1> r_minus;
      r_minus.template head<6>() =
          SE3t::log(imu_pose.t_wp * y0_eps.t_wp.inverse());
      r_minus.template tail<3>() = imu_pose.v_w - y0_eps.v_w;

      dz_dx2_fd.col(ii) = (r_plus-r_minus)/(2*dEps);
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
      Vector6t eps_vec = Vector6t::Zero();
      eps_vec[ii] = dEps;
      Pose pose_eps = pose1;
      pose_eps.t_wp = pose_eps.t_wp*SE3t::exp(eps_vec);
      // poseEps.t_wp = poseEps.t_wp * SE3t::exp(eps);
      std::vector<ImuPose> poses;
      const ImuPose imu_pose_plus =
          ImuResidual::IntegrateResidual(pose_eps,res.measurements,imu_.b_g,
                                         imu_.b_a,gravity,poses);
      Vector7t error_plus;
      error_plus.template head<3>() =
          (imu_pose_plus.t_wp * t_w2.inverse()).translation();
      error_plus.template tail<4>() =
          (imu_pose_plus.t_wp * t_w2.inverse()).unit_quaternion().coeffs();
      eps_vec[ii] = -dEps;
      pose_eps = pose1;
      pose_eps.t_wp = pose_eps.t_wp*SE3t::exp(eps_vec);
      // poseEps.t_wp = poseEps.t_wp * SE3t::exp(eps);
      poses.clear();
      const ImuPose imu_pose_minus =
          ImuResidual::IntegrateResidual(pose_eps,res.measurements,imu_.b_g,
                                         imu_.b_a,gravity,poses);
      Vector7t error_minus;
      error_minus.template head<3>() =
          (imu_pose_minus.t_wp * t_w2.inverse()).translation();
      error_minus.template tail<4>() =
          (imu_pose_minus.t_wp * t_w2.inverse()).unit_quaternion().coeffs();

      dse3_dx1_fd.col(ii).template head<7>() =
          (error_plus - error_minus)/(2*dEps);
  }

  std::cout << "dse3_dx1 = [" << std::endl <<
               dse3_dx1.format(kCleanFmt) << "]" << std::endl;
  std::cout << "dse3_dx1_Fd = [" << std::endl <<
               dse3_dx1_fd.format(kCleanFmt) << "]" << std::endl;
  std::cout << "dse3_dx1-dse3_dx1_fd = [" << std::endl <<
               (dse3_dx1-dse3_dx1_fd).format(kCleanFmt) << "] norm = " <<
               (dse3_dx1-dse3_dx1_fd).norm() << std::endl;


  for(int ii = 0 ; ii < 9 ; ii++){
      Vector9t eps_vec = Vector9t::Zero();
      eps_vec[ii] = dEps;
      Pose pose_eps = pose1;
      pose_eps.t_wp = pose_eps.t_wp*SE3t::exp(eps_vec.template head<6>());
      pose_eps.v_w += eps_vec.template tail<3>();
      std::vector<ImuPose> poses;
      const ImuPose imu_pose_plus =
          ImuResidual::IntegrateResidual(pose_eps,res.measurements,
                                         imu_.b_g,imu_.b_a,gravity,poses);

      const Vector6t error_plus =
          SE3t::log(imu_pose_plus.t_wp * t_w2.inverse());
      const Vector3t v_error_plus = imu_pose_plus.v_w - pose2.v_w;
      eps_vec[ii] = -dEps;
      pose_eps = pose1;
      pose_eps.t_wp = pose_eps.t_wp*SE3t::exp(eps_vec.template head<6>());
      pose_eps.v_w += eps_vec.template tail<3>();

      poses.clear();
      const ImuPose imu_pose_minus =
          ImuResidual::IntegrateResidual(pose_eps,res.measurements,imu_.b_g,
                                         imu_.b_a,gravity,poses);
      const Vector6t error_minus =
          SE3t::log(imu_pose_minus.t_wp * t_w2.inverse());
      const Vector3t v_error_minus = imu_pose_minus.v_w - pose2.v_w;
      dz_dx1_fd.col(ii).template head<6>() =
          (error_plus - error_minus)/(2*dEps);
      dz_dx1_fd.col(ii).template tail<3>() =
          (v_error_plus - v_error_minus)/(2*dEps);
  }


  for(int ii = 0 ; ii < 2 ; ii++){
      Vector2t eps_vec = Vector2t::Zero();
      eps_vec[ii] += dEps;
      std::vector<ImuPose> poses;
      const Vector2t g_plus = imu_.g+eps_vec;
      const ImuPose imu_pose_plus =
          ImuResidual::IntegrateResidual(pose1,res.measurements,imu_.b_g,
                                         imu_.b_a,
                                         GetGravityVector(g_plus),poses);
      const Vector6t error_plus =
          SE3t::log(imu_pose_plus.t_wp*t_w2.inverse());

      const Vector3t v_error_plus = imu_pose_plus.v_w - pose2.v_w;
      eps_vec[ii] -= 2*dEps;
      poses.clear();
      const Vector2t g_minus = imu_.g+eps_vec;
      const ImuPose imu_pose_minus =
          ImuResidual::IntegrateResidual(pose1,res.measurements,imu_.b_g,
                                         imu_.b_a,
                                         GetGravityVector(g_minus),poses);
      const Vector6t error_minus =
          SE3t::log(imu_pose_minus.t_wp*t_w2.inverse());

      const Vector3t v_error_minus = imu_pose_minus.v_w - pose2.v_w;
      dz_dg_fd.col(ii).template head<6>() =
          (error_plus - error_minus)/(2*dEps);
      dz_dg_fd.col(ii).template tail<3>() =
          (v_error_plus - v_error_minus)/(2*dEps);
  }

  Vector6t bias_vec;
  bias_vec.template head<3>() = imu_.b_g;
  bias_vec.template tail<3>() = imu_.b_a;
  for(int ii = 0 ; ii < 6 ; ii++){
      Vector6t eps_vec = Vector6t::Zero();
      eps_vec[ii] += dEps;
      std::vector<ImuPose> poses;
      const Vector6t plus_b = bias_vec + eps_vec;
      const ImuPose imu_pose_plus =
          ImuResidual::IntegrateResidual(pose1,res.measurements,
                                         plus_b.template head<3>(),
                                         plus_b.template tail<3>(),
                                         gravity,poses);
      Vector7t error_plus;
      const SE3t t_plus = (imu_pose_plus.t_wp * t_w2.inverse());
      error_plus.template head<3>() = (t_plus.translation());
      error_plus.template tail<4>() = (t_plus.unit_quaternion().coeffs());

      eps_vec[ii] -= 2*dEps;
      const Vector6t minus_b = bias_vec + eps_vec;
      poses.clear();
      const ImuPose imu_pose_minus =
          ImuResidual::IntegrateResidual(pose1,res.measurements,
                                         minus_b.template head<3>(),
                                         minus_b.template tail<3>(),
                                         gravity,poses);
      Vector7t error_minus;
      const SE3t t_minus = (imu_pose_minus.t_wp * t_w2.inverse());
      error_minus.template head<3>() = (t_minus.translation());
      error_minus.template tail<4>() = (t_minus.unit_quaternion().coeffs());
      dt_db_fd.col(ii).template head<7>() =
          (error_plus - error_minus)/(2*dEps);
  }


  // Vector6t bias_vec;
  bias_vec.template head<3>() = imu_.b_g;
  bias_vec.template tail<3>() = imu_.b_a;
  for(int ii = 0 ; ii < 6 ; ii++){
      Vector6t eps_vec = Vector6t::Zero();
      eps_vec[ii] += dEps;
      std::vector<ImuPose> poses;
      const Vector6t plus_b = bias_vec + eps_vec;
      const ImuPose imu_pose_plus =
          ImuResidual::IntegrateResidual(pose1,res.measurements,
                                         plus_b.template head<3>(),
                                         plus_b.template tail<3>(),
                                         gravity,poses);
      const Vector6t error_plus =
          SE3t::log(imu_pose_plus.t_wp * t_w2.inverse());
      const Vector3t v_error_plus = imu_pose_plus.v_w - pose2.v_w;

      eps_vec[ii] -= 2*dEps;
      const Vector6t minus_b = bias_vec + eps_vec;
      poses.clear();
      const ImuPose imu_pose_minus =
          ImuResidual::IntegrateResidual(pose1,res.measurements,
                                         minus_b.template head<3>(),
                                         minus_b.template tail<3>(),
                                         gravity,poses);
      const Vector6t error_minus =
          SE3t::log(imu_pose_minus.t_wp * t_w2.inverse());
      const Vector3t v_error_minus = imu_pose_minus.v_w - pose2.v_w;
      dz_db_fd.col(ii).template head<6>() =
          (error_plus - error_minus)/(2*dEps);
      dz_db_fd.col(ii).template tail<3>() =
          (v_error_plus - v_error_minus)/(2*dEps);
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
