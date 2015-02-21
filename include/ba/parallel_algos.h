#include <ba/BundleAdjuster.h>
#pragma once
#include <ba/Types.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace ba {
  template <typename BaType, typename Scalar>
  class ParallelProjectionResiduals {
  public:
    BaType& tracker;

    // Reduced quantities.
    std::vector<Scalar> errors;
    std::vector<Scalar> cond_errors;

    ParallelProjectionResiduals(BaType& tracker_ref) :
      tracker(tracker_ref)
    {}

    ParallelProjectionResiduals(const ParallelProjectionResiduals &other,
                                tbb::split) :
      tracker(other.tracker)
    {}

    void join(ParallelProjectionResiduals& other) {
      errors.insert(errors.end(), other.errors.begin(),
                    other.errors.end());

      cond_errors.insert(cond_errors.end(), other.cond_errors.begin(),
                    other.cond_errors.end());
    }

    void operator() (const tbb::blocked_range<int>& r) {
      for (int ii = r.begin(); ii != r.end(); ii++) {
        typename BaType::ProjectionResidual& res = tracker.proj_residuals_[ii];
        // calculate measurement jacobians

        // Tsw = T_cv * T_vw
        typename BaType::Landmark& lm = tracker.landmarks_[res.landmark_id];
        typename BaType::Pose& pose = tracker.poses_[res.x_meas_id];
        typename BaType::Pose& ref_pose = tracker.poses_[res.x_ref_id];
        calibu::CameraInterface<Scalar>* cam =
            tracker.rig_.cameras_[res.cam_id];

        const typename BaType::SE3t& t_vs_m = tracker.rig_.t_wc_[res.cam_id];
        const typename BaType::SE3t& t_vs_r = tracker.rig_.t_wc_[lm.ref_cam_id];
        const typename BaType::SE3t& t_sw_m =
            pose.GetTsw(res.cam_id, tracker.rig_);
        const typename BaType::SE3t t_ws_r =
            ref_pose.GetTsw(lm.ref_cam_id, tracker.rig_).inverse();

        Eigen::VectorXd backup_params = cam->GetParams();
        if (tracker.options_.use_per_pose_cam_params) {
          cam->SetParams(pose.cam_params);
        }

        const typename BaType::Vector2t p = BaType::kLmDim == 3 ?
              cam->Transfer3d(t_sw_m, lm.x_w.template head<3>(), lm.x_w(3)) :
              cam->Transfer3d(t_sw_m * t_ws_r, lm.x_s.template head<3>(),
                              lm.x_s(3));

        res.residual = res.z - p;
        // std::cerr << "res " << res.residual_id << " : pre" <<
        //                res.residual.norm() << std::endl;

        const typename BaType::Vector4t x_s_m = BaType::kLmDim == 1 ?
              MultHomogeneous(t_sw_m * t_ws_r, lm.x_s) :
              MultHomogeneous(t_sw_m, lm.x_w);

        const Eigen::Matrix<Scalar,2,4> dt_dp_m = cam->dTransfer3d_dray(
              typename BaType::SE3t(), x_s_m.template head<3>(),x_s_m(3));

        const Eigen::Matrix<Scalar,2,4> dt_dp_s = BaType::kLmDim == 3 ?
              dt_dp_m * t_sw_m.matrix() :
              dt_dp_m * (t_sw_m*t_ws_r).matrix();

        // Landmark Jacobian
        if (lm.is_active) {
          res.dz_dlm = -dt_dp_s.template block<2, BaType::kLmDim>(
                0, BaType::kLmDim == 3 ? 0 : 3 );
        }

        const bool diff_poses =  res.x_ref_id != res.x_meas_id;

        if (pose.is_active || ref_pose.is_active) {
          // If the reference and measurement poses are the same, the derivative
          // is zero.
          if (diff_poses) {
            res.dz_dx_meas =
                -dt_dp_m *
                dt_x_dt<Scalar>(t_sw_m, t_ws_r.matrix() * lm.x_s) *
                dt1_t2_dt2(t_vs_m.inverse()/*, pose.t_wp.inverse()*/) *
                dinv_exp_decoupled_dx(pose.t_wp);
          } else {
            res.dz_dx_meas.setZero();
          }

          // only need this if we are in inverse depth mode and the poses aren't
          // the same
          if (BaType::kLmDim == 1) {
            if (diff_poses) {
              res.dz_dx_ref =
                  -dt_dp_m *
                  dt_x_dt<Scalar>(t_sw_m * ref_pose.t_wp,
                                  t_vs_r.matrix() * lm.x_s)
                  * dt1_t2_dt2(t_sw_m/*, ref_pose.t_wp*/) *
                  dexp_decoupled_dx(ref_pose.t_wp);
            } else {
              res.dz_dx_ref.setZero();
            }

            if (BaType::kCamParamsInCalib) {
              res.dz_dcam_params =
                  -cam->dTransfer_dparams(t_sw_m * t_ws_r,
                                          lm.z_ref, lm.x_s(3));
            }

            if (BaType::kTvsInCalib) {
              // Total derivative of transfer.
              res.dz_dtvs =
                  -dt_dp_m *
                  dt_x_dt<Scalar>(t_sw_m * t_ws_r, lm.x_s) *
                  (dt1_t2_dt2(t_vs_m.inverse()) *
                   dt1_t2_dt2(pose.t_wp.inverse() * ref_pose.t_wp) *
                   dexp_decoupled_dx(t_vs_r) +
                   dt1_t2_dt1(t_vs_m.inverse(),
                              pose.t_wp.inverse() * ref_pose.t_wp * t_vs_r) *
                   dinv_exp_decoupled_dx(t_vs_m));
            }
          }
        }

        BA_TEST(_Test_dProjectionResidual_dX(res, pose, ref_pose, lm, rig_));

        if (tracker.options_.use_per_pose_cam_params) {
          cam->SetParams(backup_params);
        }

        // set the residual in m_R which is dense
        res.weight =  res.orig_weight;
        res.mahalanobis_distance = res.residual.squaredNorm() * res.weight;
        // this array is used to calculate the robust norm
        if (res.is_conditioning) {
          cond_errors.push_back(res.mahalanobis_distance);
        } else {
          errors.push_back(res.mahalanobis_distance);
        }
      }
    }
  };


  template <typename BaType, typename Scalar>
  class ParallelInertialResiduals {
  public:
    BaType& tracker;

    // Reduced quantities.
    std::vector<Scalar> errors;

    ParallelInertialResiduals(BaType& tracker_ref) :
      tracker(tracker_ref)
    {}

    ParallelInertialResiduals(const ParallelInertialResiduals &other,
                                tbb::split) :
      tracker(other.tracker)
    {}

    void join(ParallelInertialResiduals& other) {
      errors.insert(errors.end(), other.errors.begin(),
                    other.errors.end());
    }

    void operator() (const tbb::blocked_range<int>& r) {
      for (int ii = r.begin(); ii != r.end(); ii++) {
        typename BaType::ImuResidual& res = tracker.inertial_residuals_[ii];
        // set up the initial pose for the integration
        const typename BaType::Vector3t gravity = BaType::kGravityInCalib ?
              GetGravityVector(tracker.imu_.g) : tracker.imu_.g_vec;

        const typename BaType::Pose& pose1 = tracker.poses_[res.pose1_id];
        const typename BaType::Pose& pose2 = tracker.poses_[res.pose2_id];


        StartTimer(_j_evaluation_inertial_integration_);
        bool compute_covarince =
            !tracker.options_.calculate_inertial_covariance_once ||
            !res.covariance_computed;
        if (compute_covarince) {
          res.c_integration.setZero();
        }

        typename BaType::ImuPose imu_pose =
            BaType::ImuResidual::IntegrateResidual(
              pose1, res.measurements, pose1.b.template head<3>(),
              pose1.b.template tail<3>(), gravity, res.poses,
              compute_covarince ? &res.dintegration_db : nullptr,
              nullptr,
              compute_covarince ? &res.c_integration : nullptr,
              compute_covarince ? &tracker.imu_.r : nullptr);
        res.covariance_computed = true;
        // PrintTimer(_j_evaluation_inertial_integration_);

        Scalar total_dt =
            res.measurements.back().time - res.measurements.front().time;

        const typename BaType::SE3t& t_w1 = pose1.t_wp;
        const typename BaType::SE3t& t_w2 = pose2.t_wp;
        // const SE3t& t_2w = t_w2.inverse();

        // now given the poses, calculate the jacobians.
        // First subtract gravity, initial pose and velocity from the delta T and delta V
        typename BaType::SE3t t_12_0 = imu_pose.t_wp;
        // subtract starting velocity and gravity
        t_12_0.translation() -=
            (-gravity*0.5*powi(total_dt,2) + pose1.v_w*total_dt);
        // subtract starting pose
        t_12_0 = pose1.t_wp.inverse() * t_12_0;
        // Augment the velocity delta by subtracting effects of gravity
        typename BaType::Vector3t v_12_0 = imu_pose.v_w - pose1.v_w;
        v_12_0 += gravity*total_dt;
        // rotate the velocity delta so that it starts from orientation=Ident
        v_12_0 = pose1.t_wp.so3().inverse() * v_12_0;

        // derivative with respect to the start pose
        res.residual.setZero();
        res.dz_dx1.setZero();
        res.dz_dx2.setZero();
        res.dz_dg.setZero();
        res.dz_db.setZero();

        // Twa^-1 is multiplied here as we need the velocity derivative in the
        //frame of pose A, as the log is taken from this frame
        res.dz_dx1.template block<3,3>(0,6) =
            BaType::Matrix3t::Identity() * total_dt;
        for (int ii = 0; ii < 3 ; ++ii) {
          res.dz_dx1.template block<3,1>(6,3+ii) =
              t_w1.so3().matrix() *
              Sophus::SO3Group<Scalar>::generator(ii) * v_12_0;
        }

        // dr/dv (pose1)
        res.dz_dx1.template block<3,3>(6,6) =
            BaType::Matrix3t::Identity();
        // dr/dx (pose1)
        // res.dz_dx1.template block<6,6>(0,0) =  dlog_dse3*dse3_dx1;
        res.dz_dx1.template block<6,6>(0,0) =
            dLog_decoupled_dt1(imu_pose.t_wp, t_w2) *
            dt1_t2_dt1(t_w1, t_12_0) *
            dexp_decoupled_dx(t_w1);

        // the - sign is here because of the exp(-x) within the log
        // res.dz_dx2.template block<6,6>(0,0) = -dLog_dX(imu_pose.t_wp,t_2w);
        res.dz_dx2.template block<6,6>(0,0) =
            dlog_decoupled_dt2(imu_pose.t_wp, t_w2) *
            dexp_decoupled_dx(t_w2);


        // dr/dv (pose2)
        res.dz_dx2.template block<3,3>(6,6) =
            -BaType::Matrix3t::Identity();

        res.weight = res.orig_weight;
        // res.residual.template head<6>() = SE3t::log(imu_pose.t_wp*t_2w);
        res.residual.template head<6>() = log_decoupled(imu_pose.t_wp, t_w2);
        res.residual.template segment<3>(6) = imu_pose.v_w - pose2.v_w;


        const Eigen::Matrix<Scalar,6,7> dlogt1t2_dt1 =
            dLog_decoupled_dt1(imu_pose.t_wp, t_w2);

        // Transform the covariance through the multiplication by t_2w as well as
        // the log
        Eigen::Matrix<Scalar,9,10> dse3t1t2v_dt1;
        dse3t1t2v_dt1.setZero();
        dse3t1t2v_dt1.template topLeftCorner<6,7>() = dlogt1t2_dt1;
        dse3t1t2v_dt1.template bottomRightCorner<3,3>().setIdentity();

        res.cov_inv.setZero();
        Eigen::Matrix<Scalar, BaType::ImuResidual::kResSize, 1> sigmas =
            Eigen::Matrix<Scalar, BaType::ImuResidual::kResSize, 1>::Ones();
        // Write the bias uncertainties into the covariance matrix.
        if (BaType::kBiasInState) {
          sigmas.template segment<6>(9) = tracker.imu_.r_b * total_dt;
        }

        res.cov_inv.diagonal() = sigmas;

        // std::cout << "cres: " << std::endl << c_res.format(kLongFmt) << std::endl;
        res.cov_inv.template topLeftCorner<9,9>() =
            dse3t1t2v_dt1 * res.c_integration *
            dse3t1t2v_dt1.transpose();

        // Eigen::Matrix<Scalar, ImuResidual::kResSize, 1> diag = res.cov_inv.diagonal();
        // res.cov_inv = diag.asDiagonal();

        // res.cov_inv.setIdentity();

        // StreamMessage(debug_level + 1) << "cov:" << std::endl <<
        //                                   res.cov_inv << std::endl;
        res.cov_inv = res.cov_inv.inverse();
        if (!pose1.is_active || pose2.is_active) {
          // res.cov_inv *= 1000;
        }

        // bias jacbian, only if bias in the state.
        if (BaType::kBiasInState) {
          // Transform the bias jacobian for position and rotation through the
          // jacobian of multiplication by t_2w and the log
          // dt/dB
          res.dz_db.template topLeftCorner<6, 6>() = dlogt1t2_dt1 *
              res.dintegration_db.template topLeftCorner<7, 6>();

          // dV/dB
          res.dz_db.template block<3,6>(6,0) =
              res.dintegration_db.template block<3,6>(7,0);

          // dB/dB
          res.dz_db.template block<6,6>(9,0) =
              Eigen::Matrix<Scalar,6,6>::Identity();

          // The jacboian of the pose error wrt the biases.
          res.dz_dx1.template block<BaType::ImuResidual::kResSize,6>(0,9) =
              res.dz_db;
          // The process model jacobian of the biases.
          res.dz_dx2.template block<6,6>(9,9) =
              -Eigen::Matrix<Scalar,6,6>::Identity();

          // write the residual
          res.residual.template segment<6>(9) = pose1.b - pose2.b;
        }

        if (BaType::kGravityInCalib) {
          const Eigen::Matrix<Scalar,3,2> d_gravity =
              dGravity_dDirection(tracker.imu_.g);
          res.dz_dg.template block<3,2>(0,0) =
              -0.5*powi(total_dt,2) *
              BaType::Matrix3t::Identity() * d_gravity;

          res.dz_dg.template block<3,2>(6,0) =
              -total_dt *
              BaType::Matrix3t::Identity() * d_gravity;
        }

        // _Test_dImuResidual_dX<Scalar, ImuResidual::kResSize, kPoseDim>(
        //       pose1, pose2, imu_pose, res, gravity, dse3_dx1, jb_q, imu_);
        // This is used to calculate the robust norm.
        res.mahalanobis_distance =
            res.residual.transpose() * res.cov_inv * res.residual;
        errors.push_back(res.mahalanobis_distance);
      }
    }
  };
}
