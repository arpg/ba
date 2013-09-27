#include <ba/BundleAdjuster.h>


namespace ba {
#define DAMPING 1.0

////////////////////////////////////////////////////////////////////////////////
template< typename Scalar,int kLmDim, int kPoseDim, int kCalibDim >
void BundleAdjuster<Scalar,kLmDim,kPoseDim,kCalibDim>::ApplyUpdate(
    const VectorXt& delta_p, const VectorXt& delta_l,
    const VectorXt& delta_calib, const bool do_rollback)
{
  double coef = do_rollback == true ? -1.0 : 1.0;
  // update gravity terms if necessary
  if (inertial_residuals_.size() > 0) {
    const VectorXt delta_calib = delta_p.template tail(kCalibDim);
    if (kCalibDim > 0) {
      // std::cout << "Gravity delta is " <<
      // deltaCalib.template block<2,1>(0,0).transpose() <<
      // " gravity is: " << imu_.g.transpose() << std::endl;
    }

    if (kCalibDim > 2) {
      const auto& update = delta_calib.template block<6,1>(2,0)*coef;
      imu_.t_vs = SE3t::exp(update)*imu_.t_vs;
      rig_.cameras[0].T_wc = imu_.t_vs;
      std::cout << "Tvs delta is " << (update).transpose() << std::endl;
      std::cout << "Tvs is :" << std::endl << imu_.t_vs.matrix() << std::endl;
    }
  }

  // update the camera parameters
  if( kCalibDim > 8 && delta_calib.rows() > 8){
    const auto& update = delta_calib.template block<5,1>(8,0)*coef;

    std::cout << "calib delta: " << (update).transpose() << std::endl;

    const VectorXt params = rig_.cameras[0].camera.GenericParams();
    rig_.cameras[0].camera.SetGenericParams(params-(update*coef));

    std::cout << "new params: " <<
                 rig_.cameras[0].camera.GenericParams().transpose() <<
                 std::endl;
  }

  // update poses
  // std::cout << "Updating " << uNumPoses << " active poses." << std::endl;
  for (size_t ii = 0 ; ii < poses_.size() ; ++ii) {
    // only update active poses, as inactive ones are not part of the
    // optimization
    if (poses_[ii].is_active) {
      const unsigned int p_offset = poses_[ii].opt_id*kPoseDim;
      const auto& p_update =
          -delta_p.template block<6,1>(p_offset,0)*coef;
      if (kCalibDim > 2 && inertial_residuals_.size() > 0) {
        const auto& calib_update =
            -delta_calib.template block<6,1>(2,0)*coef;

        if (do_rollback == false) {
          poses_[ii].t_wp = poses_[ii].t_wp * SE3t::exp(p_update);
          poses_[ii].t_wp = poses_[ii].t_wp * SE3t::exp(calib_update);
        } else {
          poses_[ii].t_wp = poses_[ii].t_wp * SE3t::exp(calib_update);
          poses_[ii].t_wp = poses_[ii].t_wp * SE3t::exp(p_update);
        }
        std::cout << "Pose " << ii << " calib delta is " <<
                     (calib_update).transpose() << std::endl;
        poses_[ii].t_vs = imu_.t_vs;
      } else {
        poses_[ii].t_wp = poses_[ii].t_wp * SE3t::exp(p_update);
      }

      // update the velocities if they are parametrized
      if (kPoseDim >= 9) {
        poses_[ii].v_w -=
            delta_p.template block<3,1>(p_offset+6,0)*coef;
            // std::cout << "Velocity for pose " << ii << " is " <<
            // m_vPoses[ii].v_w.transpose() << std::endl;
      }

      if (kPoseDim >= 15) {
        poses_[ii].b -=
            delta_p.template block<6,1>(p_offset+9,0)*coef;
            // std::cout << "Velocity for pose " << ii << " is " <<
            // m_vPoses[ii].v_w.transpose() << std::endl;
      }

      if (kPoseDim >= 21) {
        const auto& tvs_update =
            delta_p.template block<6,1>(p_offset+15,0)*coef;
        poses_[ii].t_vs = SE3t::exp(tvs_update)*poses_[ii].t_vs;
        poses_[ii].t_wp = poses_[ii].t_wp * SE3t::exp(-tvs_update);
        std::cout << "Tvs of pose " << ii << " after update " <<
                     (tvs_update).transpose() << " is " << std::endl <<
                     poses_[ii].t_vs.matrix() << std::endl;
      }

      // std::cout << "Pose delta for " << ii << " is " <<
      //             (-delta_p.template block<kPoseDim,1>(p_offset,0)*
      //              coef).transpose() << std::endl;
    } else {
      // std::cout << " Pose " << ii << " is inactive." << std::endl;
      if (kCalibDim > 2 && inertial_residuals_.size() > 0) {
        const auto& delta_twp = -delta_calib.template block<6,1>(2,0)*coef;
        poses_[ii].t_wp = poses_[ii].t_wp * SE3t::exp(delta_twp);
        std::cout << "INACTIVE POSE " << ii << " calib delta is " <<
                     (delta_twp).transpose() << std::endl;
        poses_[ii].t_vs = imu_.t_vs;
      }
    }

    // clear the vector of Tsw values as they will need to be recalculated
    poses_[ii].t_sw.clear();
  }

  // update the landmarks
  for (size_t ii = 0 ; ii < landmarks_.size() ; ++ii) {
    if (landmarks_[ii].is_active) {
      const auto& lm_delta =
        delta_l.template segment<kLmDim>(landmarks_[ii].opt_id*kLmDim)*coef;
      if (kLmDim == 1) {
        landmarks_[ii].x_s.template tail<kLmDim>() -= lm_delta;
        // std::cout << "Delta for landmark " << ii << " is " <<
        // lm_delta.transpose() << std::endl;
        // m_vLandmarks[ii].Xs /= m_vLandmarks[ii].Xs[3];
      } else {
        landmarks_[ii].x_s.template head<kLmDim>() -= lm_delta;
      }

        // std::cout << "Adjusting landmark with zref: " <<
        // m_vLandmarks[ii].Zref.transpose() << " from " <<
        // m_vLandmarks[ii].Xs.transpose() << std::endl;

        // m_vLandmarks[ii].Xs /= m_vLandmarks[ii].Xs[3];
        // const Scalar depth = m_vLandmarks[ii].Xs.template head<3>().norm();
        // reproject this landmark
        // VectorXt origParams = m_Rig.cameras[0].camera.GenericParams();
        // m_Rig.cameras[0].camera.SetGenericParams(
        //    m_vPoses[m_vLandmarks[ii].RefPoseId].CamParams);
        // Vector3t Xs_reproj =
        //    m_Rig.cameras[0].camera.Unproject(m_vLandmarks[ii].Zref);
        // m_vLandmarks[ii].Xs.template head<3>() = Xs_reproj*depth;
        // m_Rig.cameras[0].camera.SetGenericParams(origParams);
        // std::cout << "to " << m_vLandmarks[ii].Xs.transpose() << std::endl;

      landmarks_[ii].x_w =
          MultHomogeneous(poses_[landmarks_[ii].ref_pose_id].GetTsw(
              landmarks_[ii].ref_cam_id, rig_, kPoseDim >= 21).inverse(),
              landmarks_[ii].x_s);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
template< typename Scalar,int kLmDim, int kPoseDim, int kCalibDim >
void BundleAdjuster<Scalar,kLmDim,kPoseDim,kCalibDim>::EvaluateResiduals()
{
  proj_error_ = 0;
  for (ProjectionResidual& res : proj_residuals_) {
    Landmark& lm = landmarks_[res.landmark_id];
    Pose& pose = poses_[res.x_meas_id];
    Pose& ref_pose = poses_[res.x_ref_id];
    const SE3t t_sw_m =
        pose.GetTsw(res.cam_id, rig_, kPoseDim >= 21);
    const SE3t t_ws_r =
        ref_pose.GetTsw(lm.ref_cam_id,rig_, kPoseDim >= 21).inverse();

    const Vector2t p = rig_.cameras[res.cam_id].camera.Transfer3D(
          t_sw_m*t_ws_r, lm.x_s.template head<3>(),lm.x_s(3));

    res.residual = res.z - p;
    proj_error_ += res.residual.norm() * res.weight;
  }

  binary_error_ = 0;
  for (BinaryResidual& res : binary_residuals_) {
    const Pose& pose1 = poses_[res.x1_id];
    const Pose& pose2 = poses_[res.x2_id];
    res.residual = SE3t::log(pose1.t_wp*res.t_ab*pose2.t_wp.inverse());
    binary_error_ += res.residual.norm() * res.weight;
  }

  inertial_error_ = 0;
  double total_tvs_change = 0;
  for (ImuResidual& res : inertial_residuals_) {
    // set up the initial pose for the integration
    const Vector3t gravity = GetGravityVector(imu_.g);

    const Pose& pose1 = poses_[res.pose1_id];
    const Pose& pose2 = poses_[res.pose2_id];

    // Eigen::Matrix<Scalar,10,10> jb_y;
    const ImuPose imu_pose = ImuResidual::IntegrateResidual(
          pose1,res.measurements,pose1.b.template head<3>(),
          pose1.b.template tail<3>(),gravity,res.poses);

    const SE3t t_ab = pose1.t_wp.inverse()*imu_pose.t_wp;
    const SE3t& t_wa = pose1.t_wp;
    const SE3t& t_wb = pose2.t_wp;

    res.residual.setZero();
    res.residual.template head<6>() = SE3t::log(t_wa*t_ab*t_wb.inverse());
    res.residual.template segment<3>(6) = imu_pose.v_w - pose2.v_w;
    res.residual.template segment<6>(9) = pose1.b - pose2.b;

    if (kCalibDim > 2 || kPoseDim > 15) {
      // disable imu translation error
      res.residual.template head<3>().setZero();
      res.residual.template segment<3>(6).setZero(); // velocity error
      res.residual.template segment<6>(9).setZero(); // bias
    }

    if (kPoseDim > 15) {
      res.residual.template segment<6>(15) =
          SE3t::log(pose1.t_vs*pose2.t_vs.inverse());

      if (translation_enabled_ == false) {
        total_tvs_change += res.residual.template segment<6>(15).norm();

      }
      res.residual.template segment<3>(15).setZero();
    }

    // std::cout << "EVALUATE imu res between " << res.PoseAId << " and " <<
    // res.PoseBId << ":" << res.Residual.transpose () << std::endl;
    inertial_error_ += res.residual.norm() * res.weight;
  }

  if (inertial_residuals_.size() > 0 && translation_enabled_ == false) {
    if (kCalibDim > 2) {
      const Scalar log_dif =
          SE3t::log(imu_.t_vs * last_tvs_.inverse()).norm();

      std::cout << "logDif is " << log_dif << std::endl;
      if (log_dif < 0.01 && poses_.size() >= 30) {
        std::cout << "EMABLING TRANSLATION ERRORS" << std::endl;
        translation_enabled_ = true;
      }
      last_tvs_ = imu_.t_vs;
    }

    if (kPoseDim > 15) {
      std::cout << "Total tvs change is: " << total_tvs_change << std::endl;
      if (total_tvs_change_ != 0 &&
          total_tvs_change/inertial_residuals_.size() < 0.1 &&
          poses_.size() >= 30) {
        std::cout << "EMABLING TRANSLATION ERRORS" << std::endl;
        translation_enabled_ = true;
        total_tvs_change = 0;
      }
      total_tvs_change_ = total_tvs_change;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
template< typename Scalar,int kLmDim, int kPoseDim, int kCalibDim >
void BundleAdjuster<Scalar,kLmDim,kPoseDim,kCalibDim>::Solve(
    const unsigned int uMaxIter)
{
  if( proj_residuals_.empty() && binary_residuals_.empty() &&
      unary_residuals_.empty() && inertial_residuals_.empty()) {
    return;
  }

  for (unsigned int kk = 0 ; kk < uMaxIter ; ++kk) {
    StartTimer(_BuildProblem_);
    BuildProblem();
    PrintTimer(_BuildProblem_);    ;

    const unsigned int num_poses = num_active_poses_;
    const unsigned int num_pose_params = num_poses*kPoseDim;
    const unsigned int num_lm = num_active_landmarks_;   

    StartTimer(_steup_problem_);
    StartTimer(_rhs_mult_);
    // calculate bp and bl
    VectorXt bp(num_pose_params);
    VectorXt bk;
    VectorXt bl;
    BlockMat< Eigen::Matrix<Scalar, kLmDim, kLmDim>> vi(num_lm, num_lm);

    VectorXt rhs_p(num_pose_params + kCalibDim);
    BlockMat< Eigen::Matrix<Scalar, kLmDim, kPrPoseDim>>
        jt_l_j_pr(num_lm, num_poses);

    BlockMat< Eigen::Matrix<Scalar, kPrPoseDim, kLmDim>>
        jt_pr_j_l_vi(num_poses, num_lm);

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> s(
          num_pose_params+kCalibDim, num_pose_params+kCalibDim);

    PrintTimer(_rhs_mult_);


    StartTimer(_jtj_);
    // TODO: suboptimal, the matrices are symmetric. We should only
    // multipl one half
    BlockMat<Eigen::Matrix<Scalar, kPoseDim, kPoseDim>> u(
          num_poses, num_poses);

    vi.setZero();
    u.setZero();
    bp.setZero();
    s.setZero();
    rhs_p.setZero();

    if (proj_residuals_.size() > 0) {
      BlockMat< Eigen::Matrix<Scalar, kPrPoseDim, kPrPoseDim>> jt_pr_j_pr(
            num_poses, num_poses);
      Eigen::SparseBlockProduct(jt_pr, j_pr_, jt_pr_j_pr, true);

      auto temp_u = u;
      // this is a block add, as jt_pr_j_pr does not have the same block
      // dimensions as u, due to efficiency
      Eigen::template SparseBlockAdd(temp_u, jt_pr_j_pr, u);

      VectorXt jt_pr_r_pr(num_pose_params);
      // this is a strided multiplication, as jt_pr_r_pr might have a larger
      // pose dimension than jt_pr (for efficiency)
      Eigen::SparseBlockVectorProductDenseResult(jt_pr, r_pr_, jt_pr_r_pr,
                                                 -1, kPoseDim);
      bp += jt_pr_r_pr;
    }

    // add the contribution from the binary terms if any
    if (binary_residuals_.size() > 0) {
      BlockMat< Eigen::Matrix<Scalar, kPoseDim, kPoseDim> > jt_pp_j_pp(
            num_poses, num_poses);

      Eigen::SparseBlockProduct(jt_pp_ ,j_pp_, jt_pp_j_pp, true);
      auto temp_u = u;
      Eigen::SparseBlockAdd(temp_u,jt_pp_j_pp,u);

      VectorXt jt_pp_r_pp(num_pose_params);
      Eigen::SparseBlockVectorProductDenseResult(jt_pp_, r_pp_, jt_pp_r_pp);
      bp += jt_pp_r_pp;
    }

    // add the contribution from the unary terms if any
    if (unary_residuals_.size() > 0) {
      BlockMat< Eigen::Matrix<Scalar, kPoseDim, kPoseDim> > jt_u_j_u(
            num_poses, num_poses);

      Eigen::SparseBlockProduct(jt_u_, j_u_, jt_u_j_u);
      auto temp_u = u;
      Eigen::SparseBlockAdd(temp_u, jt_u_j_u, u);

      VectorXt jt_u_r_u(num_pose_params);
      Eigen::SparseBlockVectorProductDenseResult(jt_u_, r_u_, jt_u_r_u);
      bp += jt_u_r_u;
    }

    // add the contribution from the imu terms if any
    if (inertial_residuals_.size() > 0) {
      BlockMat< Eigen::Matrix<Scalar, kPoseDim, kPoseDim> > jt_i_j_i(
            num_poses, num_poses);

      Eigen::SparseBlockProduct(jt_i_, j_i_, jt_i_j_i);
      auto temp_u = u;
      Eigen::SparseBlockAdd(temp_u, jt_i_j_i, u);

      VectorXt jt_i_r_i(num_pose_params);
      Eigen::SparseBlockVectorProductDenseResult(jt_i_, r_i_, jt_i_r_i);
      bp += jt_i_r_i;      
    }
    PrintTimer(_jtj_);

    StartTimer(_schur_complement_);
    if (kLmDim > 0 && num_lm > 0) {
      bl.resize(num_lm*kLmDim);
      bl.setZero();
      StartTimer(_schur_complement_v);
      for (int ii = 0; ii < landmarks_.size() ; ++ii) {
        Eigen::Matrix<Scalar,kLmDim,kLmDim> jtj;
        Eigen::Matrix<Scalar,kLmDim,1> jtr;
        jtj.setZero();
        jtr.setZero();
        for (const int id : landmarks_[ii].proj_residuals) {
          const ProjectionResidual& res = proj_residuals_[id];
          jtj += (res.dz_dlm.transpose() * res.dz_dlm) * res.weight;
          jtr += (res.dz_dlm.transpose() *
                  r_pr_.template block<ProjectionResidual::kResSize,1>(
                    res.residual_id*ProjectionResidual::kResSize, 0) *
                    res.weight);
        }
        bl.template block<kLmDim,1>(landmarks_[ii].opt_id*kLmDim, 0) = jtr;
        if (kLmDim == 1) {
          if (fabs(jtj(0,0)) < 1e-6) {
            jtj(0,0) += 1e-6;
          }
        }

        vi.insert(landmarks_[ii].opt_id, landmarks_[ii].opt_id) = jtj.inverse();
      }

      PrintTimer(_schur_complement_v);

      StartTimer(_schur_complement_jtpr_jl);
      BlockMat< Eigen::Matrix<Scalar, kPrPoseDim, kLmDim> > jt_pr_j_l(
            num_poses, num_lm);

      Eigen::SparseBlockProduct(jt_pr,j_l_,jt_pr_j_l);
      decltype(jt_l_j_pr)::forceTranspose(jt_pr_j_l, jt_l_j_pr);
      PrintTimer(_schur_complement_jtpr_jl);

      // attempt to solve for the poses. W_V_inv is used later on,
      // so we cache it
      StartTimer(_schur_complement_jtpr_jl_vi);
      Eigen::SparseBlockDiagonalRhsProduct(jt_pr_j_l, vi, jt_pr_j_l_vi);
      PrintTimer(_schur_complement_jtpr_jl_vi);


      StartTimer(_schur_complement_jtpr_jl_vi_jtl_jpr);
      BlockMat< Eigen::Matrix<Scalar, kPrPoseDim, kPrPoseDim>>
            jt_pr_j_l_vi_jt_l_j_pr(num_poses, num_poses);

      Eigen::SparseBlockProduct(jt_pr_j_l_vi, jt_l_j_pr,
                                jt_pr_j_l_vi_jt_l_j_pr, true);
      PrintTimer(_schur_complement_jtpr_jl_vi_jtl_jpr);

      //StartTimer(_schur_complement_jtpr_jl_vi_jtl_jpr_d);
      //MatrixXt djt_pr_j_l_vi(
      //      jt_pr_j_l_vi.rows()*kPoseDim,jt_pr_j_l_vi.cols()*kLmDim);
      //Eigen::LoadDenseFromSparse(jt_pr_j_l_vi,djt_pr_j_l_vi);

      //MatrixXt djt_l_j_pr(
      //      jt_l_j_pr.rows()*kLmDim,jt_l_j_pr.cols()*kPoseDim);
      //Eigen::LoadDenseFromSparse(jt_l_j_pr,djt_l_j_pr);

      //MatrixXt djt_pr_j_l_vi_jt_l_j_pr = djt_pr_j_l_vi * djt_l_j_pr;
      //PrintTimer(_schur_complement_jtpr_jl_vi_jtl_jpr_d);

      // this in-place operation should be fine for subtraction
      // schur_time = Tic();
      // MatrixXt du(u.rows()*kPoseDim,u.cols()*kPoseDim);
      // Eigen::LoadDenseFromSparse(u,du);

      /*std::cout << "Jp sparsity structure: " << std::endl <<
                     m_Jpr.GetSparsityStructure().format(cleanFmt) << std::endl;
      std::cout << "Jprt sparsity structure: " << std::endl <<
                    m_Jprt.GetSparsityStructure().format(cleanFmt) << std::endl;
      std::cout << "U sparsity structure: " << std::endl <<
                    U.GetSparsityStructure().format(cleanFmt) << std::endl;
      std::cout << "dU " << std::endl <<
                    dU << std::endl;
      */

       Eigen::SparseBlockSubtractDenseResult(
             u, jt_pr_j_l_vi_jt_l_j_pr,
             s.template block(
               0, 0, num_pose_params,
               num_pose_params ));

      // std::cout << "dU matrix is " << dU.format(cleanFmt) << std::endl;
      //s.template block(0, 0, num_pose_params, num_pose_params ) =
      //    du - djt_pr_j_l_vi_jt_l_j_pr;


      // now form the rhs for the pose equations
      VectorXt jt_pr_j_l_vi_bll(num_pose_params);
      Eigen::SparseBlockVectorProductDenseResult(
            jt_pr_j_l_vi, bl, jt_pr_j_l_vi_bll, -1, kPoseDim);

      rhs_p.template head(num_pose_params) = bp - jt_pr_j_l_vi_bll;

      // std::cout << "Dense S matrix is " << s.format(kCleanFmt) << std::endl;
       // std::cout << "Dense rhs matrix is " <<
       // rhs_p.transpose().format(kCleanFmt) << std::endl;

    } else {
      Eigen::LoadDenseFromSparse(
            u, s.template block(0, 0, num_pose_params, num_pose_params));
      rhs_p.template head(num_pose_params) = bp;
    }
    PrintTimer(_schur_complement_);

    // std::cout << "  Rhs calculation and schur complement took " <<
    // Toc(dMatTime) << " seconds." << std::endl;

    // fill in the calibration components if any
    if (kCalibDim && inertial_residuals_.size() > 0) {
      BlockMat<Eigen::Matrix<Scalar,kCalibDim,kCalibDim>> jt_ki_j_ki(1, 1);
      Eigen::SparseBlockProduct(jt_ki_, j_ki_, jt_ki_j_ki);
      Eigen::LoadDenseFromSparse(
            jt_ki_j_ki, s.template block<kCalibDim, kCalibDim>(
              num_pose_params, num_pose_params));

      BlockMat<Eigen::Matrix<Scalar, kPoseDim, kCalibDim>>
            jt_i_j_ki(num_poses, 1);

      Eigen::SparseBlockProduct(jt_i_, j_ki_, jt_i_j_ki);
      Eigen::LoadDenseFromSparse(
            jt_i_j_ki,
            s.template block(0, num_pose_params, num_pose_params, kCalibDim));

      s.template block(num_pose_params, 0, kCalibDim, num_pose_params) =
          s.template block(0, num_pose_params,
                           num_pose_params, kCalibDim).transpose();

      // and the rhs for the calibration params
      bk.resize(kCalibDim,1);
      Eigen::SparseBlockVectorProductDenseResult(jt_ki_, r_i_, bk);
      rhs_p.template tail<kCalibDim>() = bk;
    }

    if( kCalibDim > 8){
      /*BlockMat< Eigen::Matrix<Scalar, kCalibDim, kCalibDim>> jt_kpr_j_kpr(1, 1);
      Eigen::SparseBlockProduct(jt_kpr_, j_kpr_, jt_kpr_j_kpr);
      MatrixXt djt_kpr_j_kpr(kCalibDim, kCalibDim);
      Eigen::LoadDenseFromSparse(jt_kpr_j_kpr, djt_kpr_j_kpr);
      s.template block<kCalibDim, kCalibDim>(num_pose_params, num_pose_params)
          += djt_kpr_j_kpr;
      std::cout << "djt_kpr_j_kpr: " << djt_kpr_j_kpr << std::endl;

      BlockMat<Eigen::Matrix<Scalar, kPoseDim, kCalibDim>>
          jt_or_j_kpr(num_poses, 1);

      Eigen::SparseBlockProduct(jt_pr, j_kpr_, jt_or_j_kpr);
      MatrixXt djt_or_j_kpr(kPoseDim*num_poses, kCalibDim);
      Eigen::LoadDenseFromSparse(jt_or_j_kpr, djt_or_j_kpr);
      std::cout << "djt_or_j_kpr: " << djt_or_j_kpr << std::endl;
      s.template block(0, num_pose_params, num_pose_params, kCalibDim) +=
          djt_or_j_kpr;
      s.template block(num_pose_params, 0, kCalibDim, num_pose_params) +=
          djt_or_j_kpr.transpose();

      bk.resize(kCalibDim,1);
      Eigen::SparseBlockVectorProductDenseResult(jt_kpr_, r_pr_, bk);
      rhs_p.template tail<kCalibDim>() += bk;

      // schur complement
      BlockMat< Eigen::Matrix<Scalar, kCalibDim, kLmDim>> jt_kpr_jl(1, num_lm);
      BlockMat< Eigen::Matrix<Scalar, kLmDim, kCalibDim>> jt_l_j_kpr(num_lm, 1);
      Eigen::SparseBlockProduct(jt_kpr_,j_l_,jt_kpr_jl);
      Eigen::SparseBlockProduct(jt_l_,j_kpr_,jt_l_j_kpr);
      //Jlt_Jkpr = Jkprt_Jl.transpose();

      MatrixXt djt_pr_j_l_vi_jt_l_j_kpr(kPoseDim*num_poses, kCalibDim);
      BlockMat<Eigen::Matrix<Scalar, kPoseDim, kCalibDim>>
          jt_pr_j_l_vi_jt_l_j_kpr(num_poses, 1);

      Eigen::SparseBlockProduct(
            jt_pr_j_l_vi,jt_l_j_kpr, jt_pr_j_l_vi_jt_l_j_kpr);
      Eigen::LoadDenseFromSparse(
            jt_pr_j_l_vi_jt_l_j_kpr, djt_pr_j_l_vi_jt_l_j_kpr);

      std::cout << "jt_pr_j_l_vi_jt_l_j_kpr: " <<
                   djt_pr_j_l_vi_jt_l_j_kpr << std::endl;

      s.template block(0, num_pose_params, num_pose_params, kCalibDim) -=
          djt_pr_j_l_vi_jt_l_j_kpr;
      s.template block(num_pose_params, 0, kCalibDim, num_pose_params) -=
          djt_pr_j_l_vi_jt_l_j_kpr.transpose();

      BlockMat<Eigen::Matrix<Scalar, kCalibDim, kLmDim>>
          jt_kpr_j_l_vi(1, num_lm);
      Eigen::SparseBlockProduct(jt_kpr_jl,vi,jt_kpr_j_l_vi);

      BlockMat<Eigen::Matrix<Scalar, kCalibDim, kCalibDim>>
          jt_kpr_j_l_vi_jt_l_j_kpr(1, 1);
      Eigen::SparseBlockProduct(
            jt_kpr_j_l_vi,
            jt_l_j_kpr,
            jt_kpr_j_l_vi_jt_l_j_kpr);

      MatrixXt djt_kpr_j_l_vi_jt_l_j_kpr(kCalibDim, kCalibDim);
      Eigen::LoadDenseFromSparse(
            jt_kpr_j_l_vi_jt_l_j_kpr,
            djt_kpr_j_l_vi_jt_l_j_kpr);

      std::cout << "djt_kpr_j_l_vi_jt_l_j_kpr: " <<
                   djt_kpr_j_l_vi_jt_l_j_kpr << std::endl;

      s.template block<kCalibDim, kCalibDim>(num_pose_params, num_pose_params)
          -= djt_kpr_j_l_vi_jt_l_j_kpr;

      VectorXt jt_kpr_j_l_vi_bl;
      jt_kpr_j_l_vi_bl.resize(kCalibDim);
      Eigen::SparseBlockVectorProductDenseResult(
            jt_kpr_j_l_vi, bl, jt_kpr_j_l_vi_bl);
      // std::cout << "Eigen::SparseBlockVectorProductDenseResult(Wp_V_inv, bl,"
      // " WV_inv_bl) took  " << Toc(dSchurTime) << " seconds."  << std::endl;

      std::cout << "jt_kpr_j_l_vi_bl: " <<
                   jt_kpr_j_l_vi_bl.transpose() << std::endl;
      std::cout << "rhs_p.template tail<kCalibDim>(): " <<
                   rhs_p.template tail<kCalibDim>().transpose() << std::endl;
      rhs_p.template tail<kCalibDim>() -= jt_kpr_j_l_vi_bl;
      */
    }
    PrintTimer(_steup_problem_);

    // now we have to solve for the pose constraints
    StartTimer(_solve_);
    // Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>, Eigen::Upper> solver;
    // Eigen::SparseMatrix<Scalar> s_sparse = s.sparseView();
    // solver.compute(s_sparse);
    VectorXt delta_p = num_poses == 0 ? VectorXt() : s.ldlt().solve(rhs_p);
    //VectorXt delta_p = num_poses == 0 ? VectorXt() : solver.solve(rhs_p);
    PrintTimer(_solve_);

    StartTimer(_back_substitution_);
    VectorXt delta_l;
    if (num_lm > 0) {
      delta_l.resize(num_lm*kLmDim);
      VectorXt jt_l_j_pr_delta_p;
      jt_l_j_pr_delta_p.resize(num_lm*kLmDim );
      // this is the strided multiplication as delta_p has all pose parameters,
      // however jt_l_j_pr_delta_p is only with respect to the 6 pose parameters
      Eigen::SparseBlockVectorProductDenseResult(
            jt_l_j_pr,
            delta_p.head(num_pose_params),
            jt_l_j_pr_delta_p,
            kPoseDim, -1);

      VectorXt rhs_l;
      rhs_l.resize(num_lm*kLmDim );
      rhs_l =  bl - jt_l_j_pr_delta_p;

      for (size_t ii = 0 ; ii < num_lm ; ++ii) {
        delta_l.template block<kLmDim,1>( ii*kLmDim, 0 ).noalias() =
            vi.coeff(ii,ii)*rhs_l.template block<kLmDim,1>(ii*kLmDim,0);
        //std::cout << "Lm " << ii << " delta is " <<
        //      delta_l.template block<kLmDim,1>(ii*kLmDim, 0).transpose() <<
        //            std::endl;
      }
    }
    PrintTimer(_back_substitution_);

    VectorXt deltaCalib;
    if (kCalibDim > 0 && num_pose_params > 0) {
      deltaCalib = delta_p.template tail(kCalibDim);
      // std::cout << "Delta calib: " << deltaCalib.transpose() << std::endl;
    }

    ApplyUpdate(delta_p, delta_l, deltaCalib, false);

    const double dPrevError = proj_error_ + inertial_error_ + binary_error_;
    //std::cout << "Pre-solve norm: " << dPrevError << " with Epr:" <<
    //             proj_error_ << " and Ei:" << inertial_error_ <<
    //             " and Epp: " << binary_error_ << std::endl;
    EvaluateResiduals();
    const double dPostError = proj_error_ + inertial_error_ + binary_error_;
    //std::cout << "Post-solve norm: " << dPostError << " with Epr:" <<
    //              proj_error_ << " and Ei:" << inertial_error_ <<
    //             " and Epp: " << binary_error_ << std::endl;

    if (dPostError > dPrevError) {
      // std::cout << "Error increasing during optimization, rolling back .."<<
      //             std::endl;
      ApplyUpdate(delta_p, delta_l, deltaCalib, true);
      break;
    }
    else if ((dPrevError - dPostError)/dPrevError < 0.01) {
      //std::cout << "Error decrease less than 1%, aborting." << std::endl;
      //break;
    }
      //std::cout << "BA iteration " << kk <<  " error: " << dPostError <<
      //            std::endl;
  }


  if (kPoseDim >= 15 && poses_.size() > 0) {
    imu_.b_g = poses_.back().b.template head<3>();
    imu_.b_a = poses_.back().b.template tail<3>();
  }

  if (kPoseDim >= 21 && poses_.size() > 0) {
    imu_.t_vs = poses_.back().t_vs;
  }
  // std::cout << "Solve took " << Toc(dTime) << " seconds." << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
template< typename Scalar,int kLmDim, int kPoseDim, int kCalibDim >
void BundleAdjuster<Scalar, kLmDim, kPoseDim, kCalibDim>::BuildProblem()
{
  // resize as needed
  const unsigned int num_poses = num_active_poses_;
  const unsigned int num_lm = num_active_landmarks_;
  const unsigned int num_proj_res = proj_residuals_.size();
  const unsigned int num_bin_res = binary_residuals_.size();
  const unsigned int num_un_res = unary_residuals_.size();
  const unsigned int num_im_res= inertial_residuals_.size();

  j_pr_.resize(num_proj_res, num_poses);
  jt_pr.resize(num_poses, num_proj_res);
  j_kpr_.resize(num_proj_res, 1);
  jt_kpr_.resize(1, num_proj_res);
  j_l_.resize(num_proj_res, num_lm);
  // jt_l_.resize(num_lm, num_proj_res);
  r_pr_.resize(num_proj_res*ProjectionResidual::kResSize);

  j_pp_.resize(num_bin_res, num_poses);
  jt_pp_.resize(num_poses, num_bin_res);
  r_pp_.resize(num_bin_res*BinaryResidual::kResSize);

  j_u_.resize(num_un_res, num_poses);
  jt_u_.resize(num_poses, num_un_res);
  r_u_.resize(num_un_res*UnaryResidual::kResSize);

  j_i_.resize(num_im_res, num_poses);
  jt_i_.resize(num_poses, num_im_res);
  j_ki_.resize(num_im_res, 1);
  jt_ki_.resize(1, num_im_res);
  r_i_.resize(num_im_res*ImuResidual::kResSize);


  // these calls remove all the blocks, but KEEP allocated memory as long as
  // the object is alive
  j_pr_.setZero();
  jt_pr.setZero();
  j_kpr_.setZero();
  jt_kpr_.setZero();
  r_pr_.setZero();

  j_pp_.setZero();
  jt_pp_.setZero();
  r_pp_.setZero();

  j_u_.setZero();
  jt_u_.setZero();
  r_u_.setZero();

  j_i_.setZero();
  jt_i_.setZero();
  j_ki_.setZero();
  jt_ki_.setZero();
  r_i_.setZero();

  j_l_.setZero();
  // jt_l_.setZero();


  // used to store errors for robust norm calculation
  errors_.reserve(num_proj_res);
  errors_.clear();

  StartTimer(_j_evaluation_);
  StartTimer(_j_evaluation_proj_);
  proj_error_ = 0;
  for (ProjectionResidual& res : proj_residuals_) {
    // calculate measurement jacobians

    // Tsw = T_cv * T_vw
    Landmark& lm = landmarks_[res.landmark_id];
    Pose& pose = poses_[res.x_meas_id];
    Pose& ref_pose = poses_[res.x_ref_id];
    lm.x_s = MultHomogeneous(
          ref_pose.GetTsw(lm.ref_cam_id, rig_, kPoseDim >= 21) ,lm.x_w);

    auto& cam = rig_.cameras[res.cam_id].camera;

    const SE3t t_vs_m =
        (kPoseDim >= 21 ? pose.t_vs : rig_.cameras[res.cam_id].T_wc);
    const SE3t t_vs_r =
        (kPoseDim >= 21 ? ref_pose.t_vs :  rig_.cameras[res.cam_id].T_wc);
    const SE3t t_sw_m =
        pose.GetTsw(res.cam_id, rig_, kPoseDim >= 21);
    const SE3t t_ws_r =
        ref_pose.GetTsw(lm.ref_cam_id,rig_, kPoseDim >= 21).inverse();

    const Vector2t p = cam.Transfer3D(
          t_sw_m*t_ws_r, lm.x_s.template head<3>(),lm.x_s(3));

    res.residual = res.z - p;

    // this array is used to calculate the robust norm
    errors_.push_back(res.residual.squaredNorm());

    const Eigen::Matrix<Scalar,2,4> dt_dp_s = cam.dTransfer3D_dP(
          t_sw_m*t_ws_r, lm.x_s.template head<3>(),lm.x_s(3));

    // Landmark Jacobian
    if (lm.is_active) {
      res.dz_dlm = -dt_dp_s.template block<2,kLmDim>( 0, kLmDim == 3 ? 0 : 3 );
    }

    // if the measurement and reference poses are the same, the jacobian is zero
    if (res.x_ref_id == res.x_meas_id) {
      res.dz_dx_meas.setZero();
      res.dz_dx_ref.setZero();
    } else if (pose.is_active || ref_pose.is_active) {
      // std::cout << "Calculating j for residual with poseid " << pose.Id <<
      // " and refPoseId " << refPose.Id << std::endl;
      // derivative for the measurement pose
      const Vector4t Xs_m = MultHomogeneous(t_sw_m, lm.x_w);
      const Eigen::Matrix<Scalar,2,4> dt_dp_m = cam.dTransfer3D_dP(
            SE3t(), Xs_m.template head<3>(),Xs_m(3));

      const Eigen::Matrix<Scalar,2,4> dt_dp_m_tsv_m =
          dt_dp_m * t_vs_m.inverse().matrix();

      const Vector4t x_p = pose.t_wp.inverse().matrix() * lm.x_w;
      // this is the multiplication by the lie generators unrolled
      for (unsigned int ii=0; ii<3; ++ii) {
        res.dz_dx_meas.template block<2,1>(0,ii) =
            dt_dp_m_tsv_m.col(ii) * x_p[3];
      }
      res.dz_dx_meas.template block<2,1>(0,3) =
          (dt_dp_m_tsv_m.col(2)*x_p[1] - dt_dp_m_tsv_m.col(1)*x_p[2]);

      res.dz_dx_meas.template block<2,1>(0,4) =
          (-dt_dp_m_tsv_m.col(2)*x_p[0] + dt_dp_m_tsv_m.col(0)*x_p[2]);

      res.dz_dx_meas.template block<2,1>(0,5) =
          (dt_dp_m_tsv_m.col(1)*x_p[0] - dt_dp_m_tsv_m.col(0)*x_p[1]);

      // only need this if we are in inverse depth mode and the poses aren't
      // the same
      if (kLmDim == 1) {
        // derivative for the reference pose
        const Eigen::Matrix<Scalar,2,4> dt_dp_m_tsw_m =
            dt_dp_m * (pose.t_wp * t_vs_m).inverse().matrix();

        const Vector4t x_v = t_vs_r.matrix() * lm.x_s;
        const Eigen::Matrix<Scalar,2,4> dt_dp_m_tsw_m_twp =
            -dt_dp_m_tsw_m * ref_pose.t_wp.matrix();
        // this is the multiplication by the lie generators unrolled
        for (unsigned int ii=0; ii<3; ++ii) {
          res.dz_dx_ref.template block<2,1>(0,ii) =
              dt_dp_m_tsw_m_twp.col(ii) * x_v[3];
        }
        res.dz_dx_ref.template block<2,1>(0,3) =
            (dt_dp_m_tsw_m_twp.col(2)*x_p[1] - dt_dp_m_tsv_m.col(1)*x_v[2]);

        res.dz_dx_ref.template block<2,1>(0,4) =
            (-dt_dp_m_tsw_m_twp.col(2)*x_p[0] + dt_dp_m_tsv_m.col(0)*x_v[2]);

        res.dz_dx_ref.template block<2,1>(0,5) =
            (dt_dp_m_tsw_m_twp.col(1)*x_p[0] - dt_dp_m_tsv_m.col(0)*x_v[1]);

        for (unsigned int ii=3; ii<6; ++ii) {
          res.dz_dx_ref.template block<2,1>(0,ii) =
             dt_dp_m_tsw_m_twp * Sophus::SE3Group<Scalar>::generator(ii) * x_v;
        }

        //Eigen::Matrix<Scalar,2,6> dZ_dPr_fd;
        //for(int ii = 0; ii < 6 ; ii++) {
        //    Eigen::Matrix<Scalar,6,1> delta;
        //    delta.setZero();
        //    delta[ii] = dEps;
        //    SE3t Tss = (pose.t_wp*Tvs_m).inverse() *
        //    (refPose.t_wp*SE3t::exp(delta)) * Tvs_r;
        //
        //    const Vector2t pPlus =
        //    m_Rig.cameras[res.CameraId].camera.Transfer3D(
        //    Tss,lm.Xs.template head(3),lm.Xs[3]);

        //    delta[ii] = -dEps;
        //    Tsw = (pose.t_wp*SE3t::exp(delta)*
        //    m_Rig.cameras[meas.CameraId].T_wc).inverse();
        //
        //    Tss = (pose.t_wp*Tvs_m).inverse() *
        //    (refPose.t_wp*SE3t::exp(delta)) * Tvs_r;
        //
        //    const Vector2t pMinus =
        //    m_Rig.cameras[res.CameraId].camera.
        //    Transfer3D(Tss,lm.Xs.template head(3),lm.Xs[3]);
        //
        //    dZ_dPr_fd.col(ii) = -(pPlus-pMinus)/(2*dEps);
        //}
        //std::cout << "dZ_dPr   :" << res.dZ_dPr << std::endl;
        //std::cout << "dZ_dPr_fd:" << dZ_dPr_fd << " norm: " <<
        //(res.dZ_dPr - dZ_dPr_fd).norm() <<  std::endl;
      }

      // calculate jacobian wrt to camera parameters
      // [TEST]: This is only working for fov models
      // Vector3t Xs_m_norm = Xs_m.template head<3>() / Xs_m[3];
      // const VectorXt params =
      // m_Rig.cameras[res.CameraId].camera.GenericParams();
      //
      // res.dZ_dK =
      // -m_Rig.cameras[res.CameraId].camera.dMap_dParams(Xs_m_norm, params);

      //{
      //double dEps = 1e-9;
      //Eigen::Matrix<Scalar,2,5> dZ_dK_fd;
      //for(int ii = 0; ii < 6 ; ii++) {
      //    Eigen::Matrix<Scalar,5,1> delta;
      //    delta.setZero();
      //    delta[ii] = dEps;
      //    m_Rig.cameras[res.CameraId].camera.SetGenericParams(
      //    params + delta);
      //    const Vector2t pPlus =
      //    -m_Rig.cameras[res.CameraId].camera.Transfer3D(SE3t(),
      //    Xs_m.template head<3>(),Xs_m(3));
      //
      //    delta[ii] = -dEps;
      //    m_Rig.cameras[res.CameraId].camera.SetGenericParams(params + delta);
      //    const Vector2t pMinus =
      //    -m_Rig.cameras[res.CameraId].camera.Transfer3D(
      //    SE3t(), Xs_m.template head<3>(),Xs_m(3));
      //    dZ_dK_fd.col(ii) = (pPlus-pMinus)/(2*dEps);
      //}
      //std::cout << "dZ_dK   :" << std::endl << res.dZ_dK << std::endl;
      //std::cout << "dZ_dK_fd:" << std::endl << dZ_dK_fd << " norm: " <<
      //(res.dZ_dK - dZ_dK_fd).norm() <<  std::endl;
      //m_Rig.cameras[res.CameraId].camera.SetGenericParams(params);
      //}
    }
    // set the residual in m_R which is dense
    r_pr_.template segment<ProjectionResidual::kResSize>(res.residual_offset) =
        res.residual;
  }

  // std::cout << "Max dZ_dX norm: " << maxdZ_dX_norm << " with dZ_dX: " <<
  // std::endl << maxdZ_dX << " with dZ_dX_fd: "  << std::endl <<
  // maxdZ_dX_fd << std::endl;

  // std::cout << "Max dZ_dPm norm: " << maxdZ_dPm_norm << " with error: " <<
  // std::endl << maxdZ_dPm << " with dZ_dPm_fd: " << std::endl  <<
  // maxdZ_dPm_fd <<  std::endl;

  // std::cout << "Max dZ_dPr norm: " << maxdZ_dPr_norm << " with error: " <<
  // std::endl << maxdZ_dPr <<  " with dZ_dPr_fd: " << std::endl  <<
  // maxdZ_dPr_fd  << std::endl;

  // get the sigma for robust norm calculation. This call is O(n) on average,
  // which is desirable over O(nlogn) sort
  if (errors_.size() > 0) {
    auto it = errors_.begin()+std::floor(errors_.size()/2);
    std::nth_element(errors_.begin(),it,errors_.end());
    const Scalar sigma = sqrt(*it);
    // std::cout << "Projection error sigma is " << dSigma << std::endl;
    // See "Parameter Estimation Techniques: A Tutorial with Application to
    // Conic Fitting" by Zhengyou Zhang. PP 26 defines this magic number:
    const Scalar c_huber = 1.2107*sigma;

    // now go through the measurements and assign weights
    for( ProjectionResidual& res : proj_residuals_ ){
      // calculate the huber norm weight for this measurement
      const Scalar e = res.residual.norm();
      res.weight = e > c_huber ? c_huber/e : 1.0;
      proj_error_ += res.residual.norm() * res.weight;
    }
  }
  errors_.clear();
  PrintTimer(_j_evaluation_proj_);

  StartTimer(_j_evaluation_binary_);
  binary_error_ = 0;
  // build binary residual jacobians
  for( BinaryResidual& res : binary_residuals_ ){
    const SE3t& t_w1 = poses_[res.x1_id].t_wp;
    const SE3t& t_w2 = poses_[res.x2_id].t_wp;
    const SE3t t_2w = t_w2.inverse();
    res.dz_dx1 = dLog_dX(t_w1, res.t_ab * t_2w);
    // the negative sign here is because exp(x) is inside the inverse
    // when we invert (Twb*exp(x)).inverse
    res.dz_dx2 = -dLog_dX(t_w1 * res.t_ab, t_2w);

    res.residual = SE3t::log(t_w1*res.t_ab*t_2w);

    // finite difference checking
    //Eigen::Matrix<Scalar,6,6> dz_dx2_fd;
    //Scalar dEps = 1e-10;
    //for (int ii = 0; ii < 6 ; ii++) {
    //  Eigen::Matrix<Scalar,6,1> delta;
    //  delta.setZero();
    //  delta[ii] = dEps;
    //  const Vector6t pPlus =
    //      SE3t::log(t_w1 * res.t_ab * (t_w2*SE3t::exp(delta)).inverse());
    //  delta[ii] = -dEps;
    //  const Vector6t pMinus =
    //      SE3t::log(t_w1 * res.t_ab * (t_w2*SE3t::exp(delta)).inverse());
    //  dz_dx2_fd.col(ii) = (pPlus-pMinus)/(2*dEps);
    //}
    //std::cout << "dz_dx2:" << res.dz_dx2 << std::endl;
    //std::cout << "dz_dx2_fd:" << dz_dx2_fd << std::endl;

    r_pp_.template segment<BinaryResidual::kResSize>(res.residual_offset) =
        res.residual;

    binary_error_ += res.residual.norm() * res.weight;
  }
  PrintTimer(_j_evaluation_binary_);

  StartTimer(_j_evaluation_unary_);
  for( UnaryResidual& res : unary_residuals_ ){
    const SE3t& Twp = poses_[res.pose_id].t_wp;
    // res.dZ_dX = dLog_dX(Twp, res.t_wp.inverse());
    res.dz_dx = dLog_decoupled_dX(Twp, res.t_wp);

    //Eigen::Matrix<Scalar,6,6> J_fd;
    //Scalar dEps = 1e-10;
    //for (int ii = 0; ii < 6 ; ii++) {
    //  Eigen::Matrix<Scalar,6,1> delta;
    //  delta.setZero();
    //  delta[ii] = dEps;
    //  const Vector6t pPlus =
    //    log_decoupled(exp_decoupled(Twp,delta) , res.t_wp);
    //  delta[ii] = -dEps;
    //  const Vector6t pMinus =
    //    log_decoupled(exp_decoupled(Twp,delta) , res.t_wp);
    //  J_fd.col(ii) = (pPlus-pMinus)/(2*dEps);
    //}
    //std::cout << "Junary:" << res.dZ_dX << std::endl;
    //std::cout << "Junary_fd:" << J_fd << std::endl;

    r_u_.template segment<UnaryResidual::kResSize>(res.residual_offset) =
        log_decoupled(Twp, res.t_wp);
  }
  PrintTimer(_j_evaluation_unary_);

  StartTimer(_j_evaluation_inertial_);
  inertial_error_ = 0;
  for (ImuResidual& res : inertial_residuals_) {
    // set up the initial pose for the integration
    const Vector3t gravity = GetGravityVector(imu_.g);

    const Pose& pose1 = poses_[res.pose1_id];
    const Pose& pose2 = poses_[res.pose2_id];

    Eigen::Matrix<Scalar,10,6> jb_q;
    // Eigen::Matrix<Scalar,10,10> jb_y;
    ImuPose imu_pose = ImuResidual::IntegrateResidual(
          pose1,res.measurements,pose1.b.template head<3>(),
          pose1.b.template tail<3>(),gravity,res.poses,&jb_q/*,&jb_y*/);

    Scalar total_dt =
        res.measurements.back().time - res.measurements.front().time;

    const SE3t t_12 = pose1.t_wp.inverse()*imu_pose.t_wp;
    const SE3t& t_w1 = pose1.t_wp;
    const SE3t& t_w2 = pose2.t_wp;
    const SE3t& t_2w = t_w2.inverse();

    // now given the poses, calculate the jacobians.
    // First subtract gravity, initial pose and velocity from the delta T and delta V
    SE3t t_12_0 = imu_pose.t_wp;
    // subtract starting velocity and gravity
    t_12_0.translation() -=
        (-gravity*0.5*powi(total_dt,2) + pose1.v_w*total_dt);
    // subtract starting pose
    t_12_0 = pose1.t_wp.inverse() * t_12_0;
    // Augment the velocity delta by subtracting effects of gravity
    Vector3t v_12_0 = imu_pose.v_w - pose1.v_w;
    v_12_0 += gravity*total_dt;
    // rotate the velocity delta so that it starts from orientation=Ident
    v_12_0 = pose1.t_wp.so3().inverse() * v_12_0;

    // derivative with respect to the start pose
    res.residual.setZero();
    res.dz_dx1.setZero();
    res.dz_dx2.setZero();
    res.dz_dg.setZero();
    res.dz_db.setZero();

    // calculate the derivative of the lie log with
    // respect to the tangent plane at Twa
    const Eigen::Matrix<Scalar,6,7> dlog_dse3 =
        dLog_dSE3(imu_pose.t_wp*t_2w);

    Eigen::Matrix<Scalar,7,6> dse3_dx1;
    dse3_dx1.setZero();
    dse3_dx1.template block<3,3>(0,0) = t_w1.so3().matrix();
    // for this derivation  refer to page 16 of notes
    dse3_dx1.template block<3,3>(0,3) =
        dqx_dq<Scalar>(
          (t_w1).unit_quaternion(),
          t_12_0.translation()-t_12_0.so3()*
          t_2w.so3()*t_w2.translation()) *
        dq1q2_dq2(t_w1.unit_quaternion()) *
        dqExp_dw<Scalar>(Eigen::Matrix<Scalar,3,1>::Zero());

    dse3_dx1.template block<4,3>(3,3) =
        dq1q2_dq1((t_12_0.so3() * t_2w.so3()).unit_quaternion()) *
        dq1q2_dq2(t_w1.unit_quaternion()) *
        dqExp_dw<Scalar>(Eigen::Matrix<Scalar,3,1>::Zero());

    Eigen::Matrix<Scalar,10,6> dt_db = jb_q;
    // for this derivation refer to page 22 of notes
    dt_db.template block<3,3>(0,0) +=
        dqx_dq(imu_pose.t_wp.unit_quaternion(), t_2w.translation())*
        dt_db.template block<4,3>(3, 0);

    dt_db.template block<4,3>(3,0) =
        dq1q2_dq1(t_2w.so3().unit_quaternion())*
        dt_db.template block<4,3>(3,0) ;

    // dt/dB
    res.dz_db.template block<6,6>(0,0) =
        dlog_dse3 * dt_db.template block<7,6>(0,0);

    // dV/dB
    res.dz_db.template block<3,6>(6,0) = dt_db.template block<3,6>(7,0);
    // dB/dB
    res.dz_db.template block<6,6>(9,0) = Eigen::Matrix<Scalar,6,6>::Identity();

    // Twa^-1 is multiplied here as we need the velocity derivative in the
    //frame of pose A, as the log is taken from this frame
    res.dz_dx1.template block<3,3>(0,6) = Matrix3t::Identity()*total_dt;
    for (int ii = 0; ii < 3 ; ++ii) {
      res.dz_dx1.template block<3,1>(6,3+ii) =
          t_w1.so3().matrix() *
          Sophus::SO3Group<Scalar>::generator(ii) * v_12_0;
    }
    res.dz_dx1.template block<3,3>(6,6) = Matrix3t::Identity();
    res.dz_dx1.template block<6,6>(0,0) =  dlog_dse3*dse3_dx1;
    res.dz_dx1.template block<ImuResidual::kResSize,6>(0,9) = res.dz_db;

    // the - sign is here because of the exp(-x) within the log
    res.dz_dx2.template block<6,6>(0,0) =
        -dLog_dX(imu_pose.t_wp,t_2w);

    res.dz_dx2.template block<3,3>(6,6) = -Matrix3t::Identity();
    res.dz_dx2.template block<6,6>(9,9) =
        -Eigen::Matrix<Scalar,6,6>::Identity();

    const Eigen::Matrix<Scalar,3,2> dGravity = dGravity_dDirection(imu_.g);
    res.dz_dg.template block<3,2>(0,0) =
        -0.5*powi(total_dt,2)*Matrix3t::Identity()*dGravity;

    res.dz_dg.template block<3,2>(6,0) =
        -total_dt*Matrix3t::Identity()*dGravity;

    res.residual.template head<6>() = SE3t::log(t_w1*t_12*t_2w);
    res.residual.template segment<3>(6) = imu_pose.v_w - pose2.v_w;
    res.residual.template segment<6>(9) = pose1.b - pose2.b;

    if ((kCalibDim > 2 || kPoseDim > 15) && translation_enabled_ == false) {
      // disable imu translation error
      res.residual.template head<3>().setZero();
      res.residual.template segment<3>(6).setZero(); // velocity error
      res.residual.template segment<6>(9).setZero(); // bias
      res.dz_dx1.template block<3, kPoseDim>(0,0).setZero();
      res.dz_dx2.template block<3, kPoseDim>(0,0).setZero();
      res.dz_dx1.template block<3, kPoseDim>(6,0).setZero();
      res.dz_dx2.template block<3, kPoseDim>(6,0).setZero();
      res.dz_db.template block<3, 6>(0,0).setZero();
      res.dz_dg.template block<3, 2>(0,0).setZero();

      // disable accelerometer and gyro bias
      res.dz_dx1.template block<res.kResSize, 6>(0,9).setZero();
      res.dz_dx2.template block<res.kResSize, 6>(0,9).setZero();
    }

    if (kPoseDim > 15) {
      res.dz_dx1.template block<9,6>(0,15) =
          res.dz_dx1.template block<9, 6>(0,0);
      res.dz_dx2.template block<9,6>(0,15) =
          res.dz_dx2.template block<9, 6>(0,0);

      //Tvs bias residual
      res.residual.template segment<6>(15) =
          SE3t::log(pose1.t_vs*pose2.t_vs.inverse());

      res.dz_dx1.template block<6,6>(15,15) =
          -dLog_dX(SE3t(), pose1.t_vs * pose2.t_vs.inverse());
      res.dz_dx2.template block<6,6>(15,15) =
          dLog_dX(pose1.t_vs * pose2.t_vs.inverse(), SE3t());

      //if( m_bEnableTranslation == false ){
      res.residual.template segment<3>(15).setZero();
      // removing translation elements of Tvs
      res.dz_dx1.template block<6,3>(15,15).setZero();
      res.dz_dx2.template block<6,3>(15,15).setZero();
      res.dz_dx1.template block<9,3>(0,15).setZero();
      res.dz_dx2.template block<9,3>(0,15).setZero();
      //}

      // std::cout << "res.dZ_dX1: " << std::endl << res.dZ_dX1.format(cleanFmt) << std::endl;

      //{
      //  Scalar dEps = 1e-9;
      //  Eigen::Matrix<Scalar,6,6> drTvs_dX1_dF;
      //  for(int ii = 0 ; ii < 6 ; ii++){
      //    Vector6t eps = Vector6t::Zero();
      //    eps[ii] = dEps;
      //    Vector6t resPlus =
      //        SE3t::log(SE3t::exp(-eps)*pose1.Tvs * pose2.Tvs.inverse());
      //    eps[ii] = -dEps;
      //    Vector6t resMinus =
      //        SE3t::log(SE3t::exp(-eps)*pose1.Tvs * pose2.Tvs.inverse());
      //    drTvs_dX1_dF.col(ii) = (resPlus-resMinus)/(2*dEps);
      //  }
      //  std::cout << "drTvs_dX1 = [" << res.dZ_dX1.template block<6,6>(15,15).format(cleanFmt) << "]" << std::endl;
      //  std::cout << "drTvs_dX1_dF = [" << drTvs_dX1_dF.format(cleanFmt) << "]" << std::endl;
      //  std::cout << "drTvs_dX1 - drTvs_dX1_dF = [" << (res.dZ_dX1.template block<6,6>(15,15)- drTvs_dX1_dF).format(cleanFmt) << "]" << std::endl;
      //}
      //{
      //  Scalar dEps = 1e-9;
      //  Eigen::Matrix<Scalar,6,6> drTvs_dX2_dF;
      //  for(int ii = 0 ; ii < 6 ; ii++){
      //    Vector6t eps = Vector6t::Zero();
      //    eps[ii] = dEps;
      //    Vector6t resPlus = SE3t::log(pose1.Tvs * (SE3t::exp(-eps)*pose2.Tvs).inverse());
      //    eps[ii] = -dEps;
      //    //Vector6t resMinus = SE3t::log(SE3t::exp(-eps)*pose1.Tvs * pose2.Tvs.inverse());
      //    Vector6t resMinus = SE3t::log(pose1.Tvs * (SE3t::exp(-eps)*pose2.Tvs).inverse());
      //    drTvs_dX2_dF.col(ii) = (resPlus-resMinus)/(2*dEps);
      //  }
      //  std::cout << "drTvs_dX2 = [" << res.dZ_dX2.template block<6,6>(15,15).format(cleanFmt) << "]" << std::endl;
      //  std::cout << "drTvs_dX2_dF = [" << drTvs_dX2_dF.format(cleanFmt) << "]" << std::endl;
      //  std::cout << "drTvs_dX2 - drTvs_dX2_dF = [" << (res.dZ_dX2.template block<6,6>(15,15)- drTvs_dX2_dF).format(cleanFmt) << "]" << std::endl;
      //}
    } else {
      res.dz_dy = res.dz_dx1.template block<ImuResidual::kResSize, 6>(0,0) +
          res.dz_dx2.template block<ImuResidual::kResSize, 6>(0,0);
      if( translation_enabled_ == false ){
        res.dz_dy.template block<ImuResidual::kResSize, 3>(0,0).setZero();
      }
    }

    // std::cout << "BUILD imu res between " << res.PoseAId << " and " << res.PoseBId << ":" << res.Residual.transpose () << std::endl;

    //if(pose1.IsActive == false || pose2.IsActive == false){
    //    std::cout << "PRIOR RESIDUAL: ";
    //}
    //std::cout << "Residual for res " << res.ResidualId << " : " << res.Residual.transpose() << std::endl;
    errors_.push_back(res.residual.squaredNorm());

    //        res.SigmanInv = (res.dZ_dB * imu_.R * res.dZ_dB.transpose()).inverse();
    //        std::cout << "Sigma inv for res " << res.ResidualId << " is " << res.SigmanInv << std::endl;

    // res.dZ_dB.setZero();

    BA_TEST(_Test_dLog_dq((imu_pose.t_wp * Twb.inverse()).unit_quaternion()));


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

    std::cout << "dz_dg = [" << std::endl << res.dz_dg.format(kCleanFmt) << "]" <<
                 std::endl;
    std::cout << "dz_dg_fd = [" << std::endl << dz_dg_fd.format(kCleanFmt) << "]" <<
                 std::endl;
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



    // now that we have the deltas with subtracted initial velocity,
    // transform and gravity, we can construct the jacobian
    r_i_.template segment<ImuResidual::kResSize>(res.residual_offset) =
        res.residual;
  }

  // get the sigma for robust norm calculation.
  if (errors_.size() > 0) {
    auto it = errors_.begin()+std::floor(errors_.size()/2);
    std::nth_element(errors_.begin(),it,errors_.end());
    // const Scalar sigma = sqrt(*it);
    // See "Parameter Estimation Techniques: A Tutorial with Application to
    // Conic Fitting" by Zhengyou Zhang. PP 26 defines this magic number:
    // const Scalar c_huber = 1.2107*dSigma;

    // now go through the measurements and assign weights
    for (ImuResidual& res : inertial_residuals_) {
      // calculate the huber norm weight for this measurement
      // const Scalar e = res.residual.norm();
      //res.W *= e > c_huber ? c_huber/e : 1.0;
      inertial_error_ += res.residual.norm() * res.weight;
    }
  }
  errors_.clear();
  PrintTimer(_j_evaluation_inertial_);
  PrintTimer(_j_evaluation_);
  //TODO : The transpose insertions here are hideously expensive as they are
  // not in order. find a way around this.

  // here we sort the measurements and insert them per pose and per landmark,
  // this will mean each insert operation is O(1)
  //  dtime = Tic();
  StartTimer(_j_insertion_);
  StartTimer(_j_insertion_poses);
  // reserve space for j_pr_
  Eigen::VectorXi j_pr_sizes(num_poses);
  Eigen::VectorXi j_pp_sizes(num_poses);
  Eigen::VectorXi j_u_sizes(num_poses);
  Eigen::VectorXi j_i_sizes(num_poses);
  Eigen::VectorXi j_l_sizes(num_lm);

  for (Pose& pose : poses_) {
    if(pose.is_active){
      j_pr_sizes[pose.opt_id] = pose.proj_residuals.size();
      j_pp_sizes[pose.opt_id] = pose.binary_residuals.size();
      j_u_sizes[pose.opt_id] = pose.unary_residuals.size();
      j_i_sizes[pose.opt_id] = pose.inertial_residuals.size();
    }
  }

  for (Landmark& lm : landmarks_) {
    if (lm.is_active) {
      j_l_sizes[lm.opt_id] = lm.proj_residuals.size();
    }
  }

  if (!proj_residuals_.empty()) {
    j_pr_.reserve(j_pr_sizes);
    jt_pr.reserve(Eigen::VectorXi::Constant(jt_pr.cols(),
                                            kLmDim == 1 ? 2 : 1));
  }

  if (!binary_residuals_.empty()) {
    j_pp_.reserve(j_pp_sizes);
    jt_pp_.reserve(Eigen::VectorXi::Constant(jt_pp_.cols(), 2));
  }

  if (!unary_residuals_.empty()) {
    j_u_.reserve(j_u_sizes);
    jt_u_.reserve(Eigen::VectorXi::Constant(jt_u_.cols(), 1));
  }

  if (!inertial_residuals_.empty()) {
    j_i_.reserve(j_i_sizes);
    jt_i_.reserve(Eigen::VectorXi::Constant(jt_i_.cols(), 2));
  }

  if (num_lm > 0) {
    j_l_.reserve(j_l_sizes);
  }

  for (Pose& pose : poses_) {
    if (pose.is_active) {
      // sort the measurements by id so the sparse insert is O(1)
      std::sort(pose.proj_residuals.begin(), pose.proj_residuals.end());
      for (const int id: pose.proj_residuals) {
        const ProjectionResidual& res = proj_residuals_[id];
        // insert the jacobians into the sparse matrices
        // The weight is only multiplied by the transpose matrix, this is
        // so we can perform Jt*W*J*dx = Jt*W*r
        auto dz_dx = res.x_meas_id == pose.id ? res.dz_dx_meas : res.dz_dx_ref;
        j_pr_.insert(
          res.residual_id, pose.opt_id).setZero().template block<2,6>(0,0) =
            dz_dx;

        jt_pr.insert(
          pose.opt_id, res.residual_id).setZero().template block<6,2>(0,0) =
              dz_dx.transpose() * res.weight;
      }


      // add the pose/pose constraints
      std::sort(pose.binary_residuals.begin(), pose.binary_residuals.end());
      for (const int id: pose.binary_residuals) {
        const BinaryResidual& res = binary_residuals_[id];
        const Eigen::Matrix<Scalar,6,6>& dz_dz =
            res.x1_id == pose.id ? res.dz_dx1 : res.dz_dx2;

        j_pp_.insert(
          res.residual_id, pose.opt_id ).setZero().template block<6,6>(0,0) =
            dz_dz;

        jt_pp_.insert(
          pose.opt_id, res.residual_id ).setZero().template block<6,6>(0,0) =
            dz_dz.transpose() * res.weight;
      }

      // add the unary constraints
      std::sort(pose.unary_residuals.begin(), pose.unary_residuals.end());
      for (const int id: pose.unary_residuals) {
        const UnaryResidual& res = unary_residuals_[id];
        j_u_.insert(
          res.residual_id, pose.opt_id ).setZero().template block<6,6>(0,0) =
            res.dz_dx;

        jt_u_.insert(
          pose.opt_id, res.residual_id ).setZero().template block<6,6>(0,0) =
            res.dz_dx.transpose() * res.weight;
      }

      std::sort(pose.inertial_residuals.begin(), pose.inertial_residuals.end());
      for (const int id: pose.inertial_residuals) {
        const ImuResidual& res = inertial_residuals_[id];
        Eigen::Matrix<Scalar,ImuResidual::kResSize,kPoseDim> dz_dz =
            res.pose1_id == pose.id ? res.dz_dx1 : res.dz_dx2;

        j_i_.insert(
          res.residual_id, pose.opt_id ).setZero().
            template block<ImuResidual::kResSize,kPoseDim>(0,0) = dz_dz;

        // this down weights the velocity error
        // dz_dz.template block<3,kPoseDim>(6,0) *= 0.1;
        // up weight the Tvs translation prior
        if(kPoseDim > 15){
          dz_dz.template block<3,kPoseDim>(15,0) *= 100;
          dz_dz.template block<3,kPoseDim>(18,0) *= 10;
        }
        jt_i_.insert(
          pose.opt_id, res.residual_id ).setZero().
            template block<kPoseDim,ImuResidual::kResSize>(0,0) =
              dz_dz.transpose() * res.weight;
      }
    }
  }
  PrintTimer(_j_insertion_poses);

  // fill in calibration jacobians
  StartTimer(_j_insertion_calib);
  if (kCalibDim > 0) {
    for (const ImuResidual& res : inertial_residuals_) {
      // include gravity terms (t total)
      if (kCalibDim > 0 ){
        Eigen::Matrix<Scalar,9,2> dz_dg = res.dz_dg;
        j_ki_.insert(res.residual_id, 0 ).setZero().
            template block(0,0,9,2) = dz_dg.template block(0,0,9,2);

        // this down weights the velocity error
        dz_dg.template block<3,2>(6,0) *= 0.1;
        jt_ki_.insert( 0, res.residual_id ).setZero().
            template block(0,0,2,9) =
                dz_dg.transpose().template block(0,0,2,9) * res.weight;
      }

      // include Y terms
      if( kCalibDim > 2 ){
        j_ki_.coeffRef(res.residual_id,0).setZero().
            template block(0,2,ImuResidual::kResSize, 6) =
                res.dz_dy.template block(0,0,ImuResidual::kResSize, 6);

        jt_ki_.coeffRef( 0,res.residual_id).setZero().
            template block(2,0, 6, ImuResidual::kResSize) =
            res.dz_dy.template block(0,0,ImuResidual::kResSize, 6).transpose() *
            res.weight;
      }
    }

    for (const ProjectionResidual& res : proj_residuals_) {
      // include imu to camera terms (6 total)
      if (kCalibDim > 8) {
        const Eigen::Matrix<Scalar,2,5>& dz_dk = res.dz_dcam_params;
        j_kpr_.coeffRef(res.residual_id,0).setZero().
            template block(0,8,2,5) = dz_dk.template block(0,0,2,5);

        jt_kpr_.coeffRef(0,res.residual_id).setZero().
            template block(8,0,5,2) =
                dz_dk.template block(0,0,2,5).transpose() * res.weight;
      }
    }
  }
  PrintTimer(_j_insertion_calib);

  StartTimer(_j_insertion_landmarks);
  for (Landmark& lm : landmarks_) {
    if (lm.is_active) {
      // sort the measurements by id so the sparse insert is O(1)
      std::sort(lm.proj_residuals.begin(), lm.proj_residuals.end());
      for (const int id: lm.proj_residuals) {
        const ProjectionResidual& res = proj_residuals_[id];

        j_l_.insert( res.residual_id, lm.opt_id ) = res.dz_dlm;
      }
    }
  }

  PrintTimer  (_j_insertion_landmarks);
  PrintTimer(_j_insertion_);
}

// specializations
// template class BundleAdjuster<REAL_TYPE, ba::NOT_USED,9,8>;
template class BundleAdjuster<REAL_TYPE, 1,6,0>;
//template class BundleAdjuster<REAL_TYPE, 1,15,8>;
template class BundleAdjuster<REAL_TYPE, 1,15,2>;
//template class BundleAdjuster<REAL_TYPE, 1,21,2>;
// template class BundleAdjuster<double, 3,9>;


}


