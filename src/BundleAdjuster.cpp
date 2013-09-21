#include <ba/BundleAdjuster.h>


namespace ba {
#define DAMPING 1.0

////////////////////////////////////////////////////////////////////////////////
template< typename Scalar,int LmSize, int PoseSize, int CalibSize >
void BundleAdjuster<Scalar,LmSize,PoseSize,CalibSize>::ApplyUpdate(
    const VectorXt& delta_p, const VectorXt& delta_l,
    const VectorXt& delta_calib, const bool do_rollback)
{
  double coef = do_rollback == true ? -1.0 : 1.0;
  // update gravity terms if necessary
  if (inertial_residuals_.size() > 0) {
    const VectorXt delta_calib = delta_p.template tail(CalibSize);
    if (CalibSize > 0) {
      // std::cout << "Gravity delta is " <<
      // deltaCalib.template block<2,1>(0,0).transpose() <<
      // " gravity is: " << m_Imu.g.transpose() << std::endl;
    }

    if (CalibSize > 2) {
      const auto& update = delta_calib.template block<6,1>(2,0)*coef;
      imu_.t_vs = SE3t::exp(update)*imu_.t_vs;
      rig_.cameras[0].T_wc = imu_.t_vs;
      std::cout << "Tvs delta is " << (update).transpose() << std::endl;
      std::cout << "Tvs is :" << std::endl << imu_.t_vs.matrix() << std::endl;
    }
  }

  // update the camera parameters
  if( CalibSize > 8 && delta_calib.rows() > 8){
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
      const unsigned int p_offset = poses_[ii].opt_id*PoseSize;
      const auto& p_update =
          -delta_p.template block<6,1>(p_offset,0)*coef;
      if (CalibSize > 2 && inertial_residuals_.size() > 0) {
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
      if (PoseSize >= 9) {
        poses_[ii].v_w -=
            delta_p.template block<3,1>(p_offset+6,0)*coef;
            // std::cout << "Velocity for pose " << ii << " is " <<
            // m_vPoses[ii].V.transpose() << std::endl;
      }

      if (PoseSize >= 15) {
        poses_[ii].b -=
            delta_p.template block<6,1>(p_offset+9,0)*coef;
            // std::cout << "Velocity for pose " << ii << " is " <<
            // m_vPoses[ii].V.transpose() << std::endl;
      }

      if (PoseSize >= 21) {
        const auto& tvs_update =
            delta_p.template block<6,1>(p_offset+15,0)*coef;
        poses_[ii].t_vs = SE3t::exp(tvs_update)*poses_[ii].t_vs;
        poses_[ii].t_wp = poses_[ii].t_wp * SE3t::exp(-tvs_update);
        std::cout << "Tvs of pose " << ii << " after update " <<
                     (tvs_update).transpose() << " is " << std::endl <<
                     poses_[ii].t_vs.matrix() << std::endl;
      }

      //std::cout << "Pose delta for " << ii << " is " <<
      //           p_update.transpose() << std::endl;
    } else {
      // std::cout << " Pose " << ii << " is inactive." << std::endl;
      if (CalibSize > 2 && inertial_residuals_.size() > 0) {
        const auto& delta_twp = -delta_calib.template block<6,1>(2,0)*coef;
        if (do_rollback == false) {
          poses_[ii].t_wp = poses_[ii].t_wp * SE3t::exp(delta_twp);
        } else {
          poses_[ii].t_wp = poses_[ii].t_wp * SE3t::exp(delta_twp);
        }
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
        delta_l.template segment<LmSize>(landmarks_[ii].opt_id*LmSize)*coef;
      if (LmSize == 1) {
        landmarks_[ii].x_s.template tail<LmSize>() -= lm_delta;
        // std::cout << "Delta for landmark " << ii << " is " <<
        // lm_delta.transpose() << std::endl;
        // m_vLandmarks[ii].Xs /= m_vLandmarks[ii].Xs[3];
      } else {
        landmarks_[ii].x_s.template head<LmSize>() -= lm_delta;
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
              landmarks_[ii].ref_cam_id, rig_, PoseSize >= 21).inverse(),
              landmarks_[ii].x_s);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
template< typename Scalar,int LmSize, int PoseSize, int CalibSize >
void BundleAdjuster<Scalar,LmSize,PoseSize,CalibSize>::EvaluateResiduals()
{
  proj_error_ = 0;
  for (ProjectionResidual& res : proj_residuals_) {
    Landmark& lm = landmarks_[res.landmark_id];
    Pose& pose = poses_[res.x_meas_id];
    Pose& ref_pose = poses_[res.x_ref_id];
    const SE3t t_sw_m =
        pose.GetTsw(res.cam_id, rig_, PoseSize >= 21);
    const SE3t t_ws_r =
        ref_pose.GetTsw(lm.ref_cam_id,rig_, PoseSize >= 21).inverse();

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

    if (CalibSize > 2 || PoseSize > 15) {
      // disable imu translation error
      res.residual.template head<3>().setZero();
      res.residual.template segment<3>(6).setZero(); // velocity error
      res.residual.template segment<6>(9).setZero(); // bias
    }

    if (PoseSize > 15) {
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
    if (CalibSize > 2) {
      const Scalar log_dif =
          SE3t::log(imu_.t_vs * last_tvs_.inverse()).norm();

      std::cout << "logDif is " << log_dif << std::endl;
      if (log_dif < 0.01 && poses_.size() >= 30) {
        std::cout << "EMABLING TRANSLATION ERRORS" << std::endl;
        translation_enabled_ = true;
      }
      last_tvs_ = imu_.t_vs;
    }

    if (PoseSize > 15) {
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
template< typename Scalar,int LmSize, int PoseSize, int CalibSize >
void BundleAdjuster<Scalar,LmSize,PoseSize,CalibSize>::Solve(
    const unsigned int uMaxIter)
{
  for (unsigned int kk = 0 ; kk < uMaxIter ; ++kk) {
    StartTimer(_BuildProblem_);
    BuildProblem();
    PrintTimer(_BuildProblem_);    ;

    const unsigned int num_poses = num_active_poses_;
    const unsigned int num_pose_params = num_poses*PoseSize;
    const unsigned int num_lm = num_active_landmarks_;   

    StartTimer(_steup_problem_);
    StartTimer(_rhs_mult_);
    // calculate bp and bl
    VectorXt bp(num_pose_params);
    VectorXt bk;
    VectorXt bl;
    BlockMat< Eigen::Matrix<Scalar, LmSize, LmSize>> vi(num_lm, num_lm);

    VectorXt rhs_p(num_pose_params + CalibSize);
    BlockMat< Eigen::Matrix<Scalar, LmSize, PoseSize>>
        jt_l_j_pr(num_lm, num_poses);

    BlockMat< Eigen::Matrix<Scalar, PoseSize, LmSize>>
        jt_pr_j_l_vi(num_poses, num_lm);

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> s(
          num_pose_params+CalibSize, num_pose_params+CalibSize);

    PrintTimer(_rhs_mult_);


    StartTimer(_jtj_);
    // TODO: suboptimal, the matrices are symmetric. We should only
    // multipl one half
    BlockMat<Eigen::Matrix<Scalar, PoseSize, PoseSize>> u(
          num_poses, num_poses);

    u.setZero();
    bp.setZero();
    s.setZero();
    rhs_p.setZero();

    if (proj_residuals_.size() > 0) {
      Eigen::SparseBlockProduct(jt_pr, j_pr_, u);

      VectorXt jt_pr_r_pr(num_pose_params);
      Eigen::SparseBlockVectorProductDenseResult(jt_pr, r_pr_, jt_pr_r_pr);
      bp += jt_pr_r_pr;
    }

    // add the contribution from the binary terms if any
    if (binary_residuals_.size() > 0) {
      BlockMat< Eigen::Matrix<Scalar, PoseSize, PoseSize> > jt_pp_j_pp(
            num_poses, num_poses);

      Eigen::SparseBlockProduct(jt_pp_ ,j_pp_, jt_pp_j_pp);
      auto temp_u = u;
      Eigen::SparseBlockAdd(temp_u,jt_pp_j_pp,u);

      VectorXt jt_pp_r_pp(num_pose_params);
      Eigen::SparseBlockVectorProductDenseResult(jt_pp_, r_pp_, jt_pp_r_pp);
      bp += jt_pp_r_pp;
    }

    // add the contribution from the unary terms if any
    if (unary_residuals_.size() > 0) {
      BlockMat< Eigen::Matrix<Scalar, PoseSize, PoseSize> > jt_u_j_u(
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
      BlockMat< Eigen::Matrix<Scalar, PoseSize, PoseSize> > jt_i_j_i(
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
    if (LmSize > 0 && num_lm > 0) {
      bl.resize(num_lm*LmSize);
      StartTimer(_schur_complement_jtl_rpr_);
      Eigen::SparseBlockVectorProductDenseResult(jt_l_, r_pr_, bl);
      PrintTimer(_schur_complement_jtl_rpr_);

      StartTimer(_schur_complement_v);
      BlockMat< Eigen::Matrix<Scalar, LmSize, LmSize> > v(num_lm, num_lm);
      Eigen::SparseBlockProduct(jt_l_,j_l_,v);
      PrintTimer(_schur_complement_v);

      StartTimer(_schur_complement_jtpr_jl);
      BlockMat< Eigen::Matrix<Scalar, PoseSize, LmSize> > jt_pr_j_l(
            num_poses, num_lm);

      Eigen::SparseBlockProduct(jt_pr,j_l_,jt_pr_j_l);
      Eigen::SparseBlockProduct(jt_l_,j_pr_,jt_l_j_pr);
      // Jlt_Jpr = Jprt_Jl.transpose();
      PrintTimer(_schur_complement_jtpr_jl);


      StartTimer(_schur_complement_vi);
      // schur_time = Tic();
      // calculate the inverse of the map hessian (it should be diagonal,
      // unless a measurement is of more than one landmark,
      // which doesn't make sense)
      for (size_t ii = 0 ; ii < num_lm ; ++ii) {
        if (LmSize == 1) {
          if (fabs(v.coeffRef(ii,ii)(0,0)) < 1e-6) {
            std::cout << "Landmark coeff too small: " <<
                         v.coeffRef(ii,ii)(0,0) << std::endl;
            v.coeffRef(ii,ii)(0,0) += 1e-6;
          }
        }
        vi.coeffRef(ii,ii) = v.coeffRef(ii,ii).inverse();
      }
      PrintTimer(_schur_complement_vi);

      StartTimer(_schur_complement_jtpr_jl_vi);
      // attempt to solve for the poses. W_V_inv is used later on,
      // so we cache it
      Eigen::SparseBlockProduct(jt_pr_j_l, vi, jt_pr_j_l_vi);
      PrintTimer(_schur_complement_jtpr_jl_vi);


      StartTimer(_schur_complement_jtpr_jl_vi_jtl_jpr);
      BlockMat< Eigen::Matrix<Scalar, PoseSize, PoseSize>>
            jt_pr_j_l_vi_jt_l_j_pr(num_poses, num_poses);

      Eigen::SparseBlockProduct(jt_pr_j_l_vi, jt_l_j_pr,
                                jt_pr_j_l_vi_jt_l_j_pr);
      PrintTimer(_schur_complement_jtpr_jl_vi_jtl_jpr);

      //StartTimer(_schur_complement_jtpr_jl_vi_jtl_jpr_d);
      //MatrixXt djt_pr_j_l_vi(
      //      jt_pr_j_l_vi.rows()*PoseSize,jt_pr_j_l_vi.cols()*LmSize);
      //Eigen::LoadDenseFromSparse(jt_pr_j_l_vi,djt_pr_j_l_vi);

      //MatrixXt djt_l_j_pr(
      //      jt_l_j_pr.rows()*LmSize,jt_l_j_pr.cols()*PoseSize);
      //Eigen::LoadDenseFromSparse(jt_l_j_pr,djt_l_j_pr);

      //MatrixXt djt_pr_j_l_vi_jt_l_j_pr = djt_pr_j_l_vi * djt_l_j_pr;
      //PrintTimer(_schur_complement_jtpr_jl_vi_jtl_jpr_d);

      // this in-place operation should be fine for subtraction
      // schur_time = Tic();
      MatrixXt du(u.rows()*PoseSize,u.cols()*PoseSize);
      Eigen::LoadDenseFromSparse(u,du);

      /*std::cout << "Jp sparsity structure: " << std::endl <<
                     m_Jpr.GetSparsityStructure().format(cleanFmt) << std::endl;
      std::cout << "Jprt sparsity structure: " << std::endl <<
                    m_Jprt.GetSparsityStructure().format(cleanFmt) << std::endl;
      std::cout << "U sparsity structure: " << std::endl <<
                    U.GetSparsityStructure().format(cleanFmt) << std::endl;
      std::cout << "dU " << std::endl <<
                    dU << std::endl;
      */

       Eigen::SparseBlockSubtractDenseResult(u, jt_pr_j_l_vi_jt_l_j_pr,
                                             s.template block(
                                               0, 0, num_pose_params,
                                             num_pose_params ));

      // std::cout << "dU matrix is " << dU.format(cleanFmt) << std::endl;
      //s.template block(0, 0, num_pose_params, num_pose_params ) =
      //    du - djt_pr_j_l_vi_jt_l_j_pr;

      // std::cout << "Eigen::SparseBlockSubtractDenseResult(U, WV_invWt, "
      // "S.template block(0, 0, uNumPoseParams, uNumPoseParams )) took  " <<
      // Toc(dSchurTime) << " seconds."  << std::endl;

      // now form the rhs for the pose equations
      // schur_time = Tic();
      VectorXt jt_pr_j_l_vi_bll(num_pose_params);
      Eigen::SparseBlockVectorProductDenseResult(
            jt_pr_j_l_vi, bl, jt_pr_j_l_vi_bll);
      // std::cout << "Eigen::SparseBlockVectorProductDenseResult(Wp_V_inv, bl,
      // "WV_inv_bl) took  " << Toc(dSchurTime) << " seconds."  << std::endl;

      rhs_p.template head(num_pose_params) = bp - jt_pr_j_l_vi_bll;

      // std::cout << "Dense S matrix is " << S.format(cleanFmt) << std::endl;
      // std::cout << "Dense rhs matrix is " <<
      // rhs_p.transpose().format(cleanFmt) << std::endl;

    } else {
      Eigen::LoadDenseFromSparse(
            u, s.template block(0, 0, num_pose_params, num_pose_params));
      rhs_p.template head(num_pose_params) = bp;
    }
    PrintTimer(_schur_complement_);

    // std::cout << "  Rhs calculation and schur complement took " <<
    // Toc(dMatTime) << " seconds." << std::endl;

    // fill in the calibration components if any
    if (CalibSize && inertial_residuals_.size() > 0) {
      BlockMat<Eigen::Matrix<Scalar,CalibSize,CalibSize>> jt_ki_j_ki(1, 1);
      Eigen::SparseBlockProduct(jt_ki_, j_ki_, jt_ki_j_ki);
      Eigen::LoadDenseFromSparse(
            jt_ki_j_ki, s.template block<CalibSize, CalibSize>(
              num_pose_params, num_pose_params));

      BlockMat<Eigen::Matrix<Scalar, PoseSize, CalibSize>>
            jt_i_j_ki(num_poses, 1);

      Eigen::SparseBlockProduct(jt_i_, j_ki_, jt_i_j_ki);
      Eigen::LoadDenseFromSparse(
            jt_i_j_ki,
            s.template block(0, num_pose_params, num_pose_params, CalibSize));

      s.template block(num_pose_params, 0, CalibSize, num_pose_params) =
          s.template block(0, num_pose_params,
                           num_pose_params, CalibSize).transpose();

      // and the rhs for the calibration params
      bk.resize(CalibSize,1);
      Eigen::SparseBlockVectorProductDenseResult(jt_ki_, r_i_, bk);
      rhs_p.template tail<CalibSize>() = bk;
    }

    if( CalibSize > 8){
      BlockMat< Eigen::Matrix<Scalar, CalibSize, CalibSize>> jt_kpr_j_kpr(1, 1);
      Eigen::SparseBlockProduct(jt_kpr_, j_kpr_, jt_kpr_j_kpr);
      MatrixXt djt_kpr_j_kpr(CalibSize, CalibSize);
      Eigen::LoadDenseFromSparse(jt_kpr_j_kpr, djt_kpr_j_kpr);
      s.template block<CalibSize, CalibSize>(num_pose_params, num_pose_params)
          += djt_kpr_j_kpr;
      std::cout << "djt_kpr_j_kpr: " << djt_kpr_j_kpr << std::endl;

      BlockMat<Eigen::Matrix<Scalar, PoseSize, CalibSize>>
          jt_or_j_kpr(num_poses, 1);

      Eigen::SparseBlockProduct(jt_pr, j_kpr_, jt_or_j_kpr);
      MatrixXt djt_or_j_kpr(PoseSize*num_poses, CalibSize);
      Eigen::LoadDenseFromSparse(jt_or_j_kpr, djt_or_j_kpr);
      std::cout << "djt_or_j_kpr: " << djt_or_j_kpr << std::endl;
      s.template block(0, num_pose_params, num_pose_params, CalibSize) +=
          djt_or_j_kpr;
      s.template block(num_pose_params, 0, CalibSize, num_pose_params) +=
          djt_or_j_kpr.transpose();

      bk.resize(CalibSize,1);
      Eigen::SparseBlockVectorProductDenseResult(jt_kpr_, r_pr_, bk);
      rhs_p.template tail<CalibSize>() += bk;

      // schur complement
      BlockMat< Eigen::Matrix<Scalar, CalibSize, LmSize>> jt_kpr_jl(1, num_lm);
      BlockMat< Eigen::Matrix<Scalar, LmSize, CalibSize>> jt_l_j_kpr(num_lm, 1);
      Eigen::SparseBlockProduct(jt_kpr_,j_l_,jt_kpr_jl);
      Eigen::SparseBlockProduct(jt_l_,j_kpr_,jt_l_j_kpr);
      //Jlt_Jkpr = Jkprt_Jl.transpose();

      MatrixXt djt_pr_j_l_vi_jt_l_j_kpr(PoseSize*num_poses, CalibSize);
      BlockMat<Eigen::Matrix<Scalar, PoseSize, CalibSize>>
          jt_pr_j_l_vi_jt_l_j_kpr(num_poses, 1);

      Eigen::SparseBlockProduct(
            jt_pr_j_l_vi,jt_l_j_kpr, jt_pr_j_l_vi_jt_l_j_kpr);
      Eigen::LoadDenseFromSparse(
            jt_pr_j_l_vi_jt_l_j_kpr, djt_pr_j_l_vi_jt_l_j_kpr);

      std::cout << "jt_pr_j_l_vi_jt_l_j_kpr: " <<
                   djt_pr_j_l_vi_jt_l_j_kpr << std::endl;

      s.template block(0, num_pose_params, num_pose_params, CalibSize) -=
          djt_pr_j_l_vi_jt_l_j_kpr;
      s.template block(num_pose_params, 0, CalibSize, num_pose_params) -=
          djt_pr_j_l_vi_jt_l_j_kpr.transpose();

      BlockMat<Eigen::Matrix<Scalar, CalibSize, LmSize>>
          jt_kpr_j_l_vi(1, num_lm);
      Eigen::SparseBlockProduct(jt_kpr_jl,vi,jt_kpr_j_l_vi);

      BlockMat<Eigen::Matrix<Scalar, CalibSize, CalibSize>>
          jt_kpr_j_l_vi_jt_l_j_kpr(1, 1);
      Eigen::SparseBlockProduct(
            jt_kpr_j_l_vi,
            jt_l_j_kpr,
            jt_kpr_j_l_vi_jt_l_j_kpr);

      MatrixXt djt_kpr_j_l_vi_jt_l_j_kpr(CalibSize, CalibSize);
      Eigen::LoadDenseFromSparse(
            jt_kpr_j_l_vi_jt_l_j_kpr,
            djt_kpr_j_l_vi_jt_l_j_kpr);

      std::cout << "djt_kpr_j_l_vi_jt_l_j_kpr: " <<
                   djt_kpr_j_l_vi_jt_l_j_kpr << std::endl;

      s.template block<CalibSize, CalibSize>(num_pose_params, num_pose_params)
          -= djt_kpr_j_l_vi_jt_l_j_kpr;

      VectorXt jt_kpr_j_l_vi_bl;
      jt_kpr_j_l_vi_bl.resize(CalibSize);
      Eigen::SparseBlockVectorProductDenseResult(
            jt_kpr_j_l_vi, bl, jt_kpr_j_l_vi_bl);
      // std::cout << "Eigen::SparseBlockVectorProductDenseResult(Wp_V_inv, bl,"
      // " WV_inv_bl) took  " << Toc(dSchurTime) << " seconds."  << std::endl;

      std::cout << "jt_kpr_j_l_vi_bl: " <<
                   jt_kpr_j_l_vi_bl.transpose() << std::endl;
      std::cout << "rhs_p.template tail<CalibSize>(): " <<
                   rhs_p.template tail<CalibSize>().transpose() << std::endl;
      rhs_p.template tail<CalibSize>() -= jt_kpr_j_l_vi_bl;
    }
    PrintTimer(_steup_problem_);

    // now we have to solve for the pose constraints
    StartTimer(_solve_);
    VectorXt delta_p = num_poses == 0 ? VectorXt() : s.ldlt().solve(rhs_p);
    PrintTimer(_solve_);

    StartTimer(_back_substitution_);
    VectorXt delta_l;
    if (num_lm > 0) {
      delta_l.resize(num_lm*LmSize);
      VectorXt jt_l_j_pr_delta_p;
      jt_l_j_pr_delta_p.resize(num_lm*LmSize );
      Eigen::SparseBlockVectorProductDenseResult(
            jt_l_j_pr,
            delta_p.head(num_pose_params),
            jt_l_j_pr_delta_p);

      VectorXt rhs_l;
      rhs_l.resize(num_lm*LmSize );
      rhs_l =  bl - jt_l_j_pr_delta_p;

      for (size_t ii = 0 ; ii < num_lm ; ++ii) {
        delta_l.template block<LmSize,1>( ii*LmSize, 0 ).noalias() =
            vi.coeff(ii,ii)*rhs_l.template block<LmSize,1>(ii*LmSize,0);
      }
    }
    PrintTimer(_back_substitution_);

    VectorXt deltaCalib;
    if (CalibSize > 0 && num_pose_params > 0) {
      deltaCalib = delta_p.template tail(CalibSize);
      std::cout << "Delta calib: " << deltaCalib.transpose() << std::endl;
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
      //std::cout << "Error increasing during optimization, rolling back .."<<
      //            std::endl;
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


  if (PoseSize >= 15 && poses_.size() > 0) {
    imu_.b_g = poses_.back().b.template head<3>();
    imu_.b_a = poses_.back().b.template tail<3>();
  }

  if (PoseSize >= 21 && poses_.size() > 0) {
    imu_.t_vs = poses_.back().t_vs;
  }
  // std::cout << "Solve took " << Toc(dTime) << " seconds." << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
template< typename Scalar,int LmSize, int PoseSize, int CalibSize >
void BundleAdjuster<Scalar, LmSize, PoseSize, CalibSize>::BuildProblem()
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
  jt_l_.resize(num_lm, num_proj_res);
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
  jt_l_.setZero();


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
          ref_pose.GetTsw(lm.ref_cam_id, rig_, PoseSize >= 21) ,lm.x_w);

    auto& cam = rig_.cameras[res.cam_id].camera;

    const SE3t t_vs_m =
        (PoseSize >= 21 ? pose.t_vs : rig_.cameras[res.cam_id].T_wc);
    const SE3t t_vs_r =
        (PoseSize >= 21 ? ref_pose.t_vs :  rig_.cameras[res.cam_id].T_wc);
    const SE3t t_sw_m =
        pose.GetTsw(res.cam_id, rig_, PoseSize >= 21);
    const SE3t t_ws_r =
        ref_pose.GetTsw(lm.ref_cam_id,rig_, PoseSize >= 21).inverse();

    const Vector2t p = cam.Transfer3D(
          t_sw_m*t_ws_r, lm.x_s.template head<3>(),lm.x_s(3));

    res.residual = res.z - p;

    // this array is used to calculate the robust norm
    errors_.push_back(res.residual.squaredNorm());

    const Eigen::Matrix<Scalar,2,4> dTdP_s = cam.dTransfer3D_dP(
          t_sw_m*t_ws_r, lm.x_s.template head<3>(),lm.x_s(3));

    // Landmark Jacobian
    if (lm.is_active) {
      res.dz_dlm = -dTdP_s.template block<2,LmSize>( 0, LmSize == 3 ? 0 : 3 );
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
      if (LmSize == 1) {
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
        //    SE3t Tss = (pose.Twp*Tvs_m).inverse() *
        //    (refPose.Twp*SE3t::exp(delta)) * Tvs_r;
        //
        //    const Vector2t pPlus =
        //    m_Rig.cameras[res.CameraId].camera.Transfer3D(
        //    Tss,lm.Xs.template head(3),lm.Xs[3]);

        //    delta[ii] = -dEps;
        //    Tsw = (pose.Twp*SE3t::exp(delta)*
        //    m_Rig.cameras[meas.CameraId].T_wc).inverse();
        //
        //    Tss = (pose.Twp*Tvs_m).inverse() *
        //    (refPose.Twp*SE3t::exp(delta)) * Tvs_r;
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
    // res.dZ_dX = dLog_dX(Twp, res.Twp.inverse());
    res.dz_dx = dLog_decoupled_dX(Twp, res.t_wp);

    //Eigen::Matrix<Scalar,6,6> J_fd;
    //Scalar dEps = 1e-10;
    //for (int ii = 0; ii < 6 ; ii++) {
    //  Eigen::Matrix<Scalar,6,1> delta;
    //  delta.setZero();
    //  delta[ii] = dEps;
    //  const Vector6t pPlus =
    //    log_decoupled(exp_decoupled(Twp,delta) , res.Twp);
    //  delta[ii] = -dEps;
    //  const Vector6t pMinus =
    //    log_decoupled(exp_decoupled(Twp,delta) , res.Twp);
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
    res.dZ_dG.setZero();
    res.dz_db.setZero();

    // calculate the derivative of the lie log with
    // respect to the tangent plane at Twa
    const Eigen::Matrix<Scalar,6,7> se3_log =
        dLog_dSE3(imu_pose.t_wp*t_w2.inverse());

    Eigen::Matrix<Scalar,7,6> dse3_dx1;
    dse3_dx1.setZero();
    dse3_dx1.template block<3,3>(0,0) = t_w1.so3().matrix();
    // for this derivation  refer to page 16 of notes
    dse3_dx1.template block<3,3>(0,3) =
        dqx_dq<Scalar>(
          (t_w1).unit_quaternion(),
          t_12_0.translation()-t_12_0.so3()*
          t_w2.so3().inverse()*t_w2.translation()) *
        dq1q2_dq2(t_w1.unit_quaternion()) *
        dqExp_dw<Scalar>(Eigen::Matrix<Scalar,3,1>::Zero());

    dse3_dx1.template block<4,3>(3,3) =
        dq1q2_dq1((t_12_0.so3() * t_w2.so3().inverse()).unit_quaternion()) *
        dq1q2_dq2(t_w1.unit_quaternion()) *
        dqExp_dw<Scalar>(Eigen::Matrix<Scalar,3,1>::Zero());

    // TODO: the block<3,3>(0,0) jacobian is incorrect here due to
    // multiplication by Twb.inverse(). Fix this
    jb_q.template block<4,3>(3,0) =
        dq1q2_dq1(t_w2.inverse().unit_quaternion())*
        jb_q.template block<4,3>(3,0) ;

    // dt/dB
    res.dz_db.template block<6,6>(0,0) =
        se3_log * jb_q.template block<7,6>(0,0);

    // dV/dB
    res.dz_db.template block<3,6>(6,0) = jb_q.template block<3,6>(7,0);
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
    res.dz_dx1.template block<6,6>(0,0) =  se3_log*dse3_dx1;
    res.dz_dx1.template block<ImuResidual::kResSize,6>(0,9) = res.dz_db;

    // the - sign is here because of the exp(-x) within the log
    res.dz_dx2.template block<6,6>(0,0) =
        -dLog_dX(imu_pose.t_wp,t_w2.inverse());

    res.dz_dx2.template block<3,3>(6,6) = -Matrix3t::Identity();
    res.dz_dx2.template block<6,6>(9,9) =
        -Eigen::Matrix<Scalar,6,6>::Identity();

    const Eigen::Matrix<Scalar,3,2> dGravity = dGravity_dDirection(imu_.g);
    res.dZ_dG.template block<3,2>(0,0) =
        -0.5*powi(total_dt,2)*Matrix3t::Identity()*dGravity;

    res.dZ_dG.template block<3,2>(6,0) =
        -total_dt*Matrix3t::Identity()*dGravity;

    res.residual.template head<6>() = SE3t::log(t_w1*t_12*t_w2.inverse());
    res.residual.template segment<3>(6) = imu_pose.v_w - pose2.v_w;
    res.residual.template segment<6>(9) = pose1.b - pose2.b;

    if ((CalibSize > 2 || PoseSize > 15) && translation_enabled_ == false) {
      // disable imu translation error
      res.residual.template head<3>().setZero();
      res.residual.template segment<3>(6).setZero(); // velocity error
      res.residual.template segment<6>(9).setZero(); // bias
      res.dz_dx1.template block<3, PoseSize>(0,0).setZero();
      res.dz_dx2.template block<3, PoseSize>(0,0).setZero();
      res.dz_dx1.template block<3, PoseSize>(6,0).setZero();
      res.dz_dx2.template block<3, PoseSize>(6,0).setZero();
      res.dz_db.template block<3, 6>(0,0).setZero();
      res.dZ_dG.template block<3, 2>(0,0).setZero();

      // disable accelerometer and gyro bias
      res.dz_dx1.template block<res.kResSize, 6>(0,9).setZero();
      res.dz_dx2.template block<res.kResSize, 6>(0,9).setZero();
    }

    if (PoseSize > 15) {
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
      //        SE3t::log(SE3t::exp(-eps)*poseA.Tvs * poseB.Tvs.inverse());
      //    eps[ii] = -dEps;
      //    Vector6t resMinus =
      //        SE3t::log(SE3t::exp(-eps)*poseA.Tvs * poseB.Tvs.inverse());
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
      //    Vector6t resPlus = SE3t::log(poseA.Tvs * (SE3t::exp(-eps)*poseB.Tvs).inverse());
      //    eps[ii] = -dEps;
      //    //Vector6t resMinus = SE3t::log(SE3t::exp(-eps)*poseA.Tvs * poseB.Tvs.inverse());
      //    Vector6t resMinus = SE3t::log(poseA.Tvs * (SE3t::exp(-eps)*poseB.Tvs).inverse());
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

    //if(poseA.IsActive == false || poseB.IsActive == false){
    //    std::cout << "PRIOR RESIDUAL: ";
    //}
    //std::cout << "Residual for res " << res.ResidualId << " : " << res.Residual.transpose() << std::endl;
    errors_.push_back(res.residual.squaredNorm());

    //        res.SigmanInv = (res.dZ_dB * m_Imu.R * res.dZ_dB.transpose()).inverse();
    //        std::cout << "Sigma inv for res " << res.ResidualId << " is " << res.SigmanInv << std::endl;

    // res.dZ_dB.setZero();

    /*
      {
          Scalar dEps = 1e-9;
          Eigen::Quaternion<Scalar> q = (imuPose.Twp * Twb.inverse()).unit_quaternion();
          std::cout << "q:" << q.coeffs().transpose() << std::endl;
          Eigen::Matrix<Scalar,3,4> dLog_dq_fd;
          for(int ii = 0 ; ii < 4 ; ii++){
              Vector4t eps = Vector4t::Zero();
              eps[ii] += dEps;
              Eigen::Quaternion<Scalar> qPlus = q;
              qPlus.coeffs() += eps;
              Sophus::SO3Group<Scalar> so3_plus;
              memcpy(so3_plus.data(),qPlus.coeffs().data(),sizeof(Scalar)*4);
              Vector3t resPlus = so3_plus.log();

              eps[ii] -= 2*dEps;
              Eigen::Quaternion<Scalar> qMinus = q;
              qMinus.coeffs() += eps;
              Sophus::SO3Group<Scalar> so3_Minus;
              memcpy(so3_Minus.data(),qMinus.coeffs().data(),sizeof(Scalar)*4);
              Vector3t resMinus = so3_Minus.log();

              dLog_dq_fd.col(ii) = (resPlus-resMinus)/(2*dEps);
          }
          std::cout << "dlog_dq = [" << dLog_dq(q).format(cleanFmt) << "]" << std::endl;
          std::cout << "dlog_dqf = [" << dLog_dq_fd.format(cleanFmt) << "]" << std::endl;
          std::cout << "dlog_dq - dlog_dqf = [" << (dLog_dq(q)- dLog_dq_fd).format(cleanFmt) << "]" << std::endl;
      }



      Scalar dEps = 1e-9;
      Eigen::Matrix<Scalar,6,6> Jlog_fd;
      for(int ii = 0 ; ii < 6 ; ii++){
          Vector6t eps = Vector6t::Zero();
          eps[ii] += dEps;
          Vector6t resPlus = SE3t::log(Twa*SE3t::exp(eps) * (Twb*Tab.inverse()).inverse());
          eps[ii] -= 2*dEps;
          Vector6t resMinus = SE3t::log(Twa*SE3t::exp(eps) * (Twb*Tab.inverse()).inverse());
          Jlog_fd.col(ii) = (resPlus-resMinus)/(2*dEps);
      }
      const Eigen::Matrix<Scalar,6,6> Jlog = dLog_dX(Twa,(Twb*Tab.inverse()).inverse());
      std::cout << "Jlog = [" << Jlog.format(cleanFmt) << "]" << std::endl;
      std::cout << "Jlogf = [" << Jlog_fd.format(cleanFmt) << "]" << std::endl;
      std::cout << "Jlog - Jlogf = [" << (Jlog- Jlog_fd).format(cleanFmt) << "]" << std::endl;

      Eigen::Matrix<Scalar,6,6> dlog_dTwbf;
      for(int ii = 0 ; ii < 6 ; ii++){
          Vector6t eps = Vector6t::Zero();
          eps[ii] += dEps;
          const Vector6t resPlus = SE3t::log(Twa*Tab * (Twb*SE3t::exp(eps)).inverse());
          eps[ii] -= 2*dEps;
          const Vector6t resMinus = SE3t::log(Twa*Tab * (Twb*SE3t::exp(eps)).inverse());
          dlog_dTwbf.col(ii) = (resPlus-resMinus)/(2*dEps);
      }

      std::cout << "dlog_dTwb = [" << (-dLog_dX(Twa*Tab, Twb.inverse())).format(cleanFmt) << "]" << std::endl;
      std::cout << "dlog_dTwbf = [" << dlog_dTwbf.format(cleanFmt) << "]" << std::endl;
      std::cout << "dlog_dTwb - dlog_dTwbf = [" << (-dLog_dX(Twa*Tab, Twb.inverse()) - dlog_dTwbf).format(cleanFmt) << "]" << std::endl;



      // verify using finite differences
      Eigen::Matrix<Scalar,9,9> J_fd;
      J_fd.setZero();
      Eigen::Matrix<Scalar,9,2> Jg_fd;
      Jg_fd.setZero();
      Eigen::Matrix<Scalar,9,6> Jb_fd;
      Jb_fd.setZero();

      Eigen::Matrix<Scalar,9,9>  dRi_dx2_fd;
      for(int ii = 0; ii < 9 ; ii++){
          Eigen::Matrix<Scalar,9,1> epsVec = Eigen::Matrix<Scalar,9,1>::Zero();
          epsVec[ii] += dEps;
          ImuPose y0_eps(poseB.Twp,poseB.V, Vector3t::Zero(),0);
          y0_eps.Twp = y0_eps.Twp * SE3t::exp(epsVec.template head<6>());
          y0_eps.V += epsVec.template tail<3>();
          Eigen::Matrix<Scalar,9,1> r_plus;
          r_plus.template head<6>() = SE3t::log(imuPose.Twp * y0_eps.Twp.inverse());
          r_plus.template tail<3>() = imuPose.V - y0_eps.V;



          epsVec[ii] -= 2*dEps;
          y0_eps = ImuPose(poseB.Twp,poseB.V, Vector3t::Zero(),0);;
          y0_eps.Twp = y0_eps.Twp * SE3t::exp(epsVec.template head<6>());
          y0_eps.V += epsVec.template tail<3>();
          Eigen::Matrix<Scalar,9,1> r_minus;
          r_minus.template head<6>() = SE3t::log(imuPose.Twp * y0_eps.Twp.inverse());
          r_minus.template tail<3>() = imuPose.V - y0_eps.V;

          dRi_dx2_fd.col(ii) = (r_plus-r_minus)/(2*dEps);
      }
      std::cout << "res.dZ_dX2= " << std::endl << res.dZ_dX2.template block<9,9>(0,0).format(cleanFmt) << std::endl;
      std::cout << "dRi_dx2_fd = " << std::endl <<  dRi_dx2_fd.format(cleanFmt) << std::endl;
      std::cout << "res.dZ_dX2-dRi_dx2_fd = " << std::endl << (res.dZ_dX2.template block<9,9>(0,0)-dRi_dx2_fd).format(cleanFmt) <<
                   "norm: " << (res.dZ_dX2.template block<9,9>(0,0)-dRi_dx2_fd).norm() <<  std::endl;

      Eigen::Matrix<Scalar,7,6> dSE3_dX1_fd;
      for(int ii = 0 ; ii < 6 ; ii++){
          Vector6t eps = Vector6t::Zero();
          eps[ii] = dEps;
          Pose poseEps = poseA;
          poseEps.Twp = poseEps.Twp*SE3t::exp(eps);
          // poseEps.Twp = poseEps.Twp * SE3t::exp(eps);
          std::vector<ImuPose> poses;
          const ImuPose imuPosePlus = ImuResidual::IntegrateResidual(poseEps,res.Measurements,m_Imu.Bg,m_Imu.Ba,gravity,poses);
          Vector7t dErrorPlus;
          dErrorPlus.template head<3>() = (imuPosePlus.Twp * Twb.inverse()).translation();
          dErrorPlus.template tail<4>() = (imuPosePlus.Twp * Twb.inverse()).unit_quaternion().coeffs();
          eps[ii] = -dEps;
          poseEps = poseA;
          poseEps.Twp = poseEps.Twp*SE3t::exp(eps);
          // poseEps.Twp = poseEps.Twp * SE3t::exp(eps);
          poses.clear();
          const ImuPose imuPoseMinus = ImuResidual::IntegrateResidual(poseEps,res.Measurements,m_Imu.Bg,m_Imu.Ba,gravity,poses);
          Vector7t dErrorMinus;
          dErrorMinus.template head<3>() = (imuPoseMinus.Twp * Twb.inverse()).translation();
          dErrorMinus.template tail<4>() = (imuPoseMinus.Twp * Twb.inverse()).unit_quaternion().coeffs();

          dSE3_dX1_fd.col(ii).template head<7>() = (dErrorPlus - dErrorMinus)/(2*dEps);
      }

      std::cout << "dSE3_dX1 = [" << std::endl << dSE3_dX1.format(cleanFmt) << "]" << std::endl;
      std::cout << "dSE3_dX1_Fd = [" << std::endl << dSE3_dX1_fd.format(cleanFmt) << "]" << std::endl;
      std::cout << "dSE3_dX1-dSE3_dX1_fd = [" << std::endl << (dSE3_dX1-dSE3_dX1_fd).format(cleanFmt) << "] norm = " << (dSE3_dX1-dSE3_dX1_fd).norm() << std::endl;


      for(int ii = 0 ; ii < 6 ; ii++){
          Vector6t eps = Vector6t::Zero();
          eps[ii] = dEps;
          Pose poseEps = poseA;
          poseEps.Twp = poseEps.Twp*SE3t::exp(eps);
          // poseEps.Twp = poseEps.Twp * SE3t::exp(eps);
          std::vector<ImuPose> poses;
          const ImuPose imuPosePlus = ImuResidual::IntegrateResidual(poseEps,res.Measurements,m_Imu.Bg,m_Imu.Ba,gravity,poses);
          // const Vector6t dErrorPlus = log_decoupled(imuPosePlus.Twp, Twb);
          const Vector6t dErrorPlus = SE3t::log(imuPosePlus.Twp * Twb.inverse());
          const Vector3t vErrorPlus = imuPosePlus.V - poseB.V;
          eps[ii] = -dEps;
          poseEps = poseA;
          poseEps.Twp = poseEps.Twp*SE3t::exp(eps);
          // poseEps.Twp = poseEps.Twp * SE3t::exp(eps);
          poses.clear();
          const ImuPose imuPoseMinus = ImuResidual::IntegrateResidual(poseEps,res.Measurements,m_Imu.Bg,m_Imu.Ba,gravity,poses);
          // const Vector6t dErrorMinus = log_decoupled(imuPoseMinus.Twp, Twb);
          const Vector6t dErrorMinus = SE3t::log(imuPoseMinus.Twp * Twb.inverse());
          const Vector3t vErrorMinus = imuPoseMinus.V - poseB.V;
          J_fd.col(ii).template head<6>() = (dErrorPlus - dErrorMinus)/(2*dEps);
          J_fd.col(ii).template tail<3>() = (vErrorPlus - vErrorMinus)/(2*dEps);
      }

      for(int ii = 0 ; ii < 3 ; ii++){
          Vector3t eps = Vector3t::Zero();
          eps[ii] = dEps;
          Pose poseEps = poseA;
          poseEps.V += eps;
          std::vector<ImuPose> poses;
          const ImuPose imuPosePlus = ImuResidual::IntegrateResidual(poseEps,res.Measurements,m_Imu.Bg,m_Imu.Ba,gravity,poses);
          const Vector6t dErrorPlus = SE3t::log(imuPosePlus.Twp * Twb.inverse());
//                std::cout << "Pose plus: " << imuPosePlus.Twp.matrix() << std::endl;
          const Vector3t vErrorPlus = imuPosePlus.V - poseB.V;
          eps[ii] = -dEps;
          poseEps = poseA;
          poseEps.V += eps;
          poses.clear();
          const ImuPose imuPoseMinus = ImuResidual::IntegrateResidual(poseEps,res.Measurements,m_Imu.Bg,m_Imu.Ba,gravity,poses);
          const Vector6t dErrorMinus = SE3t::log(imuPoseMinus.Twp * Twb.inverse());
//                std::cout << "Pose minus: " << imuPoseMinus.Twp.matrix() << std::endl;
          const Vector3t vErrorMinus = imuPoseMinus.V - poseB.V;
          J_fd.col(ii+6).template head<6>() = (dErrorPlus - dErrorMinus)/(2*dEps);
          J_fd.col(ii+6).template tail<3>() = (vErrorPlus - vErrorMinus)/(2*dEps);
      }

      for(int ii = 0 ; ii < 2 ; ii++){
          Vector2t eps = Vector2t::Zero();
          eps[ii] += dEps;
          std::vector<ImuPose> poses;
          const Vector2t gPlus = m_Imu.G+eps;
          const ImuPose imuPosePlus = ImuResidual::IntegrateResidual(poseA,res.Measurements,m_Imu.Bg,m_Imu.Ba,GetGravityVector(gPlus),poses);
          const Vector6t dErrorPlus = SE3t::log(imuPosePlus.Twp * Twb.inverse());
//                std::cout << "Pose plus: " << imuPosePlus.Twp.matrix() << std::endl;
          const Vector3t vErrorPlus = imuPosePlus.V - poseB.V;
          eps[ii] -= 2*dEps;
          poses.clear();
          const Vector2t gMinus = m_Imu.G+eps;
          const ImuPose imuPoseMinus = ImuResidual::IntegrateResidual(poseA,res.Measurements,m_Imu.Bg,m_Imu.Ba,GetGravityVector(gMinus),poses);
          const Vector6t dErrorMinus = SE3t::log(imuPoseMinus.Twp * Twb.inverse());
//                std::cout << "Pose minus: " << imuPoseMinus.Twp.matrix() << std::endl;
          const Vector3t vErrorMinus = imuPoseMinus.V - poseB.V;
          Jg_fd.col(ii).template head<6>() = (dErrorPlus - dErrorMinus)/(2*dEps);
          Jg_fd.col(ii).template tail<3>() = (vErrorPlus - vErrorMinus)/(2*dEps);
      }

      Vector6t biasVec;
      biasVec.template head<3>() = m_Imu.Bg;
      biasVec.template tail<3>() = m_Imu.Ba;
      for(int ii = 0 ; ii < 6 ; ii++){
          Vector6t eps = Vector6t::Zero();
          eps[ii] += dEps;
          std::vector<ImuPose> poses;
          const Vector6t plusBiases = biasVec + eps;
          const ImuPose imuPosePlus = ImuResidual::IntegrateResidual(poseA,res.Measurements,plusBiases.template head<3>(),plusBiases.template tail<3>(),gravity,poses);
          const Vector6t dErrorPlus = SE3t::log(imuPosePlus.Twp * Twb.inverse());
          const Vector3t vErrorPlus = imuPosePlus.V - poseB.V;

          eps[ii] -= 2*dEps;
          const Vector6t minusBiases = biasVec + eps;
          poses.clear();
          const ImuPose imuPoseMinus = ImuResidual::IntegrateResidual(poseA,res.Measurements,minusBiases.template head<3>(),minusBiases.template tail<3>(),gravity,poses);
          const Vector6t dErrorMinus = SE3t::log(imuPoseMinus.Twp * Twb.inverse());
          const Vector3t vErrorMinus = imuPoseMinus.V - poseB.V;
          Jb_fd.col(ii).template head<6>() = (dErrorPlus - dErrorMinus)/(2*dEps);
          Jb_fd.col(ii).template tail<3>() = (vErrorPlus - vErrorMinus)/(2*dEps);
      }


      std::cout << "J = [" << std::endl << res.dZ_dX1.template block<9,9>(0,0).format(cleanFmt) << "]" << std::endl;
      std::cout << "Jf = [" << std::endl << J_fd.format(cleanFmt) << "]" << std::endl;
      std::cout << "J-Jf = [" << std::endl << (res.dZ_dX1.template block<9,9>(0,0)-J_fd).format(cleanFmt) << "] norm = " << (res.dZ_dX1.template block<9,9>(0,0)-J_fd).norm() << std::endl;

      std::cout << "Jg = [" << std::endl << res.dZ_dG.format(cleanFmt) << "]" << std::endl;
      std::cout << "Jgf = [" << std::endl << Jg_fd.format(cleanFmt) << "]" << std::endl;
      std::cout << "Jg-Jgf = [" << std::endl << (res.dZ_dG-Jg_fd).format(cleanFmt) << "] norm = " << (res.dZ_dG-Jg_fd).norm() << std::endl;

      std::cout << "Jb = [" << std::endl << res.dZ_dB.template block<9,6>(0,0).format(cleanFmt) << "]" << std::endl;
      std::cout << "Jbf = [" << std::endl << Jb_fd.format(cleanFmt) << "]" << std::endl;
      std::cout << "Jb-Jbf = [" << std::endl << (res.dZ_dB.template block<9,6>(0,0)-Jb_fd).format(cleanFmt) << "] norm = " << (res.dZ_dB.template block<9,6>(0,0)-Jb_fd).norm() << std::endl;
      */


    // now that we have the deltas with subtracted initial velocity, transform and gravity, we can construct the jacobian
    r_i_.template segment<ImuResidual::kResSize>(res.residual_offset) =
        res.residual;
  }

  // get the sigma for robust norm calculation.
  if (errors_.size() > 0) {
    auto it = errors_.begin()+std::floor(errors_.size()/2);
    std::nth_element(errors_.begin(),it,errors_.end());
    // const Scalar sigma = sqrt(*it);
    // See "Parameter Estimation Techniques: A Tutorial with Application to Conic
    // Fitting" by Zhengyou Zhang. PP 26 defines this magic number:
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
        Eigen::Matrix<Scalar,ImuResidual::kResSize,PoseSize> dz_dz =
            res.pose1_id == pose.id ? res.dz_dx1 : res.dz_dx2;

        j_i_.insert(
          res.residual_id, pose.opt_id ).setZero().
            template block<ImuResidual::kResSize,PoseSize>(0,0) = dz_dz;

        // this down weights the velocity error
        dz_dz.template block<3,PoseSize>(6,0) *= 0.1;
        // up weight the Tvs translation prior
        if(PoseSize > 15){
          dz_dz.template block<3,PoseSize>(15,0) *= 100;
          dz_dz.template block<3,PoseSize>(18,0) *= 10;
        }
        jt_i_.insert(
          pose.opt_id, res.residual_id ).setZero().
            template block<PoseSize,ImuResidual::kResSize>(0,0) =
              dz_dz.transpose() * res.weight;
      }
    }
  }
  PrintTimer(_j_insertion_poses);

  // fill in calibration jacobians
  StartTimer(_j_insertion_calib);
  if (CalibSize > 0) {
    for (const ImuResidual& res : inertial_residuals_) {
      // include gravity terms (t total)
      if (CalibSize > 0 ){
        Eigen::Matrix<Scalar,9,2> dz_dg = res.dZ_dG;
        j_ki_.insert(res.residual_id, 0 ).setZero().
            template block(0,0,9,2) = dz_dg.template block(0,0,9,2);

        // this down weights the velocity error
        dz_dg.template block<3,2>(6,0) *= 0.1;
        jt_ki_.insert( 0, res.residual_id ).setZero().
            template block(0,0,2,9) =
                dz_dg.transpose().template block(0,0,2,9) * res.weight;
      }

      // include Y terms
      if( CalibSize > 2 ){
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
      if (CalibSize > 8) {
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
        jt_l_.insert( lm.opt_id, res.residual_id ) =
            res.dz_dlm.transpose() * res.weight;
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


