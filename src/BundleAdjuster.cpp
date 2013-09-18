#include <ba/BundleAdjuster.h>


namespace ba {
#define DAMPING 1.0

////////////////////////////////////////////////////////////////////////////////
template< typename Scalar,int LmSize, int PoseSize, int CalibSize >
void BundleAdjuster<Scalar,LmSize,PoseSize,CalibSize>::_ApplyUpdate(
    const VectorXt& delta_p, const VectorXt& delta_l,
    const VectorXt& delta_calib, const bool do_rollback)
{
  double coef = do_rollback == true ? -1.0 : 1.0;
  // update gravity terms if necessary
  if (m_vImuResiduals.size() > 0) {
    const VectorXt delta_calib = delta_p.template tail(CalibSize);
    if (CalibSize > 0) {
      // std::cout << "Gravity delta is " <<
      // deltaCalib.template block<2,1>(0,0).transpose() <<
      // " gravity is: " << m_Imu.g.transpose() << std::endl;
    }

    if (CalibSize > 2) {
      const auto& update = delta_calib.template block<6,1>(2,0)*coef;
      m_Imu.t_vs = SE3t::exp(update)*m_Imu.t_vs;
      m_Rig.cameras[0].T_wc = m_Imu.t_vs;
      std::cout << "Tvs delta is " << (update).transpose() << std::endl;
      std::cout << "Tvs is :" << std::endl << m_Imu.t_vs.matrix() << std::endl;
    }
  }

  // update the camera parameters
  if( CalibSize > 8 && delta_calib.rows() > 8){
    const auto& update = delta_calib.template block<5,1>(8,0)*coef;

    std::cout << "calib delta: " << (update).transpose() << std::endl;

    const VectorXt params = m_Rig.cameras[0].camera.GenericParams();
    m_Rig.cameras[0].camera.SetGenericParams(params-(update*coef));

    std::cout << "new params: " <<
                 m_Rig.cameras[0].camera.GenericParams().transpose() <<
                 std::endl;
  }

  // update poses
  // std::cout << "Updating " << uNumPoses << " active poses." << std::endl;
  for (size_t ii = 0 ; ii < m_vPoses.size() ; ++ii) {
    // only update active poses, as inactive ones are not part of the
    // optimization
    if (m_vPoses[ii].is_active) {
      const unsigned int p_offset = m_vPoses[ii].opt_id*PoseSize;
      const auto& p_update =
          -delta_p.template block<6,1>(p_offset,0)*coef;
      if (CalibSize > 2 && m_vImuResiduals.size() > 0) {
        const auto& calib_update =
            -delta_calib.template block<6,1>(2,0)*coef;

        if (do_rollback == false) {
          m_vPoses[ii].t_wp = m_vPoses[ii].t_wp * SE3t::exp(p_update);
          m_vPoses[ii].t_wp = m_vPoses[ii].t_wp * SE3t::exp(calib_update);
        } else {
          m_vPoses[ii].t_wp = m_vPoses[ii].t_wp * SE3t::exp(calib_update);
          m_vPoses[ii].t_wp = m_vPoses[ii].t_wp * SE3t::exp(p_update);
        }
        std::cout << "Pose " << ii << " calib delta is " <<
                     (calib_update).transpose() << std::endl;
        m_vPoses[ii].t_vs = m_Imu.t_vs;
      } else {
        m_vPoses[ii].t_wp = m_vPoses[ii].t_wp * SE3t::exp(p_update);
      }

      // update the velocities if they are parametrized
      if (PoseSize >= 9) {
        m_vPoses[ii].v_w -=
            delta_p.template block<3,1>(p_offset+6,0)*coef;
            // std::cout << "Velocity for pose " << ii << " is " <<
            // m_vPoses[ii].V.transpose() << std::endl;
      }

      if (PoseSize >= 15) {
        m_vPoses[ii].b -=
            delta_p.template block<6,1>(p_offset+9,0)*coef;
            // std::cout << "Velocity for pose " << ii << " is " <<
            // m_vPoses[ii].V.transpose() << std::endl;
      }

      if (PoseSize >= 21) {
        const auto& tvs_update =
            delta_p.template block<6,1>(p_offset+15,0)*coef;
        m_vPoses[ii].t_vs = SE3t::exp(tvs_update)*m_vPoses[ii].t_vs;
        m_vPoses[ii].t_wp = m_vPoses[ii].t_wp * SE3t::exp(-tvs_update);
        std::cout << "Tvs of pose " << ii << " after update " <<
                     (tvs_update).transpose() << " is " << std::endl <<
                     m_vPoses[ii].t_vs.matrix() << std::endl;
      }

      // std::cout << "Pose delta for " << ii << " is " <<
      //             p_update.transpose() << std::endl;
    } else {
      // std::cout << " Pose " << ii << " is inactive." << std::endl;
      if (CalibSize > 2 && m_vImuResiduals.size() > 0) {
        const auto& delta_twp = -delta_calib.template block<6,1>(2,0)*coef;
        if (do_rollback == false) {
          m_vPoses[ii].t_wp = m_vPoses[ii].t_wp * SE3t::exp(delta_twp);
        } else {
          m_vPoses[ii].t_wp = m_vPoses[ii].t_wp * SE3t::exp(delta_twp);
        }
        std::cout << "INACTIVE POSE " << ii << " calib delta is " <<
                     (delta_twp).transpose() << std::endl;
        m_vPoses[ii].t_vs = m_Imu.t_vs;
      }
    }

    // clear the vector of Tsw values as they will need to be recalculated
    m_vPoses[ii].t_sw.clear();
  }

  // update the landmarks
  for (size_t ii = 0 ; ii < m_vLandmarks.size() ; ++ii) {
    if (m_vLandmarks[ii].is_active) {
      const auto& lm_delta =
        delta_l.template segment<LmSize>(m_vLandmarks[ii].opt_id*LmSize)*coef;
      if (LmSize == 1) {
        m_vLandmarks[ii].x_s.template tail<LmSize>() -= lm_delta;
        // std::cout << "Delta for landmark " << ii << " is " <<
        // lm_delta.transpose() << std::endl;
        // m_vLandmarks[ii].Xs /= m_vLandmarks[ii].Xs[3];
      } else {
        m_vLandmarks[ii].x_s.template head<LmSize>() -= lm_delta;
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

      m_vLandmarks[ii].x_w =
          MultHomogeneous(m_vPoses[m_vLandmarks[ii].ref_pose_id].GetTsw(
              m_vLandmarks[ii].ref_cam_id, m_Rig, PoseSize >= 21).inverse(),
              m_vLandmarks[ii].x_s);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
template< typename Scalar,int LmSize, int PoseSize, int CalibSize >
void BundleAdjuster<Scalar,LmSize,PoseSize,CalibSize>::_EvaluateResiduals()
{
  m_dProjError = 0;
  for (ProjectionResidual& res : m_vProjResiduals) {
    Landmark& lm = m_vLandmarks[res.landmark_id];
    Pose& pose = m_vPoses[res.x_meas_id];
    Pose& ref_pose = m_vPoses[res.x_ref_id];
    const SE3t t_sw_m =
        pose.GetTsw(res.cam_id, m_Rig, PoseSize >= 21);
    const SE3t t_ws_r =
        ref_pose.GetTsw(lm.ref_cam_id,m_Rig, PoseSize >= 21).inverse();

    const Vector2t p = m_Rig.cameras[res.cam_id].camera.Transfer3D(
          t_sw_m*t_ws_r, lm.x_s.template head<3>(),lm.x_s(3));

    res.residual = res.z - p;
    m_dProjError += res.residual.norm() * res.weight;
  }

  m_dImuError = 0;
  double total_tvs_change = 0;
  for (ImuResidual& res : m_vImuResiduals) {
    // set up the initial pose for the integration
    const Vector3t gravity = GetGravityVector(m_Imu.g);

    const Pose& pose1 = m_vPoses[res.pose1_id];
    const Pose& pose2 = m_vPoses[res.pose2_id];

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

      if (m_bEnableTranslation == false) {
        total_tvs_change += res.residual.template segment<6>(15).norm();

      }
      res.residual.template segment<3>(15).setZero();
    }

    // std::cout << "EVALUATE imu res between " << res.PoseAId << " and " <<
    // res.PoseBId << ":" << res.Residual.transpose () << std::endl;
    m_dImuError += res.residual.norm() * res.weight;
  }

  if (m_vImuResiduals.size() > 0 && m_bEnableTranslation == false) {
    if (CalibSize > 2) {
      const Scalar log_dif =
          SE3t::log(m_Imu.t_vs * m_dLastTvs.inverse()).norm();

      std::cout << "logDif is " << log_dif << std::endl;
      if (log_dif < 0.01 && m_vPoses.size() >= 30) {
        std::cout << "EMABLING TRANSLATION ERRORS" << std::endl;
        m_bEnableTranslation = true;
      }
      m_dLastTvs = m_Imu.t_vs;
    }

    if (PoseSize > 15) {
      std::cout << "Total tvs change is: " << total_tvs_change << std::endl;
      if (m_dTotalTvsChange != 0 &&
          total_tvs_change/m_vImuResiduals.size() < 0.1 &&
          m_vPoses.size() >= 30) {
        std::cout << "EMABLING TRANSLATION ERRORS" << std::endl;
        m_bEnableTranslation = true;
        total_tvs_change = 0;
      }
      m_dTotalTvsChange = total_tvs_change;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
template< typename Scalar,int LmSize, int PoseSize, int CalibSize >
void BundleAdjuster<Scalar,LmSize,PoseSize,CalibSize>::Solve(
    const unsigned int uMaxIter)
{
  Eigen::IOFormat clean_fmt(2, 0, ", ", ";\n" , "" , "");

  for (unsigned int kk = 0 ; kk < uMaxIter ; ++kk) {
    double it_time = Tic();

    Scalar time = Tic();
    BuildProblem();
    // std::cout << "Build problem took " << Toc(dTime) << " seconds." << std::endl;
    time = Tic();

    const unsigned int num_poses = m_uNumActivePoses;
    const unsigned int num_pose_params = num_poses*PoseSize;
    const unsigned int num_lm = m_uNumActiveLandmakrs;
    //        const unsigned int uNumMeas = m_vMeasurements.size();
    // calculate bp and bl
    Scalar mat_time = Tic();
    VectorXt bp(num_pose_params);
    VectorXt bk;
    VectorXt bl;
    Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, LmSize, LmSize>>
        Vi(num_lm, num_lm);

    VectorXt rhs_p(num_pose_params + CalibSize);
    Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, LmSize, PoseSize>>
        Jlt_Jpr(num_lm, num_poses);
    Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, PoseSize, LmSize>>
        Jprt_Jl_Vi(num_poses, num_lm);

    Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>
        S(num_pose_params + CalibSize,num_pose_params + CalibSize);

    // std::cout << "  Rhs vector mult took " << Toc(dMatTime) <<
    // " seconds." << std::endl;

    mat_time = Tic();

    // TODO: suboptimal, the matrices are symmetric. We should only multipl one half
    Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, PoseSize, PoseSize> > U(num_poses, num_poses);
    U.setZero();
    bp.setZero();
    S.setZero();
    rhs_p.setZero();

    if( m_vProjResiduals.size() > 0 ){
      Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, PoseSize, PoseSize> > Jprt_Jpr(num_poses, num_poses);
      Eigen::SparseBlockProduct(m_Jprt, m_Jpr, U);

      VectorXt Jprt_Rpr(num_pose_params);
      Eigen::SparseBlockVectorProductDenseResult(m_Jprt, m_Rpr, Jprt_Rpr);
      bp += Jprt_Rpr;
    }

    // add the contribution from the binary terms if any
    if( m_vBinaryResiduals.size() > 0 ) {
      Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, PoseSize, PoseSize> > Jppt_Jpp(num_poses, num_poses);
      Eigen::SparseBlockProduct(m_Jppt ,m_Jpp, Jppt_Jpp);
      auto Temp = U;
      Eigen::SparseBlockAdd(Temp,Jppt_Jpp,U);

      VectorXt Jppt_Rpp(num_pose_params);
      Eigen::SparseBlockVectorProductDenseResult(m_Jppt, m_Rpp, Jppt_Rpp);
      bp += Jppt_Rpp;
    }

    // add the contribution from the unary terms if any
    if( m_vUnaryResiduals.size() > 0 ) {
      Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, PoseSize, PoseSize> > Jut_Ju(num_poses, num_poses);
      Eigen::SparseBlockProduct(m_Jut, m_Ju, Jut_Ju);
      auto Temp = U;
      Eigen::SparseBlockAdd(Temp, Jut_Ju, U);

      VectorXt Jut_Ru(num_pose_params);
      Eigen::SparseBlockVectorProductDenseResult(m_Jut, m_Ru, Jut_Ru);
      bp += Jut_Ru;

      //                Eigen::LoadDenseFromSparse(U,S);
      //                std::cout << "Dense S matrix is " << S.format(cleanFmt) << std::endl;
    }

    // add the contribution from the imu terms if any
    if( m_vImuResiduals.size() > 0 ) {
      Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, PoseSize, PoseSize> > Jit_Ji(num_poses, num_poses);
      Eigen::SparseBlockProduct(m_Jit, m_Ji, Jit_Ji);
      auto Temp = U;
      Eigen::SparseBlockAdd(Temp, Jit_Ji, U);

      VectorXt Jit_Ri(num_pose_params);
      Eigen::SparseBlockVectorProductDenseResult(m_Jit, m_Ri, Jit_Ri);
      bp += Jit_Ri;
      // Eigen::LoadDenseFromSparse(U,S);
      // std::cout << "Dense S matrix is " << S.format(cleanFmt) << std::endl;
    }

    if( LmSize > 0 && num_lm > 0) {
      bl.resize(num_lm*LmSize);
      double dSchurTime = Tic();
      Eigen::SparseBlockVectorProductDenseResult(m_Jlt, m_Rpr, bl);
      // std::cout << "Eigen::SparseBlockVectorProductDenseResult(m_Jlt, m_Rpr, bl); took  " << Toc(dSchurTime) << " seconds."  << std::endl;

      dSchurTime = Tic();
      Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, LmSize, LmSize> > V(num_lm, num_lm);
      Eigen::SparseBlockProduct(m_Jlt,m_Jl,V);
      // std::cout << "Eigen::SparseBlockProduct(m_Jlt,m_Jl,V);; took  " << Toc(dSchurTime) << " seconds."  << std::endl;

      dSchurTime = Tic();
      Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, PoseSize, LmSize> > Jprt_Jl(num_poses, num_lm);
      Eigen::SparseBlockProduct(m_Jprt,m_Jl,Jprt_Jl);
      Eigen::SparseBlockProduct(m_Jlt,m_Jpr,Jlt_Jpr);
      // Jlt_Jpr = Jprt_Jl.transpose();
      // std::cout << "Eigen::SparseBlockProduct(m_Jprt,m_Jl,Wp) and Wpt = Wp.transpose(); took  " << Toc(dSchurTime) << " seconds."  << std::endl;


      dSchurTime = Tic();
      // calculate the inverse of the map hessian (it should be diagonal, unless a measurement is of more than
      // one landmark, which doesn't make sense)
      for(size_t ii = 0 ; ii < num_lm ; ii++){
        if(LmSize == 1){
          if(fabs(V.coeffRef(ii,ii)(0,0)) < 1e-6){
            std::cout << "Landmark coeff too small: " << V.coeffRef(ii,ii)(0,0) << std::endl;
            V.coeffRef(ii,ii)(0,0) += 1e-6;
          }
        }
        Vi.coeffRef(ii,ii) = V.coeffRef(ii,ii).inverse();
      }

      // std::cout << "  Inversion of V took " << Toc(dSchurTime) << " seconds." << std::endl;
      // Eigen::LoadDenseFromSparse(V_inv,S);
      // std::cout << "Vinv is " << S.format(cleanFmt) << std::endl;


      dSchurTime = Tic();
      // attempt to solve for the poses. W_V_inv is used later on, so we cache it
      Eigen::SparseBlockProduct(Jprt_Jl, Vi, Jprt_Jl_Vi);
      // std::cout << "Eigen::SparseBlockProduct(Wp, V_inv, Wp_V_inv) took  " << Toc(dSchurTime) << " seconds."  << std::endl;


      dSchurTime = Tic();
      // Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, PoseSize, PoseSize> > WV_invWt(uNumPoses, uNumPoses);
      // Eigen::SparseBlockProduct(Jprt_Jl_Vi, Jlt_Jpr, WV_invWt);
      // std::cout << "Eigen::SparseBlockProduct(Wp_V_inv, Wpt, WV_invWt) took  " << Toc(dSchurTime) << " seconds."  << std::endl;

      dSchurTime = Tic();
      Eigen::MatrixXd dJprt_Jl_Vi(Jprt_Jl_Vi.rows()*PoseSize,Jprt_Jl_Vi.cols()*LmSize);
      Eigen::LoadDenseFromSparse(Jprt_Jl_Vi,dJprt_Jl_Vi);

      Eigen::MatrixXd dJlt_Jpr(Jlt_Jpr.rows()*LmSize,Jlt_Jpr.cols()*PoseSize);
      Eigen::LoadDenseFromSparse(Jlt_Jpr,dJlt_Jpr);

      Eigen::MatrixXd dJprt_Jl_Vi_Jlt_Jpr = dJprt_Jl_Vi * dJlt_Jpr;
      // std::cout << "Same with dense took " << Toc(dSchurTime) << " seconds."  << std::endl;



      // this in-place operation should be fine for subtraction
      dSchurTime = Tic();
      Eigen::MatrixXd dU(U.rows()*PoseSize,U.cols()*PoseSize);
      Eigen::LoadDenseFromSparse(U,dU);

      /*std::cout << "Jp sparsity structure: " << std::endl << m_Jpr.GetSparsityStructure().format(cleanFmt) << std::endl;
          std::cout << "Jprt sparsity structure: " << std::endl << m_Jprt.GetSparsityStructure().format(cleanFmt) << std::endl;
          std::cout << "U sparsity structure: " << std::endl << U.GetSparsityStructure().format(cleanFmt) << std::endl;
          std::cout << "dU " << std::endl << dU << std::endl;
          */

      // Eigen::SparseBlockSubtractDenseResult(U, WV_invWt, S.template block(0, 0, uNumPoseParams, uNumPoseParams ));
      // std::cout << "dU matrix is " << dU.format(cleanFmt) << std::endl;
      S.template block(0, 0, num_pose_params, num_pose_params ) = dU - dJprt_Jl_Vi_Jlt_Jpr;
      // std::cout << "Eigen::SparseBlockSubtractDenseResult(U, WV_invWt, S.template block(0, 0, uNumPoseParams, uNumPoseParams )) took  " << Toc(dSchurTime) << " seconds."  << std::endl;

      // now form the rhs for the pose equations
      dSchurTime = Tic();
      VectorXt Jpt_Jl_Vi_bl(num_pose_params);
      Eigen::SparseBlockVectorProductDenseResult(Jprt_Jl_Vi, bl, Jpt_Jl_Vi_bl);
      // std::cout << "Eigen::SparseBlockVectorProductDenseResult(Wp_V_inv, bl, WV_inv_bl) took  " << Toc(dSchurTime) << " seconds."  << std::endl;

      rhs_p.template head(num_pose_params) = bp - Jpt_Jl_Vi_bl;

      // std::cout << "Dense S matrix is " << S.format(cleanFmt) << std::endl;
      // std::cout << "Dense rhs matrix is " << rhs_p.transpose().format(cleanFmt) << std::endl;

    }else{
      Eigen::LoadDenseFromSparse(U, S.template block(0, 0, num_pose_params, num_pose_params));
      rhs_p.template head(num_pose_params) = bp;
    }

    // std::cout << "  Rhs calculation and schur complement took " << Toc(dMatTime) << " seconds." << std::endl;

    Eigen::MatrixXd dJki;
    dJki.resize(m_vImuResiduals.size() * ImuResidual::kResSize,CalibSize);
    Eigen::LoadDenseFromSparse(m_Jki,dJki);
    // std::cout << "Dense dJki matrix is " << dJki.format(cleanFmt) << std::endl;

    // fill in the calibration components if any
    if( CalibSize && m_vImuResiduals.size() > 0 ){
      Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,CalibSize,CalibSize> > Jkit_Jki(1, 1);
      Eigen::SparseBlockProduct(m_Jkit, m_Jki, Jkit_Jki);
      Eigen::LoadDenseFromSparse(Jkit_Jki, S.template block<CalibSize, CalibSize>(num_pose_params, num_pose_params));

      Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,PoseSize,CalibSize> > Jit_Jki(num_poses, 1);
      Eigen::SparseBlockProduct(m_Jit, m_Jki, Jit_Jki);
      Eigen::LoadDenseFromSparse(Jit_Jki, S.template block(0, num_pose_params, num_pose_params, CalibSize));

      S.template block(num_pose_params, 0, CalibSize, num_pose_params) = S.template block(0, num_pose_params, num_pose_params, CalibSize).transpose();

      // and the rhs for the calibration params
      bk.resize(CalibSize,1);
      Eigen::SparseBlockVectorProductDenseResult(m_Jkit, m_Ri, bk);
      rhs_p.template tail<CalibSize>() = bk;
    }

    if( CalibSize > 8){
      Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,CalibSize,CalibSize> > Jkprt_Jkpr(1, 1);
      Eigen::SparseBlockProduct(m_Jkprt, m_Jkpr, Jkprt_Jkpr);
      Eigen::MatrixXd dJkprt_Jkpr(CalibSize, CalibSize);
      Eigen::LoadDenseFromSparse(Jkprt_Jkpr, dJkprt_Jkpr);
      S.template block<CalibSize, CalibSize>(num_pose_params, num_pose_params) += dJkprt_Jkpr;
      std::cout << "dJkprt_Jkpr: " << dJkprt_Jkpr << std::endl;

      Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,PoseSize,CalibSize> > Jprt_Jkpr(num_poses, 1);
      Eigen::SparseBlockProduct(m_Jprt, m_Jkpr, Jprt_Jkpr);
      Eigen::MatrixXd dJprt_Jkpr(PoseSize*num_poses, CalibSize);
      Eigen::LoadDenseFromSparse(Jprt_Jkpr, dJprt_Jkpr);
      std::cout << "dJprt_Jkpr: " << dJprt_Jkpr << std::endl;
      S.template block(0, num_pose_params, num_pose_params, CalibSize) += dJprt_Jkpr;
      S.template block(num_pose_params, 0, CalibSize, num_pose_params) += dJprt_Jkpr.transpose();

      bk.resize(CalibSize,1);
      Eigen::SparseBlockVectorProductDenseResult(m_Jkprt, m_Rpr, bk);
      rhs_p.template tail<CalibSize>() += bk;

      // schur complement
      Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, CalibSize, LmSize> > Jkprt_Jl(1, num_lm);
      Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, LmSize, CalibSize> > Jlt_Jkpr(num_lm, 1);
      Eigen::SparseBlockProduct(m_Jkprt,m_Jl,Jkprt_Jl);
      Eigen::SparseBlockProduct(m_Jlt,m_Jkpr,Jlt_Jkpr);
      //Jlt_Jkpr = Jkprt_Jl.transpose();

      Eigen::MatrixXd dJpt_Jl_Vi_Jlt_Jkpr(PoseSize*num_poses, CalibSize);
      Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, PoseSize, CalibSize> > Jpt_Jl_Vi_Jlt_Jkpr(num_poses, 1);
      Eigen::SparseBlockProduct(Jprt_Jl_Vi,Jlt_Jkpr,Jpt_Jl_Vi_Jlt_Jkpr);
      Eigen::LoadDenseFromSparse(Jpt_Jl_Vi_Jlt_Jkpr, dJpt_Jl_Vi_Jlt_Jkpr);
      std::cout << "dJpt_Jl_Vi_Jlt_Jkpr: " << dJpt_Jl_Vi_Jlt_Jkpr << std::endl;
      S.template block(0, num_pose_params, num_pose_params, CalibSize) -= dJpt_Jl_Vi_Jlt_Jkpr;
      S.template block(num_pose_params, 0, CalibSize, num_pose_params) -= dJpt_Jl_Vi_Jlt_Jkpr.transpose();

      Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, CalibSize, LmSize> > Jkprt_Jl_Vi(1, num_lm);
      Eigen::SparseBlockProduct(Jkprt_Jl,Vi,Jkprt_Jl_Vi);
      Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, CalibSize, CalibSize> > Jkprt_Jl_Vi_Jlt_Jkpr(1, 1);
      Eigen::SparseBlockProduct(Jkprt_Jl_Vi,Jlt_Jkpr,Jkprt_Jl_Vi_Jlt_Jkpr);
      Eigen::MatrixXd dJkprt_Jl_Vi_Jlt_Jkpr(CalibSize, CalibSize);
      Eigen::LoadDenseFromSparse(Jkprt_Jl_Vi_Jlt_Jkpr, dJkprt_Jl_Vi_Jlt_Jkpr);
      std::cout << "dJkprt_Jl_Vi_Jlt_Jkpr: " << dJkprt_Jl_Vi_Jlt_Jkpr << std::endl;
      S.template block<CalibSize, CalibSize>(num_pose_params, num_pose_params) -= dJkprt_Jl_Vi_Jlt_Jkpr;

      VectorXt Jkprt_Jl_Vi_bl;
      Jkprt_Jl_Vi_bl.resize(CalibSize);
      Eigen::SparseBlockVectorProductDenseResult(Jkprt_Jl_Vi, bl, Jkprt_Jl_Vi_bl);
      // std::cout << "Eigen::SparseBlockVectorProductDenseResult(Wp_V_inv, bl, WV_inv_bl) took  " << Toc(dSchurTime) << " seconds."  << std::endl;
      std::cout << "Jkprt_Jl_Vi_bl: " << Jkprt_Jl_Vi_bl.transpose() << std::endl;
      std::cout << "rhs_p.template tail<CalibSize>(): " << rhs_p.template tail<CalibSize>().transpose() << std::endl;
      rhs_p.template tail<CalibSize>() -= Jkprt_Jl_Vi_bl;
    }

    //          std::cout << "Dense S matrix is " << S.format(cleanFmt) << std::endl;
    //          std::cout << "Dense rhs matrix is " << rhs_p.transpose().format(cleanFmt) << std::endl;

    // std::cout << "Setup took " << Toc(dTime) << " seconds." << std::endl;

    // now we have to solve for the pose constraints
    time = Tic();
    S.template block<3,3>(S.rows()-3,S.cols()-3) += Eigen::Matrix<Scalar,3,3>::Identity()*1e-6;
    VectorXt delta_p = num_poses == 0 ? VectorXt() : S.ldlt().solve(rhs_p);
    //        VectorXt delta_p = uNumPoses == 0 ? VectorXt() : S.inverse() * rhs_p;
    // std::cout << "Cholesky solve of " << uNumPoses << " by " << uNumPoses << "matrix took " << Toc(dTime) << " seconds." << std::endl;


    VectorXt delta_l;
    if(num_lm > 0){
      delta_l.resize(num_lm*LmSize);
      VectorXt Wt_delta_p;
      Wt_delta_p.resize(num_lm*LmSize );
      Eigen::SparseBlockVectorProductDenseResult(Jlt_Jpr,delta_p.head(num_pose_params),Wt_delta_p);
      VectorXt rhs_l;
      rhs_l.resize(num_lm*LmSize );
      rhs_l =  bl - Wt_delta_p;

      for(size_t ii = 0 ; ii < num_lm ; ii++){
        delta_l.template block<LmSize,1>( ii*LmSize, 0 ).noalias() =  Vi.coeff(ii,ii)*rhs_l.template block<LmSize,1>(ii*LmSize,0);
      }
    }

    VectorXt deltaCalib;
    if( CalibSize > 0 && num_pose_params > 0 ) {
      deltaCalib = delta_p.template tail(CalibSize);
      std::cout << "Delta calib: " << deltaCalib.transpose() << std::endl;
    }

    _ApplyUpdate(delta_p, delta_l, deltaCalib, false);



    const double dPrevError = m_dProjError + m_dImuError;
    // std::cout << "Pre-solve norm: " << dPrevError << " with Epr:" << m_dProjError << " and Ei:" << m_dImuError << std::endl;
    _EvaluateResiduals();
    const double dPostError = m_dProjError + m_dImuError;
    // std::cout << "Pose-solve norm: " << dPostError << " with Epr:" << m_dProjError << " and Ei:" << m_dImuError << std::endl;
    if( dPostError > dPrevError ){
      // std::cout << "Error increasing during optimization, rolling back .." << std::endl;
      _ApplyUpdate(delta_p, delta_l, deltaCalib, true );
      break;
    }
    else if( (dPrevError - dPostError)/dPrevError < 0.01 ){
      // std::cout << "Error decrease less than 1%, aborting." << std::endl;
      break;
    }
    // std::cout << "BA iteration " << kk <<  " error: " << m_Rpr.norm() + m_Ru.norm() + m_Rpp.norm() + m_Ri.norm() << std::endl;
    // std::cout << "Iteration " << kk << " took " << Toc(dItTime) << " seconds. " << std::endl;
  }


  if(PoseSize >= 15 && m_vPoses.size() > 0){
    m_Imu.b_g = m_vPoses.back().b.template head<3>();
    m_Imu.b_a = m_vPoses.back().b.template tail<3>();
  }

  if(PoseSize >= 21 && m_vPoses.size() > 0){
    m_Imu.t_vs = m_vPoses.back().t_vs;
  }
  // std::cout << "Solve took " << Toc(dTime) << " seconds." << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////
template< typename Scalar,int LmSize, int PoseSize, int CalibSize >
void BundleAdjuster<Scalar, LmSize, PoseSize, CalibSize>::BuildProblem()
{
  Eigen::IOFormat cleanFmt(4, 0, ", ", ";\n" , "" , "");

  // resize as needed
  const unsigned int uNumPoses = m_uNumActivePoses;
  const unsigned int uNumLm = m_uNumActiveLandmakrs;
  const unsigned int uNumProjRes = m_vProjResiduals.size();
  const unsigned int uNumBinRes = m_vBinaryResiduals.size();
  const unsigned int uNumUnRes = m_vUnaryResiduals.size();
  const unsigned int uNumImuRes = m_vImuResiduals.size();

  m_Jpr.resize(uNumProjRes, uNumPoses);
  m_Jprt.resize(uNumPoses, uNumProjRes);
  m_Jkpr.resize(uNumProjRes, 1);
  m_Jkprt.resize(1, uNumProjRes);
  m_Jl.resize(uNumProjRes, uNumLm);
  m_Jlt.resize(uNumLm, uNumProjRes);
  m_Rpr.resize(uNumProjRes*ProjectionResidual::kResSize);

  m_Jpp.resize(uNumBinRes, uNumPoses);
  m_Jppt.resize(uNumPoses, uNumBinRes);
  m_Rpp.resize(uNumBinRes*BinaryResidual::kResSize);

  m_Ju.resize(uNumUnRes, uNumPoses);
  m_Jut.resize(uNumPoses, uNumUnRes);
  m_Ru.resize(uNumUnRes*UnaryResidual::kResSize);

  m_Ji.resize(uNumImuRes, uNumPoses);
  m_Jit.resize(uNumPoses, uNumImuRes);
  m_Jki.resize(uNumImuRes, 1);
  m_Jkit.resize(1, uNumImuRes);
  m_Ri.resize(uNumImuRes*ImuResidual::kResSize);


  // these calls remove all the blocks, but KEEP allocated memory as long as the object is alive
  m_Jpr.setZero();
  m_Jprt.setZero();
  m_Jkpr.setZero();
  m_Jkprt.setZero();
  m_Rpr.setZero();

  m_Jpp.setZero();
  m_Jppt.setZero();
  m_Rpp.setZero();

  m_Ju.setZero();
  m_Jut.setZero();
  m_Ru.setZero();

  m_Ji.setZero();
  m_Jit.setZero();
  m_Jki.setZero();
  m_Jkit.setZero();
  m_Ri.setZero();

  m_Jl.setZero();
  m_Jlt.setZero();


  // used to store errors for robust norm calculation
  m_vErrors.reserve(uNumProjRes);
  m_vErrors.clear();

  // set all jacobians
  Scalar dTime = Tic();

  m_dProjError = 0;
  for( ProjectionResidual& res : m_vProjResiduals ){
    // calculate measurement jacobians

    // Tsw = T_cv * T_vw
    Landmark& lm = m_vLandmarks[res.landmark_id];
    Pose& pose = m_vPoses[res.x_meas_id];
    Pose& refPose = m_vPoses[res.x_ref_id];
    lm.x_s = MultHomogeneous(refPose.GetTsw(lm.ref_cam_id, m_Rig, PoseSize >= 21) ,lm.x_w);
    const SE3t Tvs_m = (PoseSize >= 21 ? pose.t_vs :  m_Rig.cameras[res.cam_id].T_wc);
    const SE3t Tvs_r = (PoseSize >= 21 ? refPose.t_vs :  m_Rig.cameras[res.cam_id].T_wc);
    const SE3t Tsw_m = pose.GetTsw(res.cam_id, m_Rig, PoseSize >= 21);
    const SE3t Tws_r = refPose.GetTsw(lm.ref_cam_id,m_Rig, PoseSize >= 21).inverse();

    const Vector2t p = m_Rig.cameras[res.cam_id].camera.Transfer3D(Tsw_m*Tws_r, lm.x_s.template head<3>(),lm.x_s(3));
    res.residual = res.z - p;
    // std::cout << "Residual for meas " << res.ResidualId << " and landmark " << res.LandmarkId << " with camera " << res.CameraId << " is " << res.Residual.transpose() << std::endl;

    // this array is used to calculate the robust norm
    m_vErrors.push_back(res.residual.squaredNorm());

    const Eigen::Matrix<Scalar,2,4> dTdP_s = m_Rig.cameras[res.cam_id].camera.dTransfer3D_dP(Tsw_m*Tws_r,
                                                                                             lm.x_s.template head<3>(),lm.x_s(3));
    // Landmark Jacobian
    if(lm.is_active){
      res.dz_dlm = -dTdP_s.template block<2,LmSize>( 0, LmSize == 3 ? 0 : 3 );
    }

    if( pose.is_active || refPose.is_active ) {
      // std::cout << "Calculating j for residual with poseid " << pose.Id << " and refPoseId " << refPose.Id << std::endl;
      // derivative for the measurement pose
      const Vector4t Xs_m = MultHomogeneous(Tsw_m, lm.x_w);
      const Eigen::Matrix<Scalar,2,4> dTdP_m = m_Rig.cameras[res.cam_id].camera.dTransfer3D_dP(SE3t(),
                                                                                               Xs_m.template head<3>(),Xs_m(3));
      const Eigen::Matrix<Scalar,2,4> dTdP_m_Tsv_m = dTdP_m * Tvs_m.inverse().matrix();
      for(unsigned int ii=0; ii<6; ++ii){
        res.dz_dx_meas.template block<2,1>(0,ii) = -dTdP_m_Tsv_m * -Sophus::SE3Group<Scalar>::generator(ii) * pose.t_wp.inverse().matrix() * lm.x_w; // rotation
      }
      // res.dZ_dK = m_Rig.cameras[res.CameraId].camera

      // only need this if we are in inverse depth mode
      if( LmSize == 1 ){
        // derivative for the reference pose
        const Eigen::Matrix<Scalar,2,4> dTdP_m_Tsw_m = dTdP_m * (pose.t_wp * Tvs_m).inverse().matrix();
        for(unsigned int ii=0; ii<6; ++ii){
          res.dz_dx_ref.template block<2,1>(0,ii) = -dTdP_m_Tsw_m * refPose.t_wp.matrix() * Sophus::SE3Group<Scalar>::generator(ii) * Tvs_r.matrix() * lm.x_s;
        }

        //                Eigen::Matrix<Scalar,2,6> dZ_dPr_fd;
        //                for(int ii = 0; ii < 6 ; ii++) {
        //                    Eigen::Matrix<Scalar,6,1> delta;
        //                    delta.setZero();
        //                    delta[ii] = dEps;
        //                    SE3t Tss = (pose.Twp*Tvs_m).inverse() * (refPose.Twp*SE3t::exp(delta)) * Tvs_r;
        //                    const Vector2t pPlus = m_Rig.cameras[res.CameraId].camera.Transfer3D(Tss,lm.Xs.template head(3),lm.Xs[3]);
        //                    delta[ii] = -dEps;
        //                    // Tsw = (pose.Twp*SE3t::exp(delta)*m_Rig.cameras[meas.CameraId].T_wc).inverse();
        //                    Tss = (pose.Twp*Tvs_m).inverse() * (refPose.Twp*SE3t::exp(delta)) * Tvs_r;
        //                    const Vector2t pMinus = m_Rig.cameras[res.CameraId].camera.Transfer3D(Tss,lm.Xs.template head(3),lm.Xs[3]);
        //                    dZ_dPr_fd.col(ii) = -(pPlus-pMinus)/(2*dEps);
        //                }
        //                std::cout << "dZ_dPr   :" << res.dZ_dPr << std::endl;
        //                std::cout << "dZ_dPr_fd:" << dZ_dPr_fd << " norm: " << (res.dZ_dPr - dZ_dPr_fd).norm() <<  std::endl;
      }

      // calculate jacobian wrt to camera parameters
      // [TEST]: This is only working for fov models
      //            Vector3t Xs_m_norm = Xs_m.template head<3>() / Xs_m[3];
      //            const VectorXt params = m_Rig.cameras[res.CameraId].camera.GenericParams();
      //            res.dZ_dK = -m_Rig.cameras[res.CameraId].camera.dMap_dParams(Xs_m_norm, params);

      //            {
      //                double dEps = 1e-9;
      //                Eigen::Matrix<Scalar,2,5> dZ_dK_fd;
      //                for(int ii = 0; ii < 6 ; ii++) {
      //                    Eigen::Matrix<Scalar,5,1> delta;
      //                    delta.setZero();
      //                    delta[ii] = dEps;
      //                    m_Rig.cameras[res.CameraId].camera.SetGenericParams(params + delta);
      //                    const Vector2t pPlus = -m_Rig.cameras[res.CameraId].camera.Transfer3D(SE3t(),
      //                                                                                         Xs_m.template head<3>(),Xs_m(3));
      //                    delta[ii] = -dEps;
      //                    m_Rig.cameras[res.CameraId].camera.SetGenericParams(params + delta);
      //                    const Vector2t pMinus = -m_Rig.cameras[res.CameraId].camera.Transfer3D(SE3t(),
      //                                                                                          Xs_m.template head<3>(),Xs_m(3));
      //                    dZ_dK_fd.col(ii) = (pPlus-pMinus)/(2*dEps);
      //                }
      //                std::cout << "dZ_dK   :" << std::endl << res.dZ_dK << std::endl;
      //                std::cout << "dZ_dK_fd:" << std::endl << dZ_dK_fd << " norm: " << (res.dZ_dK - dZ_dK_fd).norm() <<  std::endl;
      //                m_Rig.cameras[res.CameraId].camera.SetGenericParams(params);
      //            }
    }
    // set the residual in m_R which is dense
    m_Rpr.template segment<ProjectionResidual::kResSize>(res.residual_offset) = res.residual;
  }

  // std::cout << "Max dZ_dX norm: " << maxdZ_dX_norm << " with dZ_dX: " << std::endl << maxdZ_dX << " with dZ_dX_fd: "  << std::endl << maxdZ_dX_fd << std::endl;
  // std::cout << "Max dZ_dPm norm: " << maxdZ_dPm_norm << " with error: " << std::endl << maxdZ_dPm << " with dZ_dPm_fd: " << std::endl  << maxdZ_dPm_fd <<  std::endl;
  // std::cout << "Max dZ_dPr norm: " << maxdZ_dPr_norm << " with error: " << std::endl << maxdZ_dPr <<  " with dZ_dPr_fd: " << std::endl  << maxdZ_dPr_fd  << std::endl;

  // get the sigma for robust norm calculation. This call is O(n) on average,
  // which is desirable over O(nlogn) sort
  if( m_vErrors.size() > 0 ){
    auto it = m_vErrors.begin()+std::floor(m_vErrors.size()/2);
    std::nth_element(m_vErrors.begin(),it,m_vErrors.end());
    const Scalar dSigma = sqrt(*it);
    // std::cout << "Projection error sigma is " << dSigma << std::endl;
    // See "Parameter Estimation Techniques: A Tutorial with Application to Conic
    // Fitting" by Zhengyou Zhang. PP 26 defines this magic number:
    const Scalar c_huber = 1.2107*dSigma;

    // now go through the measurements and assign weights
    for( ProjectionResidual& res : m_vProjResiduals ){
      // calculate the huber norm weight for this measurement
      const Scalar e = res.residual.norm();
      res.weight = e > c_huber ? c_huber/e : 1.0;
      m_dProjError += res.residual.norm() * res.weight;
    }
  }
  m_vErrors.clear();

  // build binary residual jacobians
  for( BinaryResidual& res : m_vBinaryResiduals ){
    const SE3t& Twa = m_vPoses[res.x1_id].t_wp;
    const SE3t& Twb = m_vPoses[res.x2_id].t_wp;
    res.dz_dx1 = dLog_dX(Twa, res.t_ab * Twb.inverse());
    // the negative sign here is because exp(x) is inside the inverse when we invert (Twb*exp(x)).inverse
    res.dZ_dX2 = -dLog_dX(Twa * res.t_ab, Twb.inverse());

    // finite difference checking
    //            Eigen::Matrix<Scalar,6,6> J_fd;
    //            Scalar dEps = 1e-10;
    //            for(int ii = 0; ii < 6 ; ii++) {
    //                Eigen::Matrix<Scalar,6,1> delta;
    //                delta.setZero();
    //                delta[ii] = dEps;
    //                // const Vector6t pPlus = SE3t::log(Twa*SE3t::exp(delta) * res.Tab * Twb.inverse());
    //                const Vector6t pPlus = log_decoupled(exp_decoupled(Twa,delta) * res.Tab, Twb);
    //                delta[ii] = -dEps;
    //                // const Vector6t pMinus = SE3t::log(Twa*SE3t::exp(delta) * res.Tab * Twb.inverse());
    //                const Vector6t pMinus = log_decoupled(exp_decoupled(Twa,delta) * res.Tab, Twb);
    //                J_fd.col(ii) = (pPlus-pMinus)/(2*dEps);
    //            }
    //            std::cout << "Jbinary:" << res.dZ_dX1 << std::endl;
    //            std::cout << "Jbinary_fd:" << J_fd << std::endl;

    // m_Rpp.template segment<BinaryResidual::ResSize>(res.ResidualOffset) = (Twa*res.Tab*Twb.inverse()).log();
    m_Rpp.template segment<BinaryResidual::kResSize>(res.residual_offset) = log_decoupled(Twa*res.t_ab, Twb);
  }

  for( UnaryResidual& res : m_vUnaryResiduals ){
    const SE3t& Twp = m_vPoses[res.pose_id].t_wp;
    // res.dZ_dX = dLog_dX(Twp, res.Twp.inverse());
    res.dz_dx = dLog_decoupled_dX(Twp, res.t_wp);

    //        Eigen::Matrix<Scalar,6,6> J_fd;
    //        Scalar dEps = 1e-10;
    //        for(int ii = 0; ii < 6 ; ii++) {
    //            Eigen::Matrix<Scalar,6,1> delta;
    //            delta.setZero();
    //            delta[ii] = dEps;
    //            // const Vector6t pPlus = SE3t::log(Twa*SE3t::exp(delta) * res.Tab * Twb.inverse());
    //            const Vector6t pPlus = log_decoupled(exp_decoupled(Twp,delta) , res.Twp);
    //            delta[ii] = -dEps;
    //            // const Vector6t pMinus = SE3t::log(Twa*SE3t::exp(delta) * res.Tab * Twb.inverse());
    //            const Vector6t pMinus = log_decoupled(exp_decoupled(Twp,delta) , res.Twp);
    //            J_fd.col(ii) = (pPlus-pMinus)/(2*dEps);
    //        }
    //        std::cout << "Junary:" << res.dZ_dX << std::endl;
    //        std::cout << "Junary_fd:" << J_fd << std::endl;

    m_Ru.template segment<UnaryResidual::kResSize>(res.residual_offset) = log_decoupled(Twp, res.t_wp);
    // m_Ru.template segment<UnaryResidual::ResSize>(res.ResidualOffset) = (Twp*res.Twp.inverse()).log();
  }

  m_dImuError = 0;
  for( ImuResidual& res : m_vImuResiduals ){
    // set up the initial pose for the integration
    const Vector3t gravity = GetGravityVector(m_Imu.g);

    const Pose& poseA = m_vPoses[res.pose1_id];
    const Pose& poseB = m_vPoses[res.pose2_id];

    Eigen::Matrix<Scalar,10,6> jb_q;
    // Eigen::Matrix<Scalar,10,10> jb_y;
    ImuPose imuPose = ImuResidual::IntegrateResidual(poseA,res.measurements,poseA.b.template head<3>(),
                                                     poseA.b.template tail<3>(),gravity,res.poses,&jb_q/*,&jb_y*/);
    Scalar totalDt = res.measurements.back().time - res.measurements.front().time;
    const SE3t Tab = poseA.t_wp.inverse()*imuPose.t_wp;
    const SE3t& Twa = poseA.t_wp;
    const SE3t& Twb = poseB.t_wp;

    // now given the poses, calculate the jacobians.
    // First subtract gravity, initial pose and velocity from the delta T and delta V
    SE3t Tab_0 = imuPose.t_wp;
    Tab_0.translation() -=(-gravity*0.5*powi(totalDt,2) + poseA.v_w*totalDt);   // subtract starting velocity and gravity
    Tab_0 = poseA.t_wp.inverse() * Tab_0;                                       // subtract starting pose
    // Augment the velocity delta by subtracting effects of gravity
    Vector3t Vab_0 = imuPose.v_w - poseA.v_w;
    Vab_0 += gravity*totalDt;
    // rotate the velocity delta so that it starts from orientation=Ident
    Vab_0 = poseA.t_wp.so3().inverse() * Vab_0;

    // derivative with respect to the start pose
    res.residual.setZero();
    res.dz_dx1.setZero();
    res.dz_dx2.setZero();
    res.dZ_dG.setZero();
    res.dz_db.setZero();

    // SE3t Tstar(Sophus::SO3Group<Scalar>(),(poseA.V*totalDt - 0.5*gravity*powi(totalDt,2)));

    // calculate the derivative of the lie log with respect to the tangent plane at Twa
    const Eigen::Matrix<Scalar,6,7> se3_log = dLog_dSE3(imuPose.t_wp*Twb.inverse());
    // std::cout << "se3_log: " << std::endl << se3_log << std::endl;

    Eigen::Matrix<Scalar,7,6> dSE3_dX1;
    dSE3_dX1.setZero();
    dSE3_dX1.template block<3,3>(0,0) = Twa.so3().matrix();
    // for this derivation  refer to page 16 of notes
    dSE3_dX1.template block<3,3>(0,3) = dqx_dq<Scalar>((Twa).unit_quaternion(),Tab_0.translation() - Tab_0.so3()*Twb.so3().inverse()*Twb.translation())*
        dq1q2_dq2(Twa.unit_quaternion()) *
        dqExp_dw<Scalar>(Eigen::Matrix<Scalar,3,1>::Zero());
    dSE3_dX1.template block<4,3>(3,3) = dq1q2_dq1((Tab_0.so3() * Twb.so3().inverse()).unit_quaternion()) *
        dq1q2_dq2(Twa.unit_quaternion()) *
        dqExp_dw<Scalar>(Eigen::Matrix<Scalar,3,1>::Zero());

    // std::cout << "jb_q: " << std::endl << jb_q << std::endl;
    // TODO: the block<3,3>(0,0) jacobian is incorrect here due to multiplication by Twb.inverse(). Fix this
    jb_q.template block<4,3>(3,0) = dq1q2_dq1(Twb.inverse().unit_quaternion())* jb_q.template block<4,3>(3,0) ;
    res.dz_db.template block<6,6>(0,0) = se3_log * jb_q.template block<7,6>(0,0);     // dt/dB
    res.dz_db.template block<3,6>(6,0) = jb_q.template block<3,6>(7,0);     // dV/dB
    res.dz_db.template block<6,6>(9,0) = Eigen::Matrix<Scalar,6,6>::Identity();   // dB/dB

    // Twa^-1 is multiplied here as we need the velocity derivative in the frame of pose A, as the log is taken from this frame
    res.dz_dx1.template block<3,3>(0,6) = Matrix3t::Identity()*totalDt;
    for( int ii = 0; ii < 3 ; ++ii ){
      res.dz_dx1.template block<3,1>(6,3+ii) = Twa.so3().matrix() * Sophus::SO3Group<Scalar>::generator(ii) * Vab_0;
    }
    res.dz_dx1.template block<3,3>(6,6) = Matrix3t::Identity();
    res.dz_dx1.template block<6,6>(0,0) =  se3_log*dSE3_dX1;
    res.dz_dx1.template block<ImuResidual::kResSize,6>(0,9) = res.dz_db;

    // the - sign is here because of the exp(-x) within the log
    res.dz_dx2.template block<6,6>(0,0) = -dLog_dX(imuPose.t_wp,Twb.inverse());
    res.dz_dx2.template block<3,3>(6,6) = -Matrix3t::Identity();
    res.dz_dx2.template block<6,6>(9,9) = -Eigen::Matrix<Scalar,6,6>::Identity();
    // res.dZ_dX2.template block<6,6>(9,9).setZero();

    const Eigen::Matrix<Scalar,3,2> dGravity = dGravity_dDirection(m_Imu.g);
    res.dZ_dG.template block<3,2>(0,0) = /*dLog.template block<3,3>(0,0) * Twa.so3().inverse().matrix() **/
        -0.5*powi(totalDt,2)*Matrix3t::Identity()*dGravity;
    res.dZ_dG.template block<3,2>(6,0) = -totalDt*Matrix3t::Identity()*dGravity;

    res.residual.template head<6>() = SE3t::log(Twa*Tab*Twb.inverse());
    res.residual.template segment<3>(6) = imuPose.v_w - poseB.v_w;
    res.residual.template segment<6>(9) = poseA.b - poseB.b;

    if( (CalibSize > 2 || PoseSize > 15) && m_bEnableTranslation == false ){
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

    if( PoseSize > 15 ){
      res.dz_dx1.template block<9,6>(0,15) = res.dz_dx1.template block<9, 6>(0,0);
      res.dz_dx2.template block<9,6>(0,15) = res.dz_dx2.template block<9, 6>(0,0);

      res.residual.template segment<6>(15) = SE3t::log(poseA.t_vs*poseB.t_vs.inverse());    //Tvs bias residual

      res.dz_dx1.template block<6,6>(15,15) = -dLog_dX(SE3t(), poseA.t_vs * poseB.t_vs.inverse());
      res.dz_dx2.template block<6,6>(15,15) = dLog_dX(poseA.t_vs * poseB.t_vs.inverse(), SE3t());

      //if( m_bEnableTranslation == false ){
      res.residual.template segment<3>(15).setZero();
      // removing translation elements of Tvs
      res.dz_dx1.template block<6,3>(15,15).setZero();
      res.dz_dx2.template block<6,3>(15,15).setZero();
      res.dz_dx1.template block<9,3>(0,15).setZero();
      res.dz_dx2.template block<9,3>(0,15).setZero();
      //}

      // std::cout << "res.dZ_dX1: " << std::endl << res.dZ_dX1.format(cleanFmt) << std::endl;

      /*{
              Scalar dEps = 1e-9;
              Eigen::Matrix<Scalar,6,6> drTvs_dX1_dF;
              for(int ii = 0 ; ii < 6 ; ii++){
                  Vector6t eps = Vector6t::Zero();
                  eps[ii] = dEps;
                  Vector6t resPlus = SE3t::log(SE3t::exp(-eps)*poseA.Tvs * poseB.Tvs.inverse());
                  eps[ii] = -dEps;
                  Vector6t resMinus = SE3t::log(SE3t::exp(-eps)*poseA.Tvs * poseB.Tvs.inverse());
                  drTvs_dX1_dF.col(ii) = (resPlus-resMinus)/(2*dEps);
              }
              std::cout << "drTvs_dX1 = [" << res.dZ_dX1.template block<6,6>(15,15).format(cleanFmt) << "]" << std::endl;
              std::cout << "drTvs_dX1_dF = [" << drTvs_dX1_dF.format(cleanFmt) << "]" << std::endl;
              std::cout << "drTvs_dX1 - drTvs_dX1_dF = [" << (res.dZ_dX1.template block<6,6>(15,15)- drTvs_dX1_dF).format(cleanFmt) << "]" << std::endl;
          }
          {
              Scalar dEps = 1e-9;
              Eigen::Matrix<Scalar,6,6> drTvs_dX2_dF;
              for(int ii = 0 ; ii < 6 ; ii++){
                  Vector6t eps = Vector6t::Zero();
                  eps[ii] = dEps;
                  Vector6t resPlus = SE3t::log(poseA.Tvs * (SE3t::exp(-eps)*poseB.Tvs).inverse());
                  eps[ii] = -dEps;
                  //Vector6t resMinus = SE3t::log(SE3t::exp(-eps)*poseA.Tvs * poseB.Tvs.inverse());
                  Vector6t resMinus = SE3t::log(poseA.Tvs * (SE3t::exp(-eps)*poseB.Tvs).inverse());
                  drTvs_dX2_dF.col(ii) = (resPlus-resMinus)/(2*dEps);
              }
              std::cout << "drTvs_dX2 = [" << res.dZ_dX2.template block<6,6>(15,15).format(cleanFmt) << "]" << std::endl;
              std::cout << "drTvs_dX2_dF = [" << drTvs_dX2_dF.format(cleanFmt) << "]" << std::endl;
              std::cout << "drTvs_dX2 - drTvs_dX2_dF = [" << (res.dZ_dX2.template block<6,6>(15,15)- drTvs_dX2_dF).format(cleanFmt) << "]" << std::endl;
          }*/
    }else{
      res.dz_dy = res.dz_dx1.template block<ImuResidual::kResSize, 6>(0,0) +
          res.dz_dx2.template block<ImuResidual::kResSize, 6>(0,0);
      if( m_bEnableTranslation == false ){
        res.dz_dy.template block<ImuResidual::kResSize, 3>(0,0).setZero();
      }
    }

    // std::cout << "BUILD imu res between " << res.PoseAId << " and " << res.PoseBId << ":" << res.Residual.transpose () << std::endl;

    //if(poseA.IsActive == false || poseB.IsActive == false){
    //    std::cout << "PRIOR RESIDUAL: ";
    //}
    //std::cout << "Residual for res " << res.ResidualId << " : " << res.Residual.transpose() << std::endl;
    m_vErrors.push_back(res.residual.squaredNorm());

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
    m_Ri.template segment<ImuResidual::kResSize>(res.residual_offset) = res.residual;
  }

  // get the sigma for robust norm calculation.
  if( m_vErrors.size() > 0 ){
    auto it = m_vErrors.begin()+std::floor(m_vErrors.size()/2);
    std::nth_element(m_vErrors.begin(),it,m_vErrors.end());
    const Scalar dSigma = sqrt(*it);
    // See "Parameter Estimation Techniques: A Tutorial with Application to Conic
    // Fitting" by Zhengyou Zhang. PP 26 defines this magic number:
    const Scalar c_huber = 1.2107*dSigma;

    // now go through the measurements and assign weights
    for( ImuResidual& res : m_vImuResiduals ){
      // calculate the huber norm weight for this measurement
      const Scalar e = res.residual.norm();
      //res.W *= e > c_huber ? c_huber/e : 1.0;
      m_dImuError += res.residual.norm() * res.weight;
    }
  }
  m_vErrors.clear();

  // std::cout << "Jacobian evaluation took " << Toc(dTime) << " seconds. " << std::endl;



  //TODO : The transpose insertions here are hideously expensive as they are not in order.
  // find a way around this.

  // here we sort the measurements and insert them per pose and per landmark, this will mean
  // each insert operation is O(1)
  dTime = Tic();
  for( Pose& pose : m_vPoses ){
    if( pose.is_active ) {
      // sort the measurements by id so the sparse insert is O(1)
      std::sort(pose.proj_residuals.begin(), pose.proj_residuals.end());
      for( const int id: pose.proj_residuals ) {
        const ProjectionResidual& res = m_vProjResiduals[id];
        // insert the jacobians into the sparse matrices
        // The weight is only multiplied by the transpose matrix, this is so we can perform Jt*W*J*dx = Jt*W*r
        auto dZ_dP = res.x_meas_id == pose.id ? res.dz_dx_meas : res.dz_dx_ref;
        if( res.x_ref_id == pose.id ){

          //std::cout << "Adding reference jacobian for pose id " << pose.Id << " and residual id " << res.ResidualId << " wih ref pose id " << res.RefPoseId << " and meas pose id " << res.MeasPoseId << std::endl;
          // std::cout << "Jacobian is " << std::endl << dZ_dP << std::endl;
          // dZ_dP.setZero();
          // dZ_dP *= -1;
        }else
        {
          //std::cout << "Adding measurement jacobian for pose id " << pose.Id << " and residual id " << res.ResidualId << " wih ref pose id " << res.RefPoseId << " and meas pose id " << res.MeasPoseId <<  std::endl;
          // dZ_dP.setZero();
        }
        m_Jpr.insert( res.residual_id, pose.opt_id ).setZero().template block<2,6>(0,0) = dZ_dP;
        m_Jprt.insert( pose.opt_id, res.residual_id ).setZero().template block<6,2>(0,0) = dZ_dP.transpose() * res.weight;

        if(PoseSize == 21){
          //                    const auto& dZ_dTvs = res.MeasPoseId == pose.Id ? res.dZ_dTvs_m : res.dZ_dTvs_r;
          //                    m_Jpr.coeffRef( res.ResidualId, pose.OptId ).template block<2,6>(0,15) = dZ_dTvs.template block(0,0,2,6);
          //                    m_Jprt.coeffRef( pose.OptId, res.ResidualId ).template block<6,2>(15,0) = dZ_dTvs.transpose().template block(0,0,6,2) * res.W;
          //                    JprVal.template block<2,6>(0,15) = res.dZ_dTvs.template block(0,0,2,6);
          //                    JprtVal.template block<6,2>(15,0) = res.dZ_dTvs.transpose().template block(0,0,6,2) * res.W;
        }
      }

      // add the pose/pose constraints
      std::sort(pose.binary_residuals.begin(), pose.binary_residuals.end());
      for( const int id: pose.binary_residuals ) {
        const BinaryResidual& res = m_vBinaryResiduals[id];
        const Eigen::Matrix<Scalar,6,6>& dZ_dZ = res.x1_id == pose.id ? res.dz_dx1 : res.dZ_dX2;
        m_Jpp.insert( res.residual_id, pose.opt_id ).setZero().template block<6,6>(0,0) = dZ_dZ;
        m_Jppt.insert( pose.opt_id, res.residual_id ).setZero().template block<6,6>(0,0) = dZ_dZ.transpose() * res.weight;
      }

      // add the unary constraints
      std::sort(pose.unary_residuals.begin(), pose.unary_residuals.end());
      for( const int id: pose.unary_residuals ) {
        const UnaryResidual& res = m_vUnaryResiduals[id];
        m_Ju.insert( res.residual_id, pose.opt_id ).setZero().template block<6,6>(0,0) = res.dz_dx;
        m_Jut.insert( pose.opt_id, res.residual_id ).setZero().template block<6,6>(0,0) = res.dz_dx.transpose() * res.weight;
      }

      std::sort(pose.inertial_residuals.begin(), pose.inertial_residuals.end());
      for( const int id: pose.inertial_residuals ) {
        const ImuResidual& res = m_vImuResiduals[id];
        Eigen::Matrix<Scalar,ImuResidual::kResSize,PoseSize> dZ_dZ = res.pose1_id == pose.id ? res.dz_dx1 : res.dz_dx2;
        m_Ji.insert( res.residual_id, pose.opt_id ).setZero().template block<ImuResidual::kResSize,PoseSize>(0,0) = dZ_dZ;
        // this down weights the velocity error
        dZ_dZ.template block<3,PoseSize>(6,0) *= 0.1;
        // up weight the Tvs translation prior
        if(PoseSize > 15){
          dZ_dZ.template block<3,PoseSize>(15,0) *= (100/**m_dTvsTransPrior*/);
          dZ_dZ.template block<3,PoseSize>(18,0) *= (10/**m_dTvsRotPrior*/);
          //                    std::cout << "m_dTvsTransPrior: " <<  m_dTvsTransPrior << " m_dTvsRotPrior:" << m_dTvsRotPrior << std::endl;
          //                    m_dTvsTransPrior += 0.00001;
          //                    m_dTvsRotPrior += 0.00002;
        }
        m_Jit.insert( pose.opt_id, res.residual_id ).setZero().template block<PoseSize,ImuResidual::kResSize>(0,0) = dZ_dZ.transpose() * res.weight;
      }
    }
  }

  // fill in calibration jacobians
  if( CalibSize > 0){
    for( const ImuResidual& res : m_vImuResiduals ){
      // include gravity terms (t total)
      if( CalibSize > 0 ){
        // std::cout << "Residual " << res.ResidualId << " : dZ_dG: " << res.dZ_dG.template block<9,2>(0,0).transpose() << std::endl;
        Eigen::Matrix<Scalar,9,2> dZ_dG = res.dZ_dG;
        m_Jki.insert(res.residual_id, 0 ).setZero().template block(0,0,9,2) = dZ_dG.template block(0,0,9,2);
        // this down weights the velocity error
        dZ_dG.template block<3,2>(6,0) *= 0.1;
        m_Jkit.insert( 0, res.residual_id ).setZero().template block(0,0,2,9) = dZ_dG.transpose().template block(0,0,2,9) * res.weight;
      }

      // include Y terms
      if( CalibSize > 2 ){
        m_Jki.coeffRef(res.residual_id,0).setZero().template block(0,2,ImuResidual::kResSize, 6) = res.dz_dy.template block(0,0,ImuResidual::kResSize, 6);
        m_Jkit.coeffRef(0,res.residual_id).setZero().template block(2,0, 6, ImuResidual::kResSize) = res.dz_dy.template block(0,0,ImuResidual::kResSize, 6).transpose() * res.weight;
        // m_Jkit.coeffRef(0,res.ResidualId).template block(2,0,6,9) = res.dZ_dB.transpose().template block(0,0,6,9) * res.W;
      }
    }

    for( const ProjectionResidual& res : m_vProjResiduals ){
      // include imu to camera terms (6 total)
      if( CalibSize > 8 ){
        const Eigen::Matrix<Scalar,2,5>& dZ_dK = res.dz_dcam_params;
        m_Jkpr.coeffRef(res.residual_id,0).setZero().template block(0,8,2,5) = dZ_dK.template block(0,0,2,5);
        m_Jkprt.coeffRef(0,res.residual_id).setZero().template block(8,0,5,2) = dZ_dK.template block(0,0,2,5).transpose() * res.weight;
        // m_Jkit.coeffRef(0,res.ResidualId).template block(2,0,6,9) = res.dZ_dB.transpose().template block(0,0,6,9) * res.W;
      }
    }
  }

  for( Landmark& lm : m_vLandmarks ){
    if( lm.is_active ){
      // sort the measurements by id so the sparse insert is O(1)
      std::sort(lm.proj_residuals.begin(), lm.proj_residuals.end());
      for( const int id: lm.proj_residuals ) {
        const ProjectionResidual& res = m_vProjResiduals[id];
        //                std::cout << "      Adding jacobian cell for measurement " << pMeas->MeasurementId << " in landmark " << pMeas->LandmarkId << std::endl;
        m_Jl.insert( res.residual_id, lm.opt_id ) = res.dz_dlm;
        m_Jlt.insert( lm.opt_id, res.residual_id ) = res.dz_dlm.transpose() * res.weight;
      }
    }
  }

  // std::cout << "Jacobian insertion took " << Toc(dTime) << " seconds. " << std::endl;
}

// specializations
// template class BundleAdjuster<REAL_TYPE, ba::NOT_USED,9,8>;
template class BundleAdjuster<REAL_TYPE, 1,6,0>;
//template class BundleAdjuster<REAL_TYPE, 1,15,8>;
template class BundleAdjuster<REAL_TYPE, 1,15,2>;
//template class BundleAdjuster<REAL_TYPE, 1,21,2>;
// template class BundleAdjuster<double, 3,9>;


}


