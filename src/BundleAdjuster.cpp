#include <ba/BundleAdjuster.h>


namespace ba {

#define DAMPING 0.5

///////////////////////////////////////////////////////////////////////////////////////////////
template< typename Scalar,int LmSize, int PoseSize, int CalibSize >
void BundleAdjuster<Scalar,LmSize,PoseSize,CalibSize>::_EvaluateResiduals()
{
    m_dProjError = 0;
   for( ProjectionResidual& res : m_vProjResiduals ){
       Landmark& lm = m_vLandmarks[res.LandmarkId];
       Pose& pose = m_vPoses[res.MeasPoseId];
       Pose& refPose = m_vPoses[res.RefPoseId];
       const SE3t Tsw_m = pose.GetTsw(res.CameraId, m_Rig, PoseSize >= 21);
       const SE3t Tws_r = refPose.GetTsw(lm.RefCamId,m_Rig, PoseSize >= 21).inverse();

       const Vector2t p = m_Rig.cameras[res.CameraId].camera.Transfer3D(Tsw_m*Tws_r, lm.Xs.template head<3>(),lm.Xs(3));
       res.Residual = res.Z - p;
       m_dProjError += res.Residual.norm() * res.W;
   }

   m_dImuError = 0;
   for( ImuResidual& res : m_vImuResiduals ){
       // set up the initial pose for the integration
       const Vector3t gravity = GetGravityVector(m_Imu.G);

       const Pose& poseA = m_vPoses[res.PoseAId];
       const Pose& poseB = m_vPoses[res.PoseBId];

       Eigen::Matrix<Scalar,10,6> jb_q;
       // Eigen::Matrix<Scalar,10,10> jb_y;
       const ImuPose imuPose = ImuResidual::IntegrateResidual(poseA,res.Measurements,poseA.B.template head<3>(),
                                                              poseA.B.template tail<3>(),gravity,res.Poses,&jb_q);
       const SE3t Tab = poseA.Twp.inverse()*imuPose.Twp;
       const SE3t& Twa = poseA.Twp;
       const SE3t& Twb = poseB.Twp;

       res.Residual.template head<6>() = log_decoupled(Twa*Tab,Twb);
       res.Residual.template segment<3>(6) = imuPose.V - poseB.V;
       res.Residual.template segment<6>(9) = poseA.B - poseB.B;
       m_dImuError += res.Residual.norm() * res.W;
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////
template< typename Scalar,int LmSize, int PoseSize, int CalibSize >
void BundleAdjuster<Scalar,LmSize,PoseSize,CalibSize>::Solve(const unsigned int uMaxIter)
{
    Eigen::IOFormat cleanFmt(2, 0, ", ", ";\n" , "" , "");


    for( unsigned int kk = 0 ; kk < uMaxIter ; kk++){
        double dItTime = Tic();

        Scalar dTime = Tic();
        _BuildProblem();
        std::cout << "Build problem took " << Toc(dTime) << " seconds." << std::endl;

        dTime = Tic();

        const unsigned int uNumPoses = m_uNumActivePoses;
        const unsigned int uNumPoseParams = uNumPoses*PoseSize;
        const unsigned int uNumLm = m_uNumActiveLandmakrs;
//        const unsigned int uNumMeas = m_vMeasurements.size();
        // calculate bp and bl
        Scalar dMatTime = Tic();
        VectorXt bp(uNumPoseParams);
        VectorXt bk;
        VectorXt bl;
        Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, LmSize, LmSize> > Vi(uNumLm, uNumLm);
        VectorXt rhs_p(uNumPoseParams + CalibSize);
        Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, LmSize, PoseSize> > Jlt_Jpr(uNumLm, uNumPoses);
        Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, PoseSize, LmSize> > Jprt_Jl_Vi(uNumPoses, uNumLm);
        Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> S(uNumPoseParams + CalibSize,uNumPoseParams + CalibSize);
        std::cout << "  Rhs vector mult took " << Toc(dMatTime) << " seconds." << std::endl;

        dMatTime = Tic();

        // TODO: suboptimal, the matrices are symmetric. We should only multipl one half
        Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, PoseSize, PoseSize> > U(uNumPoses, uNumPoses);
        U.setZero();
        bp.setZero();
        S.setZero();
        rhs_p.setZero();

        if( m_vProjResiduals.size() > 0 ){
            Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, PoseSize, PoseSize> > Jprt_Jpr(uNumPoses, uNumPoses);
            Eigen::SparseBlockProduct(m_Jprt, m_Jpr, U);

            VectorXt Jprt_Rpr(uNumPoseParams);
            Eigen::SparseBlockVectorProductDenseResult(m_Jprt, m_Rpr, Jprt_Rpr);
            bp += Jprt_Rpr;
        }

        // add the contribution from the binary terms if any
        if( m_vBinaryResiduals.size() > 0 ) {
            Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, PoseSize, PoseSize> > Jppt_Jpp(uNumPoses, uNumPoses);
            Eigen::SparseBlockProduct(m_Jppt ,m_Jpp, Jppt_Jpp);
            auto Temp = U;
            Eigen::SparseBlockAdd(Temp,Jppt_Jpp,U);

            VectorXt Jppt_Rpp(uNumPoseParams);
            Eigen::SparseBlockVectorProductDenseResult(m_Jppt, m_Rpp, Jppt_Rpp);
            bp += Jppt_Rpp;
        }

        // add the contribution from the unary terms if any
        if( m_vUnaryResiduals.size() > 0 ) {
            Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, PoseSize, PoseSize> > Jut_Ju(uNumPoses, uNumPoses);
            Eigen::SparseBlockProduct(m_Jut, m_Ju, Jut_Ju);
            auto Temp = U;
            Eigen::SparseBlockAdd(Temp, Jut_Ju, U);

            VectorXt Jut_Ru(uNumPoseParams);
            Eigen::SparseBlockVectorProductDenseResult(m_Jut, m_Ru, Jut_Ru);
            bp += Jut_Ru;

//                Eigen::LoadDenseFromSparse(U,S);
//                std::cout << "Dense S matrix is " << S.format(cleanFmt) << std::endl;
        }

        // add the contribution from the imu terms if any
        if( m_vImuResiduals.size() > 0 ) {
            Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, PoseSize, PoseSize> > Jit_Ji(uNumPoses, uNumPoses);
            Eigen::SparseBlockProduct(m_Jit, m_Ji, Jit_Ji);
            auto Temp = U;
            Eigen::SparseBlockAdd(Temp, Jit_Ji, U);

            VectorXt Jit_Ri(uNumPoseParams);
            Eigen::SparseBlockVectorProductDenseResult(m_Jit, m_Ri, Jit_Ri);
            bp += Jit_Ri;
                // Eigen::LoadDenseFromSparse(U,S);
                // std::cout << "Dense S matrix is " << S.format(cleanFmt) << std::endl;
        }

        if( LmSize > 0 && uNumLm > 0) {
            bl.resize(uNumLm*LmSize);
            double dSchurTime = Tic();
            Eigen::SparseBlockVectorProductDenseResult(m_Jlt, m_Rpr, bl);
            // std::cout << "Eigen::SparseBlockVectorProductDenseResult(m_Jlt, m_Rpr, bl); took  " << Toc(dSchurTime) << " seconds."  << std::endl;

            dSchurTime = Tic();
            Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, LmSize, LmSize> > V(uNumLm, uNumLm);
            Eigen::SparseBlockProduct(m_Jlt,m_Jl,V);
            // std::cout << "Eigen::SparseBlockProduct(m_Jlt,m_Jl,V);; took  " << Toc(dSchurTime) << " seconds."  << std::endl;

            dSchurTime = Tic();
            Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, PoseSize, LmSize> > Jprt_Jl(uNumPoses, uNumLm);
            Eigen::SparseBlockProduct(m_Jprt,m_Jl,Jprt_Jl);
            Eigen::SparseBlockProduct(m_Jlt,m_Jpr,Jlt_Jpr);
            // Jlt_Jpr = Jprt_Jl.transpose();
            // std::cout << "Eigen::SparseBlockProduct(m_Jprt,m_Jl,Wp) and Wpt = Wp.transpose(); took  " << Toc(dSchurTime) << " seconds."  << std::endl;


            dSchurTime = Tic();
            // calculate the inverse of the map hessian (it should be diagonal, unless a measurement is of more than
            // one landmark, which doesn't make sense)
            for(size_t ii = 0 ; ii < uNumLm ; ii++){
                if(LmSize == 1){
                    if(V.coeffRef(ii,ii)(0,0) < 1e-6){
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
            S.template block(0, 0, uNumPoseParams, uNumPoseParams ) = dU - dJprt_Jl_Vi_Jlt_Jpr;
            // std::cout << "Eigen::SparseBlockSubtractDenseResult(U, WV_invWt, S.template block(0, 0, uNumPoseParams, uNumPoseParams )) took  " << Toc(dSchurTime) << " seconds."  << std::endl;

            // now form the rhs for the pose equations
            dSchurTime = Tic();
            VectorXt Jpt_Jl_Vi_bl(uNumPoseParams);
            Eigen::SparseBlockVectorProductDenseResult(Jprt_Jl_Vi, bl, Jpt_Jl_Vi_bl);
            // std::cout << "Eigen::SparseBlockVectorProductDenseResult(Wp_V_inv, bl, WV_inv_bl) took  " << Toc(dSchurTime) << " seconds."  << std::endl;

            rhs_p.template head(uNumPoseParams) = bp - Jpt_Jl_Vi_bl;

            // std::cout << "Dense S matrix is " << S.format(cleanFmt) << std::endl;
            // std::cout << "Dense rhs matrix is " << rhs_p.transpose().format(cleanFmt) << std::endl;

        }else{
            Eigen::LoadDenseFromSparse(U, S.template block(0, 0, uNumPoseParams, uNumPoseParams));
            rhs_p.template head(uNumPoseParams) = bp;                        
        }

         std::cout << "  Rhs calculation and schur complement took " << Toc(dMatTime) << " seconds." << std::endl;

        Eigen::MatrixXd dJki;
        dJki.resize(m_vImuResiduals.size() * ImuResidual::ResSize,CalibSize);
        Eigen::LoadDenseFromSparse(m_Jki,dJki);
        // std::cout << "Dense dJki matrix is " << dJki.format(cleanFmt) << std::endl;

        // fill in the calibration components if any
        if( CalibSize && m_vImuResiduals.size() > 0 ){
            Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,CalibSize,CalibSize> > Jkit_Jki(1, 1);
            Eigen::SparseBlockProduct(m_Jkit, m_Jki, Jkit_Jki);
            Eigen::LoadDenseFromSparse(Jkit_Jki, S.template block<CalibSize, CalibSize>(uNumPoseParams, uNumPoseParams));

            Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,PoseSize,CalibSize> > Jit_Jki(uNumPoses, 1);
            Eigen::SparseBlockProduct(m_Jit, m_Jki, Jit_Jki);
            Eigen::LoadDenseFromSparse(Jit_Jki, S.template block(0, uNumPoseParams, uNumPoseParams, CalibSize));

            S.template block(uNumPoseParams, 0, CalibSize, uNumPoseParams) = S.template block(0, uNumPoseParams, uNumPoseParams, CalibSize).transpose();

            // and the rhs for the calibration params
            bk.resize(CalibSize,1);
            Eigen::SparseBlockVectorProductDenseResult(m_Jkit, m_Ri, bk);
            rhs_p.template tail<CalibSize>() = bk;
        }

        if( CalibSize > 2){
            Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,CalibSize,CalibSize> > Jkprt_Jkpr(1, 1);
            Eigen::SparseBlockProduct(m_Jkprt, m_Jkpr, Jkprt_Jkpr);
            Eigen::MatrixXd dJkprt_Jkpr(CalibSize, CalibSize);
            Eigen::LoadDenseFromSparse(Jkprt_Jkpr, dJkprt_Jkpr);
            S.template block<CalibSize, CalibSize>(uNumPoseParams, uNumPoseParams) += dJkprt_Jkpr;
            std::cout << "dJkprt_Jkpr: " << dJkprt_Jkpr << std::endl;

            Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,PoseSize,CalibSize> > Jprt_Jkpr(uNumPoses, 1);
            Eigen::SparseBlockProduct(m_Jprt, m_Jkpr, Jprt_Jkpr);
            Eigen::MatrixXd dJprt_Jkpr(PoseSize*uNumPoses, CalibSize);
            Eigen::LoadDenseFromSparse(Jprt_Jkpr, dJprt_Jkpr);
            std::cout << "dJprt_Jkpr: " << dJprt_Jkpr << std::endl;
            S.template block(0, uNumPoseParams, uNumPoseParams, CalibSize) += dJprt_Jkpr;
            S.template block(uNumPoseParams, 0, CalibSize, uNumPoseParams) += dJprt_Jkpr.transpose();

            bk.resize(CalibSize,1);
            Eigen::SparseBlockVectorProductDenseResult(m_Jkprt, m_Rpr, bk);
            rhs_p.template tail<CalibSize>() += bk;

            // schur complement
            Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, CalibSize, LmSize> > Jkprt_Jl(1, uNumLm);
            Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, LmSize, CalibSize> > Jlt_Jkpr(uNumLm, 1);
            Eigen::SparseBlockProduct(m_Jkprt,m_Jl,Jkprt_Jl);
            Eigen::SparseBlockProduct(m_Jlt,m_Jkpr,Jlt_Jkpr);
            //Jlt_Jkpr = Jkprt_Jl.transpose();

            Eigen::MatrixXd dJpt_Jl_Vi_Jlt_Jkpr(PoseSize*uNumPoses, CalibSize);
            Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, PoseSize, CalibSize> > Jpt_Jl_Vi_Jlt_Jkpr(uNumPoses, 1);
            Eigen::SparseBlockProduct(Jprt_Jl_Vi,Jlt_Jkpr,Jpt_Jl_Vi_Jlt_Jkpr);
            Eigen::LoadDenseFromSparse(Jpt_Jl_Vi_Jlt_Jkpr, dJpt_Jl_Vi_Jlt_Jkpr);
            std::cout << "dJpt_Jl_Vi_Jlt_Jkpr: " << dJpt_Jl_Vi_Jlt_Jkpr << std::endl;
            S.template block(0, uNumPoseParams, uNumPoseParams, CalibSize) -= dJpt_Jl_Vi_Jlt_Jkpr;
            S.template block(uNumPoseParams, 0, CalibSize, uNumPoseParams) -= dJpt_Jl_Vi_Jlt_Jkpr.transpose();

            Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, CalibSize, LmSize> > Jkprt_Jl_Vi(1, uNumLm);
            Eigen::SparseBlockProduct(Jkprt_Jl,Vi,Jkprt_Jl_Vi);
            Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar, CalibSize, CalibSize> > Jkprt_Jl_Vi_Jlt_Jkpr(1, 1);
            Eigen::SparseBlockProduct(Jkprt_Jl_Vi,Jlt_Jkpr,Jkprt_Jl_Vi_Jlt_Jkpr);
            Eigen::MatrixXd dJkprt_Jl_Vi_Jlt_Jkpr(CalibSize, CalibSize);
            Eigen::LoadDenseFromSparse(Jkprt_Jl_Vi_Jlt_Jkpr, dJkprt_Jl_Vi_Jlt_Jkpr);
            std::cout << "dJkprt_Jl_Vi_Jlt_Jkpr: " << dJkprt_Jl_Vi_Jlt_Jkpr << std::endl;
            S.template block<CalibSize, CalibSize>(uNumPoseParams, uNumPoseParams) -= dJkprt_Jl_Vi_Jlt_Jkpr;

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
        dTime = Tic();
        VectorXt delta_p = uNumPoses == 0 ? VectorXt() : S.ldlt().solve(rhs_p);
//        VectorXt delta_p = uNumPoses == 0 ? VectorXt() : S.inverse() * rhs_p;
         std::cout << "Cholesky solve of " << uNumPoses << " by " << uNumPoses << "matrix took " << Toc(dTime) << " seconds." << std::endl;

        if(uNumLm > 0){
            dTime = Tic();
            VectorXt delta_l;
            delta_l.resize(uNumLm*LmSize);
            VectorXt Wt_delta_p;
            Wt_delta_p.resize(uNumLm*LmSize );
            Eigen::SparseBlockVectorProductDenseResult(Jlt_Jpr,delta_p.head(uNumPoseParams),Wt_delta_p);
            VectorXt rhs_l;
            rhs_l.resize(uNumLm*LmSize );
            rhs_l =  bl - Wt_delta_p;

            for(size_t ii = 0 ; ii < uNumLm ; ii++){
                delta_l.template block<LmSize,1>( ii*LmSize, 0 ).noalias() =  Vi.coeff(ii,ii)*rhs_l.template block<LmSize,1>(ii*LmSize,0);
            }

            // update the landmarks
            for (size_t ii = 0 ; ii < m_vLandmarks.size() ; ii++){
                if( m_vLandmarks[ii].IsActive ){
                    if(LmSize == 1){
                        m_vLandmarks[ii].Xs.template tail<LmSize>() -= delta_l.template segment<LmSize>(m_vLandmarks[ii].OptId*LmSize) * DAMPING;
                        // std::cout << "Delta for landmark " << ii << " is " << delta_l.template segment<LmSize>(m_vLandmarks[ii].OptId*LmSize).transpose() << std::endl;
                        // m_vLandmarks[ii].Xs /= m_vLandmarks[ii].Xs[3];
                    }else{
                        m_vLandmarks[ii].Xs.template head<LmSize>() -= delta_l.template segment<LmSize>(m_vLandmarks[ii].OptId*LmSize);
                    }

                    m_vLandmarks[ii].Xw = MultHomogeneous(m_vPoses[m_vLandmarks[ii].RefPoseId].GetTsw(m_vLandmarks[ii].RefCamId, m_Rig, PoseSize >= 21).inverse(),
                                                          m_vLandmarks[ii].Xs);
                }
            }
            std::cout << "Backsubstitution of " << uNumLm << " landmarks took " << Toc(dTime) << " seconds." << std::endl;
        }

        // update gravity terms if necessary
        if( m_vImuResiduals.size() > 0 ) {
            const VectorXt deltaCalib = delta_p.template tail(CalibSize);
            if(CalibSize > 0){
                m_Imu.G -= deltaCalib.template block<2,1>(0,0) * DAMPING;
                // std::cout << "Gravity delta is " << deltaCalib.template block<2,1>(0,0).transpose() << " gravity is: " << m_Imu.G.transpose() << std::endl;
            }

            if(CalibSize > 2){
                m_Imu.Tvs = m_Imu.Tvs*SE3t::exp(-deltaCalib.template block<6,1>(2,0)); //exp_decoupled<Scalar>(m_Imu.Tvs,-deltaCalib.template block<6,1>(2,0));
                m_Rig.cameras[0].T_wc = m_Imu.Tvs;
                std::cout << "Tvs delta is " << -deltaCalib.template block<6,1>(2,0).transpose() << std::endl;
            }

            // update bias terms if necessary
            // if(CalibSize > 2){
                // m_Imu.Bg -= deltaCalib.template block<3,1>(2,0);//.template tail(CalibSize);
                // std::cout << "Bg delta is " << deltaCalib.template block<3,1>(2,0).transpose() << " bg is: " << m_Imu.Bg.transpose() << std::endl;
                // m_Imu.Ba -= deltaCalib.template block<3,1>(5,0);//.template tail(CalibSize);
                // std::cout << "Ba delta is " << deltaCalib.template block<3,1>(5,0).transpose() << " ba is: " << m_Imu.Ba.transpose() << std::endl;
            // }
        }

        // std::cout << delta_l << std::endl;

        // update poses
        // std::cout << "Updating " << uNumPoses << " active poses." << std::endl;
        for (size_t ii = 0 ; ii < m_vPoses.size() ; ii++){
            // only update active poses, as inactive ones are not part of the optimization
            if( m_vPoses[ii].IsActive ){

                 m_vPoses[ii].Twp = exp_decoupled<Scalar>(m_vPoses[ii].Twp,-delta_p.template block<6,1>(m_vPoses[ii].OptId*PoseSize,0) * DAMPING);
                 // m_vPoses[ii].Twp = m_vPoses[ii].Twp * Sophus::SE3d::exp(delta_p.template block<6,1>(m_vPoses[ii].OptId*PoseSize,0));
                // update the velocities if they are parametrized
                if(PoseSize >= 9){
                    m_vPoses[ii].V -= delta_p.template block<3,1>(m_vPoses[ii].OptId*PoseSize+6,0) * DAMPING;
                    // std::cout << "Velocity for pose " << ii << " is " << m_vPoses[ii].V.transpose() << std::endl;
                }

                if(PoseSize >= 15){
                    m_vPoses[ii].B -= delta_p.template block<6,1>(m_vPoses[ii].OptId*PoseSize+9,0) * DAMPING;
                    // std::cout << "Velocity for pose " << ii << " is " << m_vPoses[ii].V.transpose() << std::endl;
                }

                if(PoseSize >= 21){
                    m_vPoses[ii].Tvs = exp_decoupled<Scalar>(m_vPoses[ii].Tvs,-delta_p.template block<6,1>(m_vPoses[ii].OptId*PoseSize+15,0));
                    // std::cout << "Velocity for pose " << ii << " is " << m_vPoses[ii].V.transpose() << std::endl;
                }else{
                    m_vPoses[ii].Tvs = m_Imu.Tvs;
                }

                std::cout << "Pose delta for " << ii << " is " << delta_p.template block<PoseSize,1>(m_vPoses[ii].OptId*PoseSize,0).transpose() << std::endl;
                std::cout << "Pose " << ii << " is " << Eigen::Map<const Eigen::Matrix<Scalar,4,1> >(m_vPoses[ii].Twp.data()).transpose() << std::endl;
                std::cout <<  " Tvs: " << std::endl << m_vPoses[ii].Tvs.matrix() <<   std::endl;
                // clear the vector of Tsw values as they will need to be recalculated
                m_vPoses[ii].Tsw.clear();
            }
             else{
              // std::cout << " Pose " << ii << " is inactive." << std::endl;
            }
        }

        const double dPrevError = m_dProjError + m_dImuError;
        std::cout << "Pre-solve norm: " << dPrevError << " with Epr:" << m_dProjError << " and Ei:" << m_dImuError << std::endl;
        _EvaluateResiduals();
        const double dPostError = m_dProjError + m_dImuError;
        std::cout << "Pose-solve norm: " << dPostError << " with Epr:" << m_dProjError << " and Ei:" << m_dImuError << std::endl;
        // std::cout << "BA iteration " << kk <<  " error: " << m_Rpr.norm() + m_Ru.norm() + m_Rpp.norm() + m_Ri.norm() << std::endl;
        std::cout << "Iteration " << kk << " took " << Toc(dItTime) << " seconds. " << std::endl;
    }


    if(PoseSize >= 15 && m_vPoses.size() > 0){
        m_Imu.Bg = m_vPoses.back().B.template head<3>();
        m_Imu.Ba = m_vPoses.back().B.template tail<3>();        
    }

    if(PoseSize >= 21 && m_vPoses.size() > 0){
        m_Imu.Tvs = m_vPoses.back().Tvs;
    }
        // std::cout << "Solve took " << Toc(dTime) << " seconds." << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////
template< typename Scalar,int LmSize, int PoseSize, int CalibSize >
void BundleAdjuster<Scalar, LmSize, PoseSize, CalibSize>::_BuildProblem()
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
    m_Rpr.resize(uNumProjRes*ProjectionResidual::ResSize);

    m_Jpp.resize(uNumBinRes, uNumPoses);
    m_Jppt.resize(uNumPoses, uNumBinRes);
    m_Rpp.resize(uNumBinRes*BinaryResidual::ResSize);

    m_Ju.resize(uNumUnRes, uNumPoses);
    m_Jut.resize(uNumPoses, uNumUnRes);
    m_Ru.resize(uNumUnRes*UnaryResidual::ResSize);

    m_Ji.resize(uNumImuRes, uNumPoses);
    m_Jit.resize(uNumPoses, uNumImuRes);
    m_Jki.resize(uNumImuRes, 1);
    m_Jkit.resize(1, uNumImuRes);
    m_Ri.resize(uNumImuRes*ImuResidual::ResSize);


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

     Eigen::Matrix<Scalar,2,LmSize> maxdZ_dX;
     Eigen::Matrix<Scalar,2,6> maxdZ_dPm;
     Eigen::Matrix<Scalar,2,6> maxdZ_dPr;
     Eigen::Matrix<Scalar,2,LmSize> maxdZ_dX_fd;
     Eigen::Matrix<Scalar,2,6> maxdZ_dPm_fd;
     Eigen::Matrix<Scalar,2,6> maxdZ_dPr_fd;
     double maxdZ_dX_norm = 1e-12, maxdZ_dPm_norm = 1e-12, maxdZ_dPr_norm = 1e-12;

     m_dProjError = 0;
    for( ProjectionResidual& res : m_vProjResiduals ){
        // calculate measurement jacobians

        // Tsw = T_cv * T_vw
        Landmark& lm = m_vLandmarks[res.LandmarkId];
        Pose& pose = m_vPoses[res.MeasPoseId];
        Pose& refPose = m_vPoses[res.RefPoseId];
        lm.Xs = MultHomogeneous(refPose.GetTsw(lm.RefCamId, m_Rig, PoseSize >= 21) ,lm.Xw);
        const SE3t Tvs_m = (PoseSize >= 21 ? pose.Tvs :  m_Rig.cameras[res.CameraId].T_wc);
        const SE3t Tvs_r = (PoseSize >= 21 ? refPose.Tvs :  m_Rig.cameras[res.CameraId].T_wc);
        const SE3t Tsw_m = pose.GetTsw(res.CameraId, m_Rig, PoseSize >= 21);
        const SE3t Tws_r = refPose.GetTsw(lm.RefCamId,m_Rig, PoseSize >= 21).inverse();

        const Vector2t p = m_Rig.cameras[res.CameraId].camera.Transfer3D(Tsw_m*Tws_r, lm.Xs.template head<3>(),lm.Xs(3));
        res.Residual = res.Z - p;
        // std::cout << "Residual for meas " << res.ResidualId << " and landmark " << res.LandmarkId << " with camera " << res.CameraId << " is " << res.Residual.transpose() << std::endl;

        // this array is used to calculate the robust norm
        m_vErrors.push_back(res.Residual.squaredNorm());

        const Eigen::Matrix<Scalar,2,4> dTdP_s = m_Rig.cameras[res.CameraId].camera.dTransfer3D_dP(Tsw_m*Tws_r,
                                                                                             lm.Xs.template head<3>(),lm.Xs(3));
        // Landmark Jacobian
        if(lm.IsActive){
            res.dZ_dX = -dTdP_s.template block<2,LmSize>( 0, LmSize == 3 ? 0 : 3 );

            Eigen::Matrix<Scalar,2,4> dZ_dX_fd_4d;
            Scalar dEps = 1e-9;
            for(int ii = 0; ii < 6 ; ii++) {
                Eigen::Matrix<Scalar,4,1> delta;
                delta.setZero();
                delta[ii] = dEps;
                Vector4t xPlus = lm.Xs + delta;
                const Vector2t pPlus  = m_Rig.cameras[res.CameraId].camera.Transfer3D(Tsw_m*Tws_r,
                                                                                            xPlus.template head<3>(),xPlus(3) + dEps);
                delta[ii] = -dEps;
                Vector4t xMinus = lm.Xs + delta;
                const Vector2t pMinus  = m_Rig.cameras[res.CameraId].camera.Transfer3D(Tsw_m*Tws_r,
                                                                                            xMinus.template head<3>(),xMinus(3) + dEps);
                dZ_dX_fd_4d.col(ii) = -(pPlus-pMinus)/(2*dEps);
            }

            Eigen::MatrixXd dZ_dX_fd = LmSize == 3 ? dZ_dX_fd_4d.template block(0,0,2,3) : dZ_dX_fd_4d.template block(0,3,2,1);

            if( (res.dZ_dX + dZ_dX_fd).norm() > maxdZ_dX_norm ){
                maxdZ_dX_norm = (res.dZ_dX - dZ_dX_fd).norm();
                maxdZ_dX = res.dZ_dX;
                maxdZ_dX_fd = dZ_dX_fd;
            }
//                    std::cout << "dZ_dX   :" << res.dZ_dX << std::endl;
//                    std::cout << "dZ_dX_fd:" << dZ_dX_fd << " norm: " << (res.dZ_dX - dZ_dX_fd).norm() <<  std::endl;
        }

        if( pose.IsActive || refPose.IsActive ) {
            // std::cout << "Calculating j for residual with poseid " << pose.Id << " and refPoseId " << refPose.Id << std::endl;
            // derivative for the measurement pose
            Eigen::Matrix<Scalar,4,1> Xs_m = MultHomogeneous(Tsw_m, lm.Xw);
            const Eigen::Matrix<Scalar,2,4> dTdP_m = m_Rig.cameras[res.CameraId].camera.dTransfer3D_dP(SE3t(),
                                                                                                     Xs_m.template head<3>(),Xs_m(3));            

            const Eigen::Matrix<Scalar,2,4> dTdP_m_Tsv_m = dTdP_m * Tvs_m.inverse().matrix();
            for(unsigned int ii=0; ii<3; ++ii){
                res.dZ_dPm.template block<2,1>(0,ii) = -dTdP_m_Tsv_m * pose.Twp.inverse().matrix() * -Sophus::SE3Group<Scalar>::generator(ii) * lm.Xw;    // translation
            }
            for(unsigned int ii=3; ii<6; ++ii){
                res.dZ_dPm.template block<2,1>(0,ii) = -dTdP_m_Tsv_m * -Sophus::SE3Group<Scalar>::generator(ii) * pose.Twp.inverse().matrix() * lm.Xw; // rotation
            }

            Eigen::Matrix<Scalar,2,6> dZ_dPm_fd;
            Scalar dEps = 1e-9;
            for(int ii = 0; ii < 6 ; ii++) {
                Eigen::Matrix<Scalar,6,1> delta;
                delta.setZero();
                delta[ii] = dEps;
                SE3t Tss = (exp_decoupled(pose.Twp,delta)*Tvs_m).inverse() * refPose.Twp * Tvs_r;
                const Vector2t pPlus = m_Rig.cameras[res.CameraId].camera.Transfer3D(Tss,lm.Xs.template head(3),lm.Xs[3]);
                delta[ii] = -dEps;
                // Tsw = (pose.Twp*SE3t::exp(delta)*m_Rig.cameras[meas.CameraId].T_wc).inverse();
                Tss = (exp_decoupled(pose.Twp,delta)*Tvs_m).inverse() * refPose.Twp * Tvs_r;
                const Vector2t pMinus = m_Rig.cameras[res.CameraId].camera.Transfer3D(Tss,lm.Xs.template head(3),lm.Xs[3]);
                dZ_dPm_fd.col(ii) = -(pPlus-pMinus)/(2*dEps);
            }

            if( (res.dZ_dPm - dZ_dPm_fd).norm() > maxdZ_dPm_norm ){
                maxdZ_dPm_norm = (res.dZ_dPm - dZ_dPm_fd).norm();
                maxdZ_dPm = res.dZ_dPm;
                maxdZ_dPm_fd = dZ_dPm_fd;
            }
//            std::cout << "dZ_dPm   :" << res.dZ_dPm << std::endl;
//            std::cout << "dZ_dPm_fd:" << dZ_dPm_fd << " norm: " << (res.dZ_dPm - dZ_dPm_fd).norm() <<  std::endl;

            // only need this if we are in inverse depth mode
            if( LmSize == 1 ){
                // derivative for the reference pose
                const Eigen::Matrix<Scalar,2,4> dTdP_m_Tsw_m = dTdP_m * (pose.Twp * Tvs_m).inverse().matrix();
                for(unsigned int ii=0; ii<3; ++ii){
                    res.dZ_dPr.template block<2,1>(0,ii) = -dTdP_m_Tsw_m * Sophus::SE3Group<Scalar>::generator(ii) * Tvs_r.matrix() * lm.Xs;
                }
                for(unsigned int ii=3; ii<6; ++ii){
                    res.dZ_dPr.template block<2,1>(0,ii) = -dTdP_m_Tsw_m * refPose.Twp.matrix() * Sophus::SE3Group<Scalar>::generator(ii) * Tvs_r.matrix() * lm.Xs;
                }

                Eigen::Matrix<Scalar,2,6> dZ_dPr_fd;
                for(int ii = 0; ii < 6 ; ii++) {
                    Eigen::Matrix<Scalar,6,1> delta;
                    delta.setZero();
                    delta[ii] = dEps;
                    SE3t Tss = (pose.Twp*Tvs_m).inverse() * (exp_decoupled(refPose.Twp,delta)) * Tvs_r;
                    const Vector2t pPlus = m_Rig.cameras[res.CameraId].camera.Transfer3D(Tss,lm.Xs.template head(3),lm.Xs[3]);
                    delta[ii] = -dEps;
                    // Tsw = (pose.Twp*SE3t::exp(delta)*m_Rig.cameras[meas.CameraId].T_wc).inverse();
                    Tss = (pose.Twp*Tvs_m).inverse() * (exp_decoupled(refPose.Twp,delta)) * Tvs_r;
                    const Vector2t pMinus = m_Rig.cameras[res.CameraId].camera.Transfer3D(Tss,lm.Xs.template head(3),lm.Xs[3]);
                    dZ_dPr_fd.col(ii) = -(pPlus-pMinus)/(2*dEps);
                }

                if( (res.dZ_dPr - dZ_dPr_fd).norm() > maxdZ_dPr_norm ){
                    maxdZ_dPr_norm = (res.dZ_dPr - dZ_dPr_fd).norm();
                    maxdZ_dPr = dZ_dPr_fd;
                    maxdZ_dPr_fd = res.dZ_dPr;
                }

//                std::cout << "dZ_dPr   :" << res.dZ_dPr << std::endl;
//                std::cout << "dZ_dPr_fd:" << dZ_dPr_fd << " norm: " << (res.dZ_dPr - dZ_dPr_fd).norm() <<  std::endl;
            }

            // derivative for the measurement Tsv
            if(PoseSize > 21 && m_vImuResiduals.size() != 0){
                Eigen::Matrix<Scalar,4,1> Xv_m = MultHomogeneous(pose.Twp.inverse(), lm.Xw);
                const Eigen::Matrix<Scalar,2,4> dTvdP_m = m_Rig.cameras[res.CameraId].camera.dTransfer3D_dP(pose.Tvs.inverse(),
                                                                                                         Xv_m.template head<3>(),Xv_m(3));

                res.dZ_dTvs_m.template block<2,3>(0,0) = -dTvdP_m.template block<2,3>(0,0);
                for(unsigned int ii=3; ii<6; ++ii){
                    res.dZ_dTvs_m.template block<2,1>(0,ii) = -dTvdP_m * Sophus::SE3Group<Scalar>::generator(ii) * Xv_m;
                }

                /*
                Eigen::Matrix<Scalar,2,6> J_fd;
                Scalar dEps = 1e-6;
                for(int ii = 0; ii < 6 ; ii++) {
                    Eigen::Matrix<Scalar,6,1> delta;
                    delta.setZero();
                    delta[ii] = dEps;
                    SE3t Tsw = (pose.Twp*exp_decoupled(pose.Tvs,delta)).inverse();
                    const Vector2t pPlus = m_Rig.cameras[res.CameraId].camera.Transfer3D(Tsw,lm.Xw.template head(3),lm.Xw[3]);
                    delta[ii] = -dEps;
                    // Tsw = (pose.Twp*SE3t::exp(delta)*m_Rig.cameras[meas.CameraId].T_wc).inverse();
                    Tsw = (pose.Twp*exp_decoupled(pose.Tvs,delta)).inverse();
                    const Vector2t pMinus = m_Rig.cameras[res.CameraId].camera.Transfer3D(Tsw,lm.Xw.template head(3),lm.Xw[3]);
                    J_fd.col(ii) = (pPlus-pMinus)/(2*dEps);
                }
                std::cout << "Jproj_tvs1:" << res.dZ_dTvs_m << std::endl;
                std::cout << "Jproj_fd_tvs1:" << J_fd << std::endl;
                */

                res.dZ_dTvs_r.template block<2,3>(0,0) = dTdP_s.template block<2,3>(0,0);
                for(unsigned int ii=3; ii<6; ++ii){
                    res.dZ_dTvs_r.template block<2,1>(0,ii) = dTdP_s * -Sophus::SE3Group<Scalar>::generator(ii) * lm.Xs;
                }

                /*
                Eigen::Matrix<Scalar,2,6> J_fd2;
                // Scalar dEps = 1e-6;
                for(int ii = 0; ii < 6 ; ii++) {
                    Eigen::Matrix<Scalar,6,1> delta;
                    delta.setZero();
                    delta[ii] = dEps;
                    SE3t Tss = (pose.Twp*pose.Tvs).inverse() * refPose.Twp*exp_decoupled(refPose.Tvs,delta);
                    const Vector2t pPlus = m_Rig.cameras[res.CameraId].camera.Transfer3D(Tss,lm.Xs.template head(3),lm.Xs[3]);
                    delta[ii] = -dEps;
                    // Tsw = (pose.Twp*SE3t::exp(delta)*m_Rig.cameras[meas.CameraId].T_wc).inverse();
                    Tss = (pose.Twp*pose.Tvs).inverse() * refPose.Twp*exp_decoupled(refPose.Tvs,delta);
                    const Vector2t pMinus = m_Rig.cameras[res.CameraId].camera.Transfer3D(Tss,lm.Xs.template head(3),lm.Xs[3]);
                    J_fd2.col(ii) = (pPlus-pMinus)/(2*dEps);
                }
                std::cout << "Jproj_tvs2:" << res.dZ_dTvs_r << std::endl;
                std::cout << "Jproj_fd_tvs2:" << J_fd2 << std::endl;
                */

                res.dZ_dTvs_r.template block<2,3>(0,0).setZero();
                res.dZ_dTvs_m.template block<2,3>(0,0).setZero();

            }else if( CalibSize > 2 && m_vImuResiduals.size() != 0 ){
                Eigen::Matrix<Scalar,4,1> Xs_m = MultHomogeneous(pose.Twp.inverse(), lm.Xw);
                const Eigen::Matrix<Scalar,2,4> dTvdPs_m = m_Rig.cameras[res.CameraId].camera.dTransfer3D_dP(SE3t(),
                                                                                                         Xs_m.template head<3>(),Xs_m(3));
                Eigen::Matrix<Scalar,4,4> Tws = m_Rig.cameras[res.CameraId].T_wc.matrix();
                Eigen::Matrix<Scalar,4,4> Tsw = m_Rig.cameras[res.CameraId].T_wc.inverse().matrix();
                for(unsigned int ii=0; ii<6; ++ii){
                    res.dZ_dTvs_m.template block<2,1>(0,ii) = dTvdPs_m * ( (-SE3t::generator(ii)*Tsw * pose.Twp.inverse().matrix() * refPose.Twp.matrix() * Tws * lm.Xs ) +
                            (Tsw * pose.Twp.inverse().matrix() * refPose.Twp.matrix() * Tws * SE3t::generator(ii) * lm.Xs));
                }
                res.dZ_dTvs_m.template block<2,3>(0,0).setZero();

//                Eigen::Matrix<Scalar,2,6> J_fd_tvs;
//                Scalar dEps = 1e-6;
//                for(int ii = 0; ii < 6 ; ii++) {
//                    Eigen::Matrix<Scalar,6,1> delta;
//                    delta.setZero();
//                    delta[ii] = dEps;
//                    SE3t Tvs = m_Rig.cameras[res.CameraId].T_wc * SE3t::exp(delta);
//                    const Vector2t pPlus = m_Rig.cameras[res.CameraId].camera.Transfer3D((pose.Twp * Tvs).inverse() *
//                            refPose.Twp * Tvs,lm.Xs.template head(3),lm.Xs[3]);
//                    delta[ii] = -dEps;
//                    // Tsw = (pose.Twp*SE3t::exp(delta)*m_Rig.cameras[meas.CameraId].T_wc).inverse();
//                    Tvs = m_Rig.cameras[res.CameraId].T_wc * SE3t::exp(delta);
//                    const Vector2t pMinus = m_Rig.cameras[res.CameraId].camera.Transfer3D((pose.Twp * Tvs).inverse() *
//                            refPose.Twp * Tvs,lm.Xs.template head(3),lm.Xs[3]);
//                    J_fd_tvs.col(ii) = (pPlus-pMinus)/(2*dEps);
//                }
//                std::cout << "Jproj_tvs:" << res.dZ_dTvs_m << std::endl;
//                std::cout << "Jproj_fd_tvs:" << J_fd_tvs << std::endl;
            }else{
                res.dZ_dTvs_m.setZero();
                res.dZ_dTvs_r.setZero();
            }


            /*
            Eigen::Matrix<Scalar,2,6> J_fd;
            Scalar dEps = 1e-6;
            for(int ii = 0; ii < 6 ; ii++) {
                Eigen::Matrix<Scalar,6,1> delta;
                delta.setZero();
                delta[ii] = dEps;
                // SE3t Tsw = (pose.Twp*SE3t::exp(delta)*m_Rig.cameras[meas.CameraId].T_wc).inverse();
                SE3t Tsw = (exp_decoupled(pose.Twp,delta)*m_Rig.cameras[res.CameraId].T_wc).inverse();
                const Vector2t pPlus = m_Rig.cameras[res.CameraId].camera.Transfer3D(Tsw,lm.Xw.template head(3),lm.Xw[3]);
                delta[ii] = -dEps;
                // Tsw = (pose.Twp*SE3t::exp(delta)*m_Rig.cameras[meas.CameraId].T_wc).inverse();
                Tsw = (exp_decoupled(pose.Twp,delta)*m_Rig.cameras[res.CameraId].T_wc).inverse();
                const Vector2t pMinus = m_Rig.cameras[res.CameraId].camera.Transfer3D(Tsw,lm.Xw.template head(3),lm.Xw[3]);
                J_fd.col(ii) = (pPlus-pMinus)/(2*dEps);
            }
            std::cout << "Jproj:" << res.dZ_dP << std::endl;
            std::cout << "Jproj_fd:" << J_fd << std::endl;
            */
        }
        // set the residual in m_R which is dense
        m_Rpr.template segment<ProjectionResidual::ResSize>(res.ResidualOffset) = res.Residual;
    }

    std::cout << "Max dZ_dX norm: " << maxdZ_dX_norm << " with dZ_dX: " << std::endl << maxdZ_dX << " with dZ_dX_fd: "  << std::endl << maxdZ_dX_fd << std::endl;
    std::cout << "Max dZ_dPm norm: " << maxdZ_dPm_norm << " with error: " << std::endl << maxdZ_dPm << " with dZ_dPm_fd: " << std::endl  << maxdZ_dPm_fd <<  std::endl;
    std::cout << "Max dZ_dPr norm: " << maxdZ_dPr_norm << " with error: " << std::endl << maxdZ_dPr <<  " with dZ_dPr_fd: " << std::endl  << maxdZ_dPr_fd  << std::endl;

    // get the sigma for robust norm calculation. This call is O(n) on average,
    // which is desirable over O(nlogn) sort
    if( m_vErrors.size() > 0 ){
        auto it = m_vErrors.begin()+std::floor(m_vErrors.size()/2);
        std::nth_element(m_vErrors.begin(),it,m_vErrors.end());
        const Scalar dSigma = sqrt(*it);
        std::cout << "Projection error sigma is " << dSigma << std::endl;
        // See "Parameter Estimation Techniques: A Tutorial with Application to Conic
        // Fitting" by Zhengyou Zhang. PP 26 defines this magic number:
        const Scalar c_huber = 1.2107*dSigma;

        // now go through the measurements and assign weights
        for( ProjectionResidual& res : m_vProjResiduals ){
            // calculate the huber norm weight for this measurement
            const Scalar e = res.Residual.norm();
            res.W = e > c_huber ? c_huber/e : 1.0;
            m_dProjError += res.Residual.norm() * res.W;
//            if( GetPose(res.RefPoseId).IsActive == false && GetPose(res.MeasPoseId).IsActive == false ){
//                std::cout << "Outside landmark residual is " << res.Residual.norm() << std::endl;
//            }
        }
    }
    m_vErrors.clear();

    // build binary residual jacobians
    for( BinaryResidual& res : m_vBinaryResiduals ){
        const SE3t& Twa = m_vPoses[res.PoseAId].Twp;
        const SE3t& Twb = m_vPoses[res.PoseBId].Twp;
        res.dZ_dX1 = dLog_dX(Twa, res.Tab * Twb.inverse());
        // the negative sign here is because exp(x) is inside the inverse when we invert (Twb*exp(x)).inverse
        res.dZ_dX2 = -dLog_dX(Twa * res.Tab, Twb.inverse());

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
            m_Rpp.template segment<BinaryResidual::ResSize>(res.ResidualOffset) = log_decoupled(Twa*res.Tab, Twb);
    }

    for( UnaryResidual& res : m_vUnaryResiduals ){
        const SE3t& Twp = m_vPoses[res.PoseId].Twp;
        // res.dZ_dX = dLog_dX(Twp, res.Twp.inverse());
        res.dZ_dX = dLog_decoupled_dX(Twp, res.Twp);

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

        m_Ru.template segment<UnaryResidual::ResSize>(res.ResidualOffset) = log_decoupled(Twp, res.Twp);
        // m_Ru.template segment<UnaryResidual::ResSize>(res.ResidualOffset) = (Twp*res.Twp.inverse()).log();
    }

    m_dImuError = 0;
    for( ImuResidual& res : m_vImuResiduals ){
        // set up the initial pose for the integration
        const Vector3t gravity = GetGravityVector(m_Imu.G);

        const Pose& poseA = m_vPoses[res.PoseAId];
        const Pose& poseB = m_vPoses[res.PoseBId];

        Eigen::Matrix<Scalar,10,6> jb_q;
        // Eigen::Matrix<Scalar,10,10> jb_y;
        ImuPose imuPose = ImuResidual::IntegrateResidual(poseA,res.Measurements,poseA.B.template head<3>(),
                                                         poseA.B.template tail<3>(),gravity,res.Poses,&jb_q/*,&jb_y*/);
        Scalar totalDt = res.Measurements.back().Time - res.Measurements.front().Time;
        const SE3t Tab = poseA.Twp.inverse()*imuPose.Twp;
        const SE3t& Twa = poseA.Twp;
        const SE3t& Twb = poseB.Twp;

        // now given the poses, calculate the jacobians.
        // First subtract gravity, initial pose and velocity from the delta T and delta V
        SE3t Tab_0 = imuPose.Twp;
        Tab_0.translation() -=(-gravity*0.5*powi(totalDt,2) + poseA.V*totalDt);   // subtract starting velocity and gravity
        Tab_0 = poseA.Twp.inverse() * Tab_0;                                       // subtract starting pose
        // Augment the velocity delta by subtracting effects of gravity
        Vector3t Vab_0 = imuPose.V - poseA.V;
        Vab_0 += gravity*totalDt;
        // rotate the velocity delta so that it starts from orientation=Ident
        Vab_0 = poseA.Twp.so3().inverse() * Vab_0;

        // derivative with respect to the start pose
        res.dZ_dX1.setZero();
        res.dZ_dX2.setZero();
        res.dZ_dG.setZero();
        res.dZ_dB.setZero();

        // SE3t Tstar(Sophus::SO3Group<Scalar>(),(poseA.V*totalDt - 0.5*gravity*powi(totalDt,2)));

        // calculate the derivative of the lie log with respect to the tangent plane at Twa        
        Eigen::Matrix<Scalar,6,6> dLog;
        dLog.setZero();
        dLog.template block<3,3>(0,0) = Eigen::Matrix3d::Identity();
        dLog.template block<3,3>(3,3) = dLog_dq((Twa.so3()*Tab_0.so3()*Twb.so3().inverse()).unit_quaternion()) *
                dq1q2_dq1((Tab_0.so3() * Twb.so3().inverse()).unit_quaternion()) *
                dq1q2_dq2(Twa.unit_quaternion()) *
                dqExp_dw<Scalar>(Eigen::Matrix<Scalar,3,1>::Zero());

        dLog.template block<3,3>(0,3) = dqx_dq<Scalar>((Twa).unit_quaternion(),Tab_0.translation())*
                dq1q2_dq2(Twa.unit_quaternion()) *
                dqExp_dw<Scalar>(Eigen::Matrix<Scalar,3,1>::Zero());

        // now add the log jacobians to the bias jacobian terms
        res.dZ_dB.template block<3,6>(3,0) = dLog_dq((imuPose.Twp.so3() * Twb.so3().inverse()).unit_quaternion()) *
                                        dq1q2_dq1(Twb.so3().inverse().unit_quaternion()) * jb_q.template block<4,6>(3,0);
        res.dZ_dB.template block<3,6>(0,0) = jb_q.template block<3,6>(0,0);
        res.dZ_dB.template block<3,6>(6,0) = jb_q.template block<3,6>(7,0);
        res.dZ_dB.template block<6,6>(9,0) = Eigen::Matrix<Scalar,6,6>::Identity();   // dB/dB        


        res.dZ_dX1.template block<6,6>(0,0) = dLog;
        // Twa^-1 is multiplied here as we need the velocity derivative in the frame of pose A, as the log is taken from this frame
        res.dZ_dX1.template block<3,3>(0,6) = Matrix3t::Identity()*totalDt;
        for( int ii = 0; ii < 3 ; ++ii ){
            res.dZ_dX1.template block<3,1>(6,3+ii) = Twa.so3().matrix() * Sophus::SO3Group<Scalar>::generator(ii) * Vab_0;
        }
        res.dZ_dX1.template block<3,3>(6,6) = Matrix3t::Identity();

        res.dZ_dX1.template block<3,6>(0,0) = jb_q.template block<3,6>(0,0);
        res.dZ_dX1.template block<3,6>(6,0) = jb_q.template block<3,6>(7,0);
        res.dZ_dX1.template block<ImuResidual::ResSize,6>(0,9) = res.dZ_dB;
        // res.dZ_dX1.template block<ImuResidual::ResSize,6>(0,9).setZero();
        // std::cout << "dZ_dX1" << res.dZ_dX1 << std::endl;

        // the - sign is here because of the exp(-x) within the log
        res.dZ_dX2.template block<6,6>(0,0) = -dLog_decoupled_dX(imuPose.Twp,Twb);//-dLog_dX(Twa*Tab,Twb.inverse());
        res.dZ_dX2.template block<3,3>(6,6) = -Matrix3t::Identity();
        res.dZ_dX2.template block<6,6>(9,9) = -Eigen::Matrix<Scalar,6,6>::Identity();
        // res.dZ_dX2.template block<6,6>(9,9).setZero();

        const Eigen::Matrix<Scalar,3,2> dGravity = dGravity_dDirection(m_Imu.G);
        res.dZ_dG.template block<3,2>(0,0) = /*dLog.template block<3,3>(0,0) * Twa.so3().inverse().matrix() **/
                                    -0.5*powi(totalDt,2)*Matrix3t::Identity()*dGravity;
        res.dZ_dG.template block<3,2>(6,0) = -totalDt*Matrix3t::Identity()*dGravity;

        res.Residual.template head<6>() = log_decoupled(Twa*Tab,Twb);
        res.Residual.template segment<3>(6) = imuPose.V - poseB.V;
        res.Residual.template segment<6>(9) = poseA.B - poseB.B;

        if(PoseSize > 15){
            res.Residual.template segment<6>(15) = log_decoupled(poseA.Tvs, poseB.Tvs);
            //std::cout << "Adding tvs residual of " << res.Residual.template segment<6>(15).transpose() << std::endl;
            const Eigen::Matrix<Scalar,6,6> dLog = dLog_decoupled_dX(poseA.Tvs, poseB.Tvs);
            res.dZ_dX1.template block<6,6>(15,15) = dLog;
            res.dZ_dX2.template block<6,6>(15,15) = -dLog;

            // disable imu translation error
            res.Residual.template head<3>().setZero();     // translation error
            res.Residual.template segment<3>(6).setZero(); // velocity error
            res.Residual.template segment<6>(9).setZero(); // bias
            res.dZ_dX1.template block<3, PoseSize>(0,0).setZero();
            res.dZ_dX2.template block<3, PoseSize>(0,0).setZero();
            res.dZ_dX1.template block<3, PoseSize>(6,0).setZero();
            res.dZ_dX2.template block<3, PoseSize>(6,0).setZero();
            res.dZ_dB.template block<3, 6>(0,0).setZero();
            res.dZ_dG.template block<3, 2>(0,0).setZero();

            // disable accelerometer bias
            res.dZ_dX1.template block<res.ResSize, 3>(0,12).setZero();
            res.dZ_dX2.template block<res.ResSize, 3>(0,12).setZero();
        }

        if(CalibSize > 2){
            // disable imu translation error
            res.Residual.template head<3>().setZero();
            res.Residual.template segment<3>(6).setZero(); // velocity error
            res.Residual.template segment<6>(9).setZero(); // bias
            res.dZ_dX1.template block<3, PoseSize>(0,0).setZero();
            res.dZ_dX2.template block<3, PoseSize>(0,0).setZero();
            res.dZ_dX1.template block<3, PoseSize>(6,0).setZero();
            res.dZ_dX2.template block<3, PoseSize>(6,0).setZero();
            res.dZ_dB.template block<3, 6>(0,0).setZero();
            res.dZ_dG.template block<3, 2>(0,0).setZero();

            // disable accelerometer and gyro bias
            res.dZ_dX1.template block<res.ResSize, 6>(0,9).setZero();
            res.dZ_dX2.template block<res.ResSize, 6>(0,9).setZero();
        }
        //if(poseA.IsActive == false || poseB.IsActive == false){
        //    std::cout << "PRIOR RESIDUAL: ";
        //}
        //std::cout << "Residual for res " << res.ResidualId << " : " << res.Residual.transpose() << std::endl;
        m_vErrors.push_back(res.Residual.squaredNorm());

//        res.SigmanInv = (res.dZ_dB * m_Imu.R * res.dZ_dB.transpose()).inverse();
//        std::cout << "Sigma inv for res " << res.ResidualId << " is " << res.SigmanInv << std::endl;

        // res.dZ_dB.setZero();

        /*

        Scalar dEps = 1e-12;
        Eigen::Matrix<Scalar,6,6> Jlog;
        for(int ii = 0 ; ii < 6 ; ii++){
            Vector6t eps = Vector6t::Zero();
            eps[ii] += dEps;
            Vector6t resPlus = log_decoupled(exp_decoupled(Twa,eps),Twb*Tab.inverse());
            eps[ii] -= 2*dEps;
            Vector6t resMinus = log_decoupled(exp_decoupled(Twa,eps),Twb*Tab.inverse());
            Jlog.col(ii) = (resPlus-resMinus)/(2*dEps);
        }

        std::cout << "Jlog = [" << dLog_decoupled_dX(Twa,Twb*Tab.inverse()).format(cleanFmt) << "]" << std::endl;
        std::cout << "Jlogf = [" << Jlog.format(cleanFmt) << "]" << std::endl;
        std::cout << "Jlog - Jlogf = [" << (dLog_decoupled_dX(Twa,Twb*Tab.inverse())- Jlog).format(cleanFmt) << "]" << std::endl;

        Eigen::Matrix<Scalar,6,6> dlog_dTwbf;
        for(int ii = 0 ; ii < 6 ; ii++){
            Vector6t eps = Vector6t::Zero();
            eps[ii] += dEps;
            const Vector6t resPlus = log_decoupled(Twa*Tab,(exp_decoupled(Twb,eps)));
            eps[ii] -= 2*dEps;
            const Vector6t resMinus = log_decoupled(Twa*Tab,(exp_decoupled(Twb,eps)));
            dlog_dTwbf.col(ii) = (resPlus-resMinus)/(2*dEps);
        }

        std::cout << "dlog_dTwb = [" << (-dLog_decoupled_dX(Twa*Tab,Twb)).format(cleanFmt) << "]" << std::endl;
        std::cout << "dlog_dTwbf = [" << dlog_dTwbf.format(cleanFmt) << "]" << std::endl;
        std::cout << "dlog_dTwb - dlog_dTwbf = [" << (-dLog_decoupled_dX(Twa*Tab,Twb)- dlog_dTwbf).format(cleanFmt) << "]" << std::endl;

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
            y0_eps.Twp = exp_decoupled<Scalar>(y0_eps.Twp,epsVec.template head<6>());
            y0_eps.V += epsVec.template tail<3>();
            Eigen::Matrix<Scalar,9,1> r_plus;
            r_plus.template head<6>() = log_decoupled(imuPose.Twp,y0_eps.Twp);
            r_plus.template tail<3>() = imuPose.V - y0_eps.V;



            epsVec[ii] -= 2*dEps;
            y0_eps = ImuPose(poseB.Twp,poseB.V, Vector3t::Zero(),0);;
            y0_eps.Twp = exp_decoupled<Scalar>(y0_eps.Twp,epsVec.template head<6>());
            y0_eps.V += epsVec.template tail<3>();
            Eigen::Matrix<Scalar,9,1> r_minus;
            r_minus.template head<6>() = log_decoupled(imuPose.Twp,y0_eps.Twp);
            r_minus.template tail<3>() = imuPose.V - y0_eps.V;

            dRi_dx2_fd.col(ii) = (r_plus-r_minus)/(2*dEps);
        }
        std::cout << "res.dZ_dX2= " << std::endl << res.dZ_dX2.format(cleanFmt) << std::endl;
        std::cout << "dRi_dx2_fd = " << std::endl <<  dRi_dx2_fd.format(cleanFmt) << std::endl;
        std::cout << "res.dZ_dX2-dRi_dx2_fd = " << std::endl << (res.dZ_dX2-dRi_dx2_fd).format(cleanFmt) << "norm: " << (res.dZ_dX2-dRi_dx2_fd).norm() <<  std::endl;


        for(int ii = 0 ; ii < 6 ; ii++){
            Vector6t eps = Vector6t::Zero();
            eps[ii] += dEps;
            Pose poseEps = poseA;
            poseEps.Twp = exp_decoupled(poseEps.Twp,eps);
            // poseEps.Twp = poseEps.Twp * SE3t::exp(eps);
            std::vector<ImuPose> poses;
            const ImuPose imuPosePlus = ImuResidual::IntegrateResidual(poseEps,res.Measurements,m_Imu.Bg,m_Imu.Ba,gravity,poses);
            // const Vector6t dErrorPlus = log_decoupled(imuPosePlus.Twp, Twb);
            const Vector6t dErrorPlus = log_decoupled(imuPosePlus.Twp, Twb);
            const Vector3t vErrorPlus = imuPosePlus.V - poseB.V;
            eps[ii] -= 2*dEps;
            poseEps = poseA;
            poseEps.Twp = exp_decoupled(poseEps.Twp,eps);
            // poseEps.Twp = poseEps.Twp * SE3t::exp(eps);
            poses.clear();
            const ImuPose imuPoseMinus = ImuResidual::IntegrateResidual(poseEps,res.Measurements,m_Imu.Bg,m_Imu.Ba,gravity,poses);
            // const Vector6t dErrorMinus = log_decoupled(imuPoseMinus.Twp, Twb);
            const Vector6t dErrorMinus = log_decoupled(imuPoseMinus.Twp, Twb);
            const Vector3t vErrorMinus = imuPoseMinus.V - poseB.V;
            J_fd.col(ii).template head<6>() = (dErrorPlus - dErrorMinus)/(2*dEps);
            J_fd.col(ii).template tail<3>() = (vErrorPlus - vErrorMinus)/(2*dEps);
        }

        for(int ii = 0 ; ii < 3 ; ii++){
            Vector3t eps = Vector3t::Zero();
            eps[ii] += dEps;
            Pose poseEps = poseA;
            poseEps.V += eps;
            std::vector<ImuPose> poses;
            const ImuPose imuPosePlus = ImuResidual::IntegrateResidual(poseEps,res.Measurements,m_Imu.Bg,m_Imu.Ba,gravity,poses);
            const Vector6t dErrorPlus = log_decoupled(imuPosePlus.Twp, Twb);
//                std::cout << "Pose plus: " << imuPosePlus.Twp.matrix() << std::endl;
            const Vector3t vErrorPlus = imuPosePlus.V - poseB.V;
            eps[ii] -= 2*dEps;
            poseEps = poseA;
            poseEps.V += eps;
            poses.clear();
            const ImuPose imuPoseMinus = ImuResidual::IntegrateResidual(poseEps,res.Measurements,m_Imu.Bg,m_Imu.Ba,gravity,poses);
            const Vector6t dErrorMinus = log_decoupled(imuPoseMinus.Twp, Twb);
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
            const Vector6t dErrorPlus = log_decoupled(imuPosePlus.Twp, Twb);
//                std::cout << "Pose plus: " << imuPosePlus.Twp.matrix() << std::endl;
            const Vector3t vErrorPlus = imuPosePlus.V - poseB.V;
            eps[ii] -= 2*dEps;
            poses.clear();
            const Vector2t gMinus = m_Imu.G+eps;
            const ImuPose imuPoseMinus = ImuResidual::IntegrateResidual(poseA,res.Measurements,m_Imu.Bg,m_Imu.Ba,GetGravityVector(gMinus),poses);
            const Vector6t dErrorMinus = log_decoupled(imuPoseMinus.Twp, Twb);
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
            const Vector6t dErrorPlus = log_decoupled(imuPosePlus.Twp, Twb);
            const Vector3t vErrorPlus = imuPosePlus.V - poseB.V;

            eps[ii] -= 2*dEps;
            const Vector6t minusBiases = biasVec + eps;
            poses.clear();
            const ImuPose imuPoseMinus = ImuResidual::IntegrateResidual(poseA,res.Measurements,minusBiases.template head<3>(),minusBiases.template tail<3>(),gravity,poses);
            const Vector6t dErrorMinus = log_decoupled(imuPoseMinus.Twp, Twb);
            const Vector3t vErrorMinus = imuPoseMinus.V - poseB.V;
            Jb_fd.col(ii).template head<6>() = (dErrorPlus - dErrorMinus)/(2*dEps);
            Jb_fd.col(ii).template tail<3>() = (vErrorPlus - vErrorMinus)/(2*dEps);
        }


        std::cout << "J = [" << std::endl << res.dZ_dX1.format(cleanFmt) << "]" << std::endl;
        std::cout << "Jf = [" << std::endl << J_fd.format(cleanFmt) << "]" << std::endl;
        std::cout << "J-Jf = [" << std::endl << (res.dZ_dX1-J_fd).format(cleanFmt) << "] norm = " << (res.dZ_dX1-J_fd).norm() << std::endl;

        std::cout << "Jg = [" << std::endl << res.dZ_dG.format(cleanFmt) << "]" << std::endl;
        std::cout << "Jgf = [" << std::endl << Jg_fd.format(cleanFmt) << "]" << std::endl;
        std::cout << "Jg-Jgf = [" << std::endl << (res.dZ_dG-Jg_fd).format(cleanFmt) << "] norm = " << (res.dZ_dG-Jg_fd).norm() << std::endl;

        std::cout << "Jb = [" << std::endl << res.dZ_dB.format(cleanFmt) << "]" << std::endl;
        std::cout << "Jbf = [" << std::endl << Jb_fd.format(cleanFmt) << "]" << std::endl;
        std::cout << "Jb-Jbf = [" << std::endl << (res.dZ_dB-Jb_fd).format(cleanFmt) << "] norm = " << (res.dZ_dB-Jb_fd).norm() << std::endl;
        */

        // now that we have the deltas with subtracted initial velocity, transform and gravity, we can construct the jacobian
        m_Ri.template segment<ImuResidual::ResSize>(res.ResidualOffset) = res.Residual;
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
            const Scalar e = res.Residual.norm();
            //res.W *= e > c_huber ? c_huber/e : 1.0;
            m_dImuError += res.Residual.norm() * res.W;
        }
    }
    m_vErrors.clear();

    std::cout << "Jacobian evaluation took " << Toc(dTime) << " seconds. " << std::endl;



    //TODO : The transpose insertions here are hideously expensive as they are not in order.
    // find a way around this.

    // here we sort the measurements and insert them per pose and per landmark, this will mean
    // each insert operation is O(1)
    dTime = Tic();
    for( Pose& pose : m_vPoses ){
        if( pose.IsActive ) {
            // sort the measurements by id so the sparse insert is O(1)
            std::sort(pose.ProjResiduals.begin(), pose.ProjResiduals.end());
            for( const int id: pose.ProjResiduals ) {
                const ProjectionResidual& res = m_vProjResiduals[id];
                // insert the jacobians into the sparse matrices
                // The weight is only multiplied by the transpose matrix, this is so we can perform Jt*W*J*dx = Jt*W*r
                auto dZ_dP = res.MeasPoseId == pose.Id ? res.dZ_dPm : res.dZ_dPr;
                if( res.RefPoseId == pose.Id ){

                    //std::cout << "Adding reference jacobian for pose id " << pose.Id << " and residual id " << res.ResidualId << " wih ref pose id " << res.RefPoseId << " and meas pose id " << res.MeasPoseId << std::endl;
                    // std::cout << "Jacobian is " << std::endl << dZ_dP << std::endl;
                    // dZ_dP.setZero();
                    // dZ_dP *= -1;
                }else
                {
                    //std::cout << "Adding measurement jacobian for pose id " << pose.Id << " and residual id " << res.ResidualId << " wih ref pose id " << res.RefPoseId << " and meas pose id " << res.MeasPoseId <<  std::endl;
                    // dZ_dP.setZero();
                }
                m_Jpr.insert( res.ResidualId, pose.OptId ).setZero().template block<2,6>(0,0) = dZ_dP;
                m_Jprt.insert( pose.OptId, res.ResidualId ).setZero().template block<6,2>(0,0) = dZ_dP.transpose() * res.W;

                if(PoseSize == 21){
                    const auto& dZ_dTvs = res.MeasPoseId == pose.Id ? res.dZ_dTvs_m : res.dZ_dTvs_r;
                    m_Jpr.coeffRef( res.ResidualId, pose.OptId ).template block<2,6>(0,15) = dZ_dTvs.template block(0,0,2,6);
                    m_Jprt.coeffRef( pose.OptId, res.ResidualId ).template block<6,2>(15,0) = dZ_dTvs.transpose().template block(0,0,6,2) * res.W;
//                    JprVal.template block<2,6>(0,15) = res.dZ_dTvs.template block(0,0,2,6);
//                    JprtVal.template block<6,2>(15,0) = res.dZ_dTvs.transpose().template block(0,0,6,2) * res.W;
                }
            }

            // add the pose/pose constraints
            std::sort(pose.BinaryResiduals.begin(), pose.BinaryResiduals.end());
            for( const int id: pose.BinaryResiduals ) {
                const BinaryResidual& res = m_vBinaryResiduals[id];
                const Eigen::Matrix<Scalar,6,6>& dZ_dZ = res.PoseAId == pose.Id ? res.dZ_dX1 : res.dZ_dX2;
                m_Jpp.insert( res.ResidualId, pose.OptId ).setZero().template block<6,6>(0,0) = dZ_dZ;
                m_Jppt.insert( pose.OptId, res.ResidualId ).setZero().template block<6,6>(0,0) = dZ_dZ.transpose() * res.W;
            }

            // add the unary constraints
            std::sort(pose.UnaryResiduals.begin(), pose.UnaryResiduals.end());
            for( const int id: pose.UnaryResiduals ) {
                const UnaryResidual& res = m_vUnaryResiduals[id];
                m_Ju.insert( res.ResidualId, pose.OptId ).setZero().template block<6,6>(0,0) = res.dZ_dX;
                m_Jut.insert( pose.OptId, res.ResidualId ).setZero().template block<6,6>(0,0) = res.dZ_dX.transpose() * res.W;
            }

            std::sort(pose.ImuResiduals.begin(), pose.ImuResiduals.end());
            for( const int id: pose.ImuResiduals ) {
                const ImuResidual& res = m_vImuResiduals[id];
                Eigen::Matrix<Scalar,ImuResidual::ResSize,PoseSize> dZ_dZ = res.PoseAId == pose.Id ? res.dZ_dX1 : res.dZ_dX2;
                m_Ji.insert( res.ResidualId, pose.OptId ).setZero().template block<ImuResidual::ResSize,PoseSize>(0,0) = dZ_dZ;
                // dZ_dZ.template block<3,PoseSize>(6,0) *= 0.1;
                m_Jit.insert( pose.OptId, res.ResidualId ).setZero().template block<PoseSize,ImuResidual::ResSize>(0,0) = dZ_dZ.transpose() * res.W;
            }
        }
    }

    // fill in calibration jacobians
    if( CalibSize > 0){
        for( const auto& res : m_vImuResiduals ){
            // include gravity terms (t total)
            if( CalibSize > 0 ){
                // std::cout << "Residual " << res.ResidualId << " : dZ_dG: " << res.dZ_dG.template block<9,2>(0,0).transpose() << std::endl;
                Eigen::Matrix<Scalar,9,2> dZ_dG = res.dZ_dG;
                m_Jki.insert(res.ResidualId, 0 ).setZero().template block(0,0,9,2) = dZ_dG.template block(0,0,9,2);
                // dZ_dG.template block<3,2>(6,0) *= 0.1;
                m_Jkit.insert( 0, res.ResidualId ).setZero().template block(0,0,2,9) = dZ_dG.transpose().template block(0,0,2,9) * res.W;
            }

            // include bias terms (6 total)
            /*if( CalibSize > 2 ){
                Eigen::Matrix<Scalar,9,6> dZ_dB = res.dZ_dB.template block(0,0,9,6);
                m_Jki.coeffRef(res.ResidualId,0).setZero().template block(0,2,9,6) = dZ_dB;
                m_Jkit.coeffRef(0,res.ResidualId).setZero().template block(2,0,6,9) = dZ_dB.transpose() * res.W;
                // m_Jkit.coeffRef(0,res.ResidualId).template block(2,0,6,9) = res.dZ_dB.transpose().template block(0,0,6,9) * res.W;
            }*/
        }

        for( const auto& res : m_vProjResiduals ){
            // include imu to camera terms (6 total)
            if( CalibSize > 2 ){
                Eigen::Matrix<Scalar,2,6> dZ_dTvs = res.dZ_dTvs_m;
                m_Jkpr.coeffRef(res.ResidualId,0).setZero().template block(0,2,2,6) = dZ_dTvs.template block(0,0,2,6);
                m_Jkprt.coeffRef(0,res.ResidualId).setZero().template block(2,0,6,2) = dZ_dTvs.template block(0,0,2,6).transpose() * res.W;
                // m_Jkit.coeffRef(0,res.ResidualId).template block(2,0,6,9) = res.dZ_dB.transpose().template block(0,0,6,9) * res.W;
            }
        }
    }

    for( Landmark& lm : m_vLandmarks ){
        if( lm.IsActive ){
            // sort the measurements by id so the sparse insert is O(1)
            std::sort(lm.ProjResiduals.begin(), lm.ProjResiduals.end());
            for( const int id: lm.ProjResiduals ) {
                const ProjectionResidual& res = m_vProjResiduals[id];
    //                std::cout << "      Adding jacobian cell for measurement " << pMeas->MeasurementId << " in landmark " << pMeas->LandmarkId << std::endl;
                m_Jl.insert( res.ResidualId, lm.OptId ) = res.dZ_dX;
                m_Jlt.insert( lm.OptId, res.ResidualId ) = res.dZ_dX.transpose() * res.W;
            }
        }
    }

    std::cout << "Jacobian insertion took " << Toc(dTime) << " seconds. " << std::endl;    
}

// specializations
// template class BundleAdjuster<REAL_TYPE, ba::NOT_USED,9,8>;
template class BundleAdjuster<REAL_TYPE, 1,6,0>;
// template class BundleAdjuster<REAL_TYPE, 1,15,8>;
template class BundleAdjuster<REAL_TYPE, 1,15,2>;
// template class BundleAdjuster<REAL_TYPE, 1,21,2>;
// template class BundleAdjuster<double, 3,9>;


}


