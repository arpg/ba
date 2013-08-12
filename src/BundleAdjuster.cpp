#include <ba/BundleAdjuster.h>

namespace ba {

///////////////////////////////////////////////////////////////////////////////////////////////
template< typename Scalar,int LmSize, int PoseSize >
void BundleAdjuster<Scalar,LmSize,PoseSize>::Solve(const unsigned int uMaxIter)
{
    Eigen::IOFormat cleanFmt(2, 0, ", ", ";\n" , "" , "");
    // Scalar dTime = Tic();
    // first build the jacobian and residual vector
    // TODO: do more than 1 iteration
    for( unsigned int kk = 0 ; kk < uMaxIter ; kk++){
        Scalar dTime = Tic();
        _BuildProblem();
        // std::cout << "Build problem took " << Toc(dTime) << " seconds." << std::endl;

        dTime = Tic();

        const unsigned int uNumPoses = m_uNumActivePoses;
        const unsigned int uNumLm = m_vLandmarks.size();
//        const unsigned int uNumMeas = m_vMeasurements.size();
        // calculate bp and bl
        Scalar dMatTime = Tic();
        VectorXt bp(uNumPoses*PoseSize);
        VectorXt bl;
        Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,LmSize,LmSize> > V_inv(uNumLm,uNumLm);
        VectorXt rhs_p(uNumPoses*PoseSize);
        Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,LmSize,PoseSize> > Wt(uNumLm,uNumPoses);
        Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> S(uNumPoses*PoseSize,uNumPoses*PoseSize);
        std::cout << "  Rhs vector mult took " << Toc(dMatTime) << " seconds." << std::endl;

        dMatTime = Tic();

        // TODO: suboptimal, the matrices are symmetric. We should only multipl one half
        Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,PoseSize,PoseSize> > U(uNumPoses,uNumPoses);
        U.setZero();
        bp.setZero();

        if( m_vProjResiduals.size() > 0 ){
            Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,PoseSize,PoseSize> > Jprt_Jpr(uNumPoses,uNumPoses);
            Eigen::SparseBlockProduct(m_Jprt,m_Jpr,Jprt_Jpr);
            auto Temp = U;
            Eigen::SparseBlockAdd(Temp,Jprt_Jpr,U);

            VectorXt Jprt_Rpr(uNumPoses*PoseSize);
            Eigen::SparseBlockVectorProductDenseResult(m_Jprt,m_Rpr,Jprt_Rpr);
            bp += Jprt_Rpr;
        }

        // add the contribution from the binary terms if any
        if( m_vBinaryResiduals.size() > 0 ) {
            Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,PoseSize,PoseSize> > Jppt_Jpp(uNumPoses,uNumPoses);
            Eigen::SparseBlockProduct(m_Jppt,m_Jpp,Jppt_Jpp);
            auto Temp = U;
            Eigen::SparseBlockAdd(Temp,Jppt_Jpp,U);

            VectorXt Jppt_Rpp(uNumPoses*PoseSize);
            Eigen::SparseBlockVectorProductDenseResult(m_Jppt,m_Rpp,Jppt_Rpp);
            bp += Jppt_Rpp;
        }

        // add the contribution from the unary terms if any
        if( m_vUnaryResiduals.size() > 0 ) {
            Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,PoseSize,PoseSize> > Jut_Ju(uNumPoses,uNumPoses);
            Eigen::SparseBlockProduct(m_Jut,m_Ju,Jut_Ju);
            auto Temp = U;
            Eigen::SparseBlockAdd(Temp,Jut_Ju,U);

            VectorXt Jut_Ru(uNumPoses*PoseSize);
            Eigen::SparseBlockVectorProductDenseResult(m_Jut,m_Ru,Jut_Ru);
            bp += Jut_Ru;

//                Eigen::LoadDenseFromSparse(U,S);
//                std::cout << "Dense S matrix is " << S.format(cleanFmt) << std::endl;
        }

        // add the contribution from the imu terms if any
        if( m_vImuResiduals.size() > 0 ) {
            Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,PoseSize,PoseSize> > Jit_Ji(uNumPoses,uNumPoses);
            Eigen::SparseBlockProduct(m_Jit,m_Ji,Jit_Ji);
            auto Temp = U;
            Eigen::LoadDenseFromSparse(Jit_Ji,S);
            Eigen::SparseBlockAdd(Temp,Jit_Ji,U);

            VectorXt Jit_Ri(uNumPoses*PoseSize);
            Eigen::SparseBlockVectorProductDenseResult(m_Jit,m_Ri,Jit_Ri);
            bp += Jit_Ri;
                // Eigen::LoadDenseFromSparse(U,S);
                // std::cout << "Dense S matrix is " << S.format(cleanFmt) << std::endl;
        }

        if( LmSize > 0 && uNumLm > 0) {
            bl.resize(uNumLm*LmSize);
            Eigen::SparseBlockVectorProductDenseResult(m_Jlt,m_Rpr,bl);

//                 std::cout << "bl = " << bl.transpose().format(cleanFmt) << std::endl;

            Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,LmSize,LmSize> > V(uNumLm,uNumLm);
            Eigen::SparseBlockProduct(m_Jlt,m_Jl,V);

//                 Eigen::LoadDenseFromSparse(V,S);
//                 std::cout << "V is " << S.format(cleanFmt) << std::endl;

            // TODO this is really suboptimal, we should write a function to transpose the matrix
            Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,PoseSize,LmSize> > W(uNumPoses,uNumLm);
            Eigen::SparseBlockProduct(m_Jprt,m_Jl,W);

            Eigen::SparseBlockProduct(m_Jlt,m_Jpr,Wt);

//                 Eigen::LoadDenseFromSparse(W,S);
//                 std::cout << "W is " << S.format(cleanFmt) << std::endl;

            // std::cout << "  Outer product took " << Toc(dMatTime) << " seconds." << std::endl;

            dMatTime = Tic();
            // calculate the inverse of the map hessian (it should be diagonal, unless a measurement is of more than
            // one landmark, which doesn't make sense)
            for(size_t ii = 0 ; ii < uNumLm ; ii++){
                if(LmSize == 1){
                    if(V.coeffRef(ii,ii)(0,0) < 1e-6){
                        V.coeffRef(ii,ii)(0,0) += 1e-6;
                    }
                }
                V_inv.coeffRef(ii,ii) = V.coeffRef(ii,ii).inverse();
            }
            // std::cout << "  Inversion of V took " << Toc(dMatTime) << " seconds." << std::endl;
//                 Eigen::LoadDenseFromSparse(V_inv,S);
//                 std::cout << "Vinv is " << S.format(cleanFmt) << std::endl;

            dMatTime = Tic();
            // attempt to solve for the poses. W_V_inv is used later on, so we cache it
            Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,PoseSize,LmSize> > W_V_inv(uNumPoses,uNumLm);
            Eigen::SparseBlockProduct(W,V_inv,W_V_inv);

            Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,PoseSize,PoseSize> > WV_invWt(uNumPoses,uNumPoses);
            Eigen::SparseBlockProduct(W_V_inv,Wt,WV_invWt);

            // Eigen::LoadDenseFromSparse(WV_invWt,S);
            // std::cout << "WV_invWt is " << S.format(cleanFmt) << std::endl;

            // this in-place operation should be fine for subtraction
            Eigen::SparseBlockSubtractDenseResult(U,WV_invWt,S);

            // now form the rhs for the pose equations
            VectorXt WV_inv_bl(uNumPoses*PoseSize);
            Eigen::SparseBlockVectorProductDenseResult(W_V_inv,bl,WV_inv_bl);

            rhs_p = bp - WV_inv_bl;

//                 std::cout << "Dense S matrix is " << S.format(cleanFmt) << std::endl;
//                 std::cout << "Dense rhs matrix is " << rhs_p.transpose().format(cleanFmt) << std::endl;
            // std::cout << "  Rhs calculation took " << Toc(dMatTime) << " seconds." << std::endl;
        }else{
            Eigen::LoadDenseFromSparse(U,S);
            rhs_p = bp;
                std::cout << "Dense S matrix is " << S.format(cleanFmt) << std::endl;
                std::cout << "Dense rhs matrix is " << rhs_p.transpose().format(cleanFmt) << std::endl;
        }

        // std::cout << "Setup took " << Toc(dTime) << " seconds." << std::endl;

        // now we have to solve for the pose constraints
        dTime = Tic();
        VectorXt delta_p = uNumPoses == 0 ? VectorXt() : S.ldlt().solve(rhs_p);
        // std::cout << "Cholesky solve of " << uNumPoses << " by " << uNumPoses << "matrix took " << Toc(dTime) << " seconds." << std::endl;

        if( uNumLm > 0) {
            dTime = Tic();
            VectorXt delta_l;
            delta_l.resize(uNumLm*LmSize);
            VectorXt Wt_delta_p;
            Wt_delta_p.resize(uNumLm*LmSize );
            Eigen::SparseBlockVectorProductDenseResult(Wt,delta_p,Wt_delta_p);
            VectorXt rhs_l;
            rhs_l.resize(uNumLm*LmSize );
            rhs_l =  bl - Wt_delta_p;

            for(size_t ii = 0 ; ii < uNumLm ; ii++){
                delta_l.template block<LmSize,1>( ii*LmSize, 0 ).noalias() =  V_inv.coeff(ii,ii)*rhs_l.template block<LmSize,1>(ii*LmSize,0);
            }

            // update the landmarks
            for (size_t ii = 0 ; ii < uNumLm ; ii++){
                if(LmSize == 1){
                    m_vLandmarks[ii].Xs.template tail<LmSize>() += delta_l.template segment<LmSize>(m_vLandmarks[ii].OptId*LmSize);
                }else{
                    m_vLandmarks[ii].Xs.template head<LmSize>() += delta_l.template segment<LmSize>(m_vLandmarks[ii].OptId*LmSize);
                }
            }
            std::cout << "Backsubstitution of " << uNumLm << " landmarks took " << Toc(dTime) << " seconds." << std::endl;
        }

        // std::cout << delta_l << std::endl;

        // update poses
        // std::cout << "Updating " << uNumPoses << " active poses." << std::endl;
        for (size_t ii = 0 ; ii < m_vPoses.size() ; ii++){
            // only update active poses, as inactive ones are not part of the optimization
            if( m_vPoses[ii].IsActive ){
                 // std::cout << "Pose delta for " << ii << " is " << delta_p.template block<PoseSize,1>(m_vPoses[ii].OptId*PoseSize,0).transpose() << std::endl;
                 m_vPoses[ii].Twp = exp_decoupled<Scalar>(m_vPoses[ii].Twp,delta_p.template block<6,1>(m_vPoses[ii].OptId*PoseSize,0));
                 // m_vPoses[ii].Twp = m_vPoses[ii].Twp * Sophus::SE3d::exp(delta_p.template block<6,1>(m_vPoses[ii].OptId*PoseSize,0));
                // update the velocities if they are parametrized
                if(PoseSize == 9){
                    m_vPoses[ii].V += delta_p.template block<3,1>(m_vPoses[ii].OptId*PoseSize+6,0);
                    std::cout << "Velocity for pose " << ii << " is " << m_vPoses[ii].V << std::endl;
                }
                // clear the vector of Tsw values as they will need to be recalculated
                m_vPoses[ii].Tsw.clear();
            }
            // else{
            //  std::cout << " Pose " << ii << " is inactive." << std::endl;
            //  }
        }
        std::cout << "BA iteration " << kk <<  " error: " << m_Rpr.norm() + m_Ru.norm() + m_Rpp.norm() + m_Ri.norm() << std::endl;
    }

    // update the global position of the landmarks from the sensor position
    for (Landmark& lm : m_vLandmarks) {
        lm.Xw = MultHomogeneous(m_vPoses[lm.RefPoseId].GetTsw(lm.RefCamId,m_Rig).inverse(), lm.Xs);
    }
        // std::cout << "Solve took " << Toc(dTime) << " seconds." << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////
template< typename Scalar,int LmSize, int PoseSize >
void BundleAdjuster<Scalar,LmSize,PoseSize>::_BuildProblem()
{
    Eigen::IOFormat cleanFmt(4, 0, ", ", ";\n" , "" , "");

    // resize as needed
    const unsigned int uNumPoses = m_uNumActivePoses;
    const unsigned int uNumLm = m_vLandmarks.size();
    const unsigned int uNumProjRes = m_vProjResiduals.size();
    const unsigned int uNumBinRes = m_vBinaryResiduals.size();
    const unsigned int uNumUnRes = m_vUnaryResiduals.size();
    const unsigned int uNumImuRes = m_vImuResiduals.size();

    m_Jpr.resize(uNumProjRes,uNumPoses);
    m_Jprt.resize(uNumPoses,uNumProjRes);
    m_Jl.resize(uNumProjRes,uNumLm);
    m_Jlt.resize(uNumLm,uNumProjRes);
    m_Rpr.resize(uNumProjRes*ProjectionResidual::ResSize);

    m_Jpp.resize(uNumBinRes,uNumPoses);
    m_Jppt.resize(uNumPoses,uNumBinRes);
    m_Rpp.resize(uNumBinRes*BinaryResidual::ResSize);

    m_Ju.resize(uNumUnRes,uNumPoses);
    m_Jut.resize(uNumPoses,uNumUnRes);
    m_Ru.resize(uNumUnRes*UnaryResidual::ResSize);

    m_Ji.resize(uNumImuRes,uNumPoses);
    m_Jit.resize(uNumPoses,uNumImuRes);
    m_Jki.resize(uNumImuRes*ImuResidual::ResSize,8);
    m_Ri.resize(uNumImuRes*ImuResidual::ResSize);


    // these calls remove all the blocks, but KEEP allocated memory as long as the object is alive
    m_Jpr.setZero();
    m_Jprt.setZero();
    m_Rpr.setZero();

    m_Jpp.setZero();
    m_Jppt.setZero();
    m_Rpp.setZero();

    m_Ju.setZero();
    m_Jut.setZero();
    m_Ru.setZero();

    m_Ji.setZero();
    m_Jit.setZero();
    // m_Jki.setZero();
    // m_Jkit.setZero();
    m_Ri.setZero();

    m_Jl.setZero();
    m_Jlt.setZero();


    // used to store errors for robust norm calculation
    m_vErrors.reserve(uNumProjRes);
    m_vErrors.clear();


    Scalar dPortionTransfer = 0, dPortionJac = 0, dPortionSparse = 0;

    // set all jacobians
    Scalar dTime = Tic();
    for( ProjectionResidual& res : m_vProjResiduals ){
        Scalar dPortTime = Tic();
        // calculate measurement jacobians

        // Tsw = T_cv * T_vw
        Landmark& lm = m_vLandmarks[res.LandmarkId];
        Pose& pose = m_vPoses[res.PoseId];
        Pose& refPose = m_vPoses[lm.RefPoseId];
        lm.Xs = MultHomogeneous(refPose.GetTsw(lm.RefCamId,m_Rig) ,lm.Xw);
        const SE3t parentTws = refPose.GetTsw(lm.RefCamId,m_Rig).inverse();

        const Vector2t p = m_Rig.cameras[res.CameraId].camera.Transfer3D(pose.GetTsw(res.CameraId,m_Rig)*parentTws,
                                                                                lm.Xs.template head<3>(),lm.Xs(3));
        res.Residual = res.Z - p;
        // std::cout << "Residual for meas " << res.ResidualId << " and landmark " << res.LandmarkId << " with camera " << res.CameraId << " is " << res.Residual.transpose() << std::endl;

        // this array is used to calculate the robust norm
        m_vErrors.push_back(res.Residual.squaredNorm());

        dPortionTransfer += Toc(dPortTime);

        dPortTime = Tic();
        const Eigen::Matrix<Scalar,2,4> dTdP = m_Rig.cameras[res.CameraId].camera.dTransfer3D_dP(pose.GetTsw(res.CameraId,m_Rig)*parentTws,
                                                                                                 lm.Xs.template head<3>(),lm.Xs(3));
        res.dZ_dX = dTdP.template block<2,LmSize>( 0, LmSize == 3 ? 0 : 3 );
        // std::cout << "Jacobian for landmark " << res.LandmarkId << ", with Xs = " << std::endl << lm.Xs.transpose().format(cleanFmt) << " is " << res.dZ_dX;

        if( pose.IsActive ) {
            const Eigen::Matrix<Scalar,2,4> dTdP = m_Rig.cameras[res.CameraId].camera.dTransfer3D_dP(pose.GetTsw(res.CameraId,m_Rig),
                                                                                                     lm.Xw.template head<3>(),lm.Xw(3));
            res.dZ_dP.template block<2,3>(0,0) = -dTdP.template block<2,3>(0,0);
            for(unsigned int ii=3; ii<6; ++ii){
                res.dZ_dP.template block<2,1>(0,ii) = dTdP * -Sophus::SE3Group<Scalar>::generator(ii) * lm.Xw;
            }

//            Eigen::Matrix<Scalar,2,6> J_fd;
//            Scalar dEps = 1e-6;
//            for(int ii = 0; ii < 6 ; ii++) {
//                Eigen::Matrix<Scalar,6,1> delta;
//                delta.setZero();
//                delta[ii] = dEps;
//                // SE3t Tsw = (pose.Twp*SE3t::exp(delta)*m_Rig.cameras[meas.CameraId].T_wc).inverse();
//                SE3t Tsw = (exp_decoupled(pose.Twp,delta)*m_Rig.cameras[res.CameraId].T_wc).inverse();
//                const Vector2t pPlus = m_Rig.cameras[res.CameraId].camera.Transfer3D(Tsw,lm.Xw.template head(3),lm.Xw[3]);
//                delta[ii] = -dEps;
//                // Tsw = (pose.Twp*SE3t::exp(delta)*m_Rig.cameras[meas.CameraId].T_wc).inverse();
//                Tsw = (exp_decoupled(pose.Twp,delta)*m_Rig.cameras[res.CameraId].T_wc).inverse();
//                const Vector2t pMinus = m_Rig.cameras[res.CameraId].camera.Transfer3D(Tsw,lm.Xw.template head(3),lm.Xw[3]);
//                J_fd.col(ii) = (pPlus-pMinus)/(2*dEps);
//            }
//            std::cout << "Jproj:" << res.dZ_dP << std::endl;
//            std::cout << "Jproj_fd:" << J_fd << std::endl;
        }


        dPortionJac += Toc(dPortTime);

        // set the residual in m_R which is dense
        m_Rpr.template segment<ProjectionResidual::ResSize>(res.ResidualOffset) = res.Residual;
    }

    // build binary residual jacobians
    for( BinaryResidual& res : m_vBinaryResiduals ){
        const SE3t& Twa = m_vPoses[res.PoseAId].Twp;
        const SE3t& Twb = m_vPoses[res.PoseBId].Twp;
        res.dZ_dX1 = dLog_dX(Twa, res.Tab * Twb.inverse());
        // the negative sign here is because exp(x) is inside the inverse when we invert (Twb*exp(x)).inverse
        res.dZ_dX2 = -dLog_dX(Twa * res.Tab , Twb.inverse());

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
            m_Rpp.template segment<BinaryResidual::ResSize>(res.ResidualOffset) = log_decoupled( Twa*res.Tab,Twb.inverse() );
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

        m_Ru.template segment<UnaryResidual::ResSize>(res.ResidualOffset) = log_decoupled( Twp,res.Twp );
        // m_Ru.template segment<UnaryResidual::ResSize>(res.ResidualOffset) = (Twp*res.Twp.inverse()).log();
    }

    for( ImuResidual& res : m_vImuResiduals ){
        // set up the initial pose for the integration
        const Vector3t gravity = ImuCalibration::GetGravityVector(m_Imu.G);
        const Eigen::Matrix<Scalar,3,2> dGravity = ImuCalibration::dGravity_dDirection(m_Imu.G);
        const Pose& poseA = m_vPoses[res.PoseAId];
        const Pose& poseB = m_vPoses[res.PoseBId];


//            ImuPose imuPose(poseA.Twp,poseA.V,Vector3t::Zero(),poseA.Time);
//            ImuMeasurement* pPrevMeas = 0;
//            res.Poses.clear();
//            res.Poses.reserve(res.Measurements.size()+1);
//            res.Poses.push_back(imuPose);

//            // integrate forward in time, and retain all the poses
//            for(ImuMeasurement& meas : res.Measurements){
//                if(pPrevMeas != 0){
//                    totalDt += meas.Time - pPrevMeas->Time;
//                    imuPose = ImuResidual::IntegrateImu(imuPose,*pPrevMeas,meas,m_Imu.Bg,m_Imu.Ba,gravity);
//                    res.Poses.push_back(imuPose);
//                }
//                pPrevMeas = &meas;
//            }

        Eigen::Matrix<Scalar,10,6> jb_q;
        // Eigen::Matrix<Scalar,10,10> jb_y;
        ImuPose imuPose = ImuResidual::IntegrateResidual(poseA,res.Measurements,m_Imu.Bg,m_Imu.Ba,gravity,res.Poses,&jb_q/*,&jb_y*/);
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

        SE3t Tstar(Sophus::SO3Group<Scalar>(),(poseA.V*totalDt - 0.5*gravity*powi(totalDt,2)));

        // calculate the derivative of the lie log with respect to the tangent plane at Twa
        // Eigen::Matrix<Scalar,6,6> dLog  = dLog_dX(Tstar*Twa,Tab_0*Twb.inverse());
        Eigen::Matrix<Scalar,6,6> dLog;//  = dLog_decoupled_dX(Tstar*Twa,Twb*Tab_0.inverse());
        // dLog.template block<3,3>(0,3) = dqx_dq(imuPose.Twp.unit_quaternion(),imuPose.Twp.translation()) * dq1q2_dq2((Tstar*Twa).unit_quaternion())*dqExp_dw<double>(Eigen::Vector3d::Zero());
        // const Eigen::Matrix<Scalar,6,6> dLog_b  = dLog_decoupled_dX(imuPose.Twp,Twb.inverse());

        // rotation error  = log(A) = log(Twa * Tab_0 * Twb.inverse();
        // derror/dw = dlog_dq * aA/dTwa * dTwa/dw
        dLog.template block<3,3>(0,0) = Twa.so3().matrix();
        dLog.template block<3,3>(3,3) = dLog_dq((Twa.so3()*Tab_0.so3()*Twb.so3().inverse()).unit_quaternion()) *
                dq1q2_dq1((Tab_0.so3() * Twb.so3().inverse()).unit_quaternion()) *
                dq1q2_dq2(Twa.unit_quaternion()) *
                dqExp_dw<Scalar>(Eigen::Matrix<Scalar,3,1>::Zero());

        dLog.template block<3,3>(0,3) = dqx_dq<double>((Twa).unit_quaternion(),Tab_0.translation())*
                dq1q2_dq2(Twa.unit_quaternion()) *
                dqExp_dw<Scalar>(Eigen::Matrix<Scalar,3,1>::Zero());

        // now add the log jacobians to the bias jacobian terms
        res.dZ_dB.template block<3,6>(3,0) = dLog_dq((imuPose.Twp.so3() * Twb.so3().inverse()).unit_quaternion()) *
                                        dq1q2_dq1(Twb.so3().inverse().unit_quaternion()) * jb_q.template block<4,6>(3,0);

//            const SE3t se3 = imuPose.Twp*Twb.inverse();
//            Vector3t upsilon_omega;
//            Scalar theta;
//            upsilon_omega = Sophus::SO3Group<Scalar>::logAndTheta(se3.so3(), &theta);

//              const Matrix3t Omega = Sophus::SO3Group<Scalar>::hat(upsilon_omega.template tail<3>());
//              const Matrix3t V_inv =
//                  ( Matrix3t::Identity() - 0.5*Omega + ( 1 - theta/(2*tan(theta/2))) / (theta*theta)*(Omega*Omega) );
//            std::cout << "Vinv " << std::endl << V_inv << std::endl;
//            upsilon_omega = V_inv*se3.translation();

//            Eigen::Matrix<Scalar,6,6> temp;
//            temp.block<3,6>(0,0) = V_inv * jb_q.block<3,6>(0,0);
//            temp.block<3,6>(3,0) =dLog_dq((imuPose.Twp.so3() * Twb.so3().inverse()).unit_quaternion()) *
//                                    dq1q2_dq1(Twb.so3().inverse().unit_quaternion()) * jb_q.block<4,6>(3,0);
        Eigen::Matrix<Scalar,6,7> dLog_dqx;
        for(int ii = 0 ; ii < 7 ; ii++){
            Scalar dEps = 1e-9;
            Eigen::Matrix<Scalar,7,1> epsVec = Eigen::Matrix<Scalar,7,1>::Zero();
            epsVec[ii] += dEps;
            SE3t Twp_imu_plus = imuPose.Twp;
            Twp_imu_plus.translation() += epsVec.template head<3>();
            const Vector4t coeffsPlus = Twp_imu_plus.unit_quaternion().coeffs() + epsVec.template tail<4>();
            memcpy(Twp_imu_plus.so3().data(),coeffsPlus.data(),sizeof(Scalar)*4);
            const Vector6t resPlus = log_decoupled(Twp_imu_plus , Twb.inverse());

            epsVec[ii] -= 2*dEps;
            SE3t Twp_imu_minus = imuPose.Twp;
            Twp_imu_minus.translation() += epsVec.template head<3>();
            const Vector4t coeffsMinus = Twp_imu_minus.unit_quaternion().coeffs() + epsVec.template tail<4>();
            memcpy(Twp_imu_minus.so3().data(),coeffsMinus.data(),sizeof(Scalar)*4);
            const Vector6t resMinus = log_decoupled(Twp_imu_minus , Twb.inverse());
            dLog_dqx.col(ii) = (resPlus-resMinus)/(2*dEps);
        }
        std::cout << "dLog_dqx = " << std::endl << dLog_dqx.format(cleanFmt) << std::endl;
//            res.dZ_dB.template block<3,6>(0,0) = (dLog_dqx*jb_q.template block<7,6>(0,0)).template block<3,6>(0,0);
//            res.dZ_dB.template block<3,6>(0,0) = dLog_b.template block<3,3>(0,0) * imuPose.Twp.so3().inverse().matrix() * jb_q.template block<3,6>(0,0) +
//                                        dLog_dq((imuPose.Twp.so3() * Twb.so3().inverse()).unit_quaternion()) * jb_q.template block<4,6>(3,0);
        res.dZ_dB.template block<3,6>(0,0) = jb_q.template block<3,6>(0,0);
        res.dZ_dB.template block<3,6>(6,0) = jb_q.template block<3,6>(7,0);

        res.dZ_dX1.template block<6,6>(0,0) = dLog;
        // Twa^-1 is multiplied here as we need the velocity derivative in the frame of pose A, as the log is taken from this frame
        res.dZ_dX1.template block<3,3>(0,6) = /*dLog.template block<3,3>(0,0) * Twa.so3().inverse().matrix() * */Matrix3t::Identity()*totalDt;
        for( int ii = 0; ii < 3 ; ++ii ){
            res.dZ_dX1.template block<3,1>(6,3+ii) = Twa.so3().matrix() * Sophus::SO3Group<Scalar>::generator(ii) * Vab_0;
        }
        res.dZ_dX1.template block<3,3>(6,6) = Matrix3t::Identity();
        // the - sign is here because of the exp(-x) within the log
        res.dZ_dX2.template block<6,6>(0,0) = -dLog_decoupled_dX(imuPose.Twp,Twb);//-dLog_dX(Twa*Tab,Twb.inverse());
        res.dZ_dX2.template block<3,3>(6,6) = -Matrix3t::Identity();

        res.dZ_dG.template block<3,2>(0,0) = /*dLog.template block<3,3>(0,0) * Twa.so3().inverse().matrix() **/
                                    -0.5*powi(totalDt,2)*Matrix3t::Identity()*dGravity;
        res.dZ_dG.template block<3,2>(6,0) = -totalDt*Matrix3t::Identity()*dGravity;

        Scalar dEps = 1e-12;
        Eigen::Matrix<Scalar,6,6> Jlog;
        for(int ii = 0 ; ii < 6 ; ii++){
            Vector6t eps = Vector6t::Zero();
            eps[ii] += dEps;
            const Vector6t resPlus = log_decoupled(exp_decoupled(Twa,eps),Tab*Twb.inverse());
            eps[ii] -= 2*dEps;
            const Vector6t resMinus = log_decoupled(exp_decoupled(Twa,eps),Tab*Twb.inverse());
            Jlog.col(ii) = (resPlus-resMinus)/(2*dEps);
        }

        std::cout << "Jlog = [" << dLog_decoupled_dX(Twa,Tab*Twb.inverse()).format(cleanFmt) << "]" << std::endl;
        std::cout << "Jlogf = [" << Jlog.format(cleanFmt) << "]" << std::endl;
        std::cout << "Jlog - Jlogf = [" << (dLog_decoupled_dX(Twa,Tab*Twb.inverse())- Jlog).format(cleanFmt) << "]" << std::endl;

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
            y0_eps.Twp = exp_decoupled<double>(y0_eps.Twp,epsVec.template head<6>());
            y0_eps.V += epsVec.template tail<3>();
            Eigen::Matrix<Scalar,9,1> r_plus;
            r_plus.template head<6>() = log_decoupled(imuPose.Twp,y0_eps.Twp);
            r_plus.template tail<3>() = imuPose.V - y0_eps.V;



            epsVec[ii] -= 2*dEps;
            y0_eps = ImuPose(poseB.Twp,poseB.V, Vector3t::Zero(),0);;
            y0_eps.Twp = exp_decoupled<double>(y0_eps.Twp,epsVec.template head<6>());
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
            // poseEps.Twp = exp_decoupled(poseEps.Twp,eps);
            poseEps.Twp = poseEps.Twp * Sophus::SE3d::exp(eps);
            std::vector<ImuPose> poses;
            const ImuPose imuPosePlus = ImuResidual::IntegrateResidual(poseEps,res.Measurements,m_Imu.Bg,m_Imu.Ba,gravity,poses);
            // const Vector6t dErrorPlus = log_decoupled(imuPosePlus.Twp, Twb);
            const Vector6t dErrorPlus = log_decoupled(imuPosePlus.Twp, Twb);
            const Vector3t vErrorPlus = imuPosePlus.V - poseB.V;
            eps[ii] -= 2*dEps;
            poseEps = poseA;
            // poseEps.Twp = exp_decoupled(poseEps.Twp,eps);
            poseEps.Twp = poseEps.Twp * Sophus::SE3d::exp(eps);
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
            const ImuPose imuPosePlus = ImuResidual::IntegrateResidual(poseA,res.Measurements,m_Imu.Bg,m_Imu.Ba,m_Imu.GetGravityVector((Vector2t)(m_Imu.G+eps)),poses);
            const Vector6t dErrorPlus = log_decoupled(imuPosePlus.Twp, Twb);
//                std::cout << "Pose plus: " << imuPosePlus.Twp.matrix() << std::endl;
            const Vector3t vErrorPlus = imuPosePlus.V - poseB.V;
            eps[ii] -= 2*dEps;
            poses.clear();
            const ImuPose imuPoseMinus = ImuResidual::IntegrateResidual(poseA,res.Measurements,m_Imu.Bg,m_Imu.Ba,m_Imu.GetGravityVector((Vector2t)(m_Imu.G+eps)),poses);
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
//                Eigen::Matrix<Scalar,10,1> plusVec;
//                plusVec.template head<3>() = imuPosePlus.Twp.translation();
//                plusVec.template segment<4>(3) = imuPosePlus.Twp.so3().unit_quaternion().coeffs();
//                plusVec.template tail<3>() = imuPosePlus.V;
            const Vector6t dErrorPlus = log_decoupled(imuPosePlus.Twp, Twb);
//                std::cout << "Pose plus: " << imuPosePlus.Twp.matrix() << std::endl;
            const Vector3t vErrorPlus = imuPosePlus.V - poseB.V;

            eps[ii] -= 2*dEps;
            const Vector6t minusBiases = biasVec + eps;
            poses.clear();
            const ImuPose imuPoseMinus = ImuResidual::IntegrateResidual(poseA,res.Measurements,minusBiases.template head<3>(),minusBiases.template tail<3>(),gravity,poses);
//                Eigen::Matrix<Scalar,10,1> minusVec;
//                minusVec.template head<3>() = imuPoseMinus.Twp.translation();
//                minusVec.template segment<4>(3) = imuPoseMinus.Twp.so3().unit_quaternion().coeffs();
//                minusVec.template tail<3>() = imuPoseMinus.V;
            const Vector6t dErrorMinus = log_decoupled(imuPoseMinus.Twp, Twb);
//                std::cout << "Pose minus: " << imuPoseMinus.Twp.matrix() << std::endl;
            const Vector3t vErrorMinus = imuPoseMinus.V - poseB.V;
            Jb_fd.col(ii).template head<6>() = (dErrorPlus - dErrorMinus)/(2*dEps);
            Jb_fd.col(ii).template tail<3>() = (vErrorPlus - vErrorMinus)/(2*dEps);
//                Jb_fd.col(ii) = (plusVec-minusVec)/(2*dEps);
        }


        std::cout << "J = [" << std::endl << res.dZ_dX1.format(cleanFmt) << "]" << std::endl;
        std::cout << "Jf = [" << std::endl << J_fd.format(cleanFmt) << "]" << std::endl;
        std::cout << "J-Jf = [" << std::endl << (res.dZ_dX1-J_fd).format(cleanFmt) << "] norm = " << (res.dZ_dX1-J_fd).norm() << std::endl;

        std::cout << "Jg = [" << std::endl << res.dZ_dG.format(cleanFmt) << "]" << std::endl;
        std::cout << "Jgf = [" << std::endl << Jg_fd.format(cleanFmt) << "]" << std::endl;
        std::cout << "Jg-Jgf = [" << std::endl << (res.dZ_dG-Jg_fd).format(cleanFmt) << "] norm = " << (res.dZ_dG-Jg_fd).norm() << std::endl;

        std::cout << "Jb = [" << std::endl << res.dZ_dB.format(cleanFmt)/*jb_q*/ << "]" << std::endl;
        std::cout << "Jbf = [" << std::endl << Jb_fd.format(cleanFmt) << "]" << std::endl;
        std::cout << "Jb-Jbf = [" << std::endl << (res.dZ_dB-Jb_fd).format(cleanFmt) << "] norm = " << (res.dZ_dB-Jb_fd).norm() << std::endl;

        // now that we have the deltas with subtracted initial velocity, transform and gravity, we can construct the jacobian
        m_Ri.template segment<6>(res.ResidualOffset) = log_decoupled(Twa*Tab,Twb.inverse());
        m_Ri.template segment<3>(res.ResidualOffset+6) = imuPose.V - poseB.V;
    }

    // get the sigma for robust norm calculation. This call is O(n) on average,
    // which is desirable over O(nlogn) sort
    if( m_vErrors.size() > 0 ){
        auto it = m_vErrors.begin()+std::floor(m_vErrors.size()/2);
        std::nth_element(m_vErrors.begin(),it,m_vErrors.end());
        const Scalar dSigma = sqrt(*it);
        // See "Parameter Estimation Techniques: A Tutorial with Application to Conic
        // Fitting" by Zhengyou Zhang. PP 26 defines this magic number:
        const Scalar c_huber = 1.2107*dSigma;

        // now go through the measurements and assign weights
        for( ProjectionResidual& res : m_vProjResiduals ){
            // calculate the huber norm weight for this measurement
            const Scalar e = res.Residual.norm();
            // this is square rooted as normally we use JtWJ and JtWr, therefore in order to multiply
            // the weight directly into the jacobian/residual we must use the square root
            res.W = sqrt(e > c_huber ? c_huber/e : 1.0);
            m_Rpr.template segment<ProjectionResidual::ResSize>(res.ResidualOffset) *= res.W;
        }
    }

    //TODO : The transpose insertions here are hideously expensive as they are not in order.
    // find a way around this.

    // here we sort the measurements and insert them per pose and per landmark, this will mean
    // each insert operation is O(1)
    Scalar dPortTime = Tic();
    for( Pose& pose : m_vPoses ){
        if( pose.IsActive ) {
            // sort the measurements by id so the sparse insert is O(1)
            std::sort(pose.ProjResiduals.begin(), pose.ProjResiduals.end());
            for( const int id: pose.ProjResiduals ) {
                const ProjectionResidual& res = m_vProjResiduals[id];
                // insert the jacobians into the sparse matrices
                m_Jpr.insert(res.ResidualId,pose.OptId).setZero().template block<2,6>(0,0) = res.dZ_dP * res.W;
                m_Jprt.insert(pose.OptId,res.ResidualId).setZero().template block<6,2>(0,0) = res.dZ_dP.transpose() * res.W;
            }

            // add the pose/pose constraints
            std::sort(pose.BinaryResiduals.begin(), pose.BinaryResiduals.end());
            for( const int id: pose.BinaryResiduals ) {
                const BinaryResidual& res = m_vBinaryResiduals[id];
                const Eigen::Matrix<Scalar,6,6>& dZ_dZ = res.PoseAId == pose.Id ? res.dZ_dX1 : res.dZ_dX2;
                m_Jpp.insert(res.ResidualId,pose.OptId).setZero().template block<6,6>(0,0) = dZ_dZ * res.W;
                m_Jppt.insert(pose.OptId,res.ResidualId).setZero().template block<6,6>(0,0) = dZ_dZ.transpose() * res.W;
            }

            // add the unary constraints
            std::sort(pose.UnaryResiduals.begin(), pose.UnaryResiduals.end());
            for( const int id: pose.UnaryResiduals ) {
                const UnaryResidual& res = m_vUnaryResiduals[id];
                m_Ju.insert(res.ResidualId,pose.OptId).setZero().template block<6,6>(0,0) = res.dZ_dX * res.W;
                m_Jut.insert(pose.OptId,res.ResidualId).setZero().template block<6,6>(0,0) = res.dZ_dX.transpose() * res.W;
            }

            std::sort(pose.ImuResiduals.begin(), pose.ImuResiduals.end());
            for( const int id: pose.ImuResiduals ) {
                const ImuResidual& res = m_vImuResiduals[id];
                const Eigen::Matrix<Scalar,9,9>& dZ_dZ = res.PoseAId == pose.Id ? res.dZ_dX1 : res.dZ_dX2;
                m_Ji.insert(res.ResidualId,pose.OptId).setZero().template block<9,9>(0,0) = dZ_dZ * res.W;
                m_Jit.insert(pose.OptId,res.ResidualId).setZero().template block<9,9>(0,0) = dZ_dZ.transpose() * res.W;
                m_Jki.template block<ImuResidual::ResSize,2>(res.ResidualId*ImuResidual::ResSize,0) = res.dZ_dG;
                m_Jki.template block<ImuResidual::ResSize,6>(res.ResidualId*ImuResidual::ResSize,2) = res.dZ_dB;
            }
        }
    }

    for( Landmark& lm : m_vLandmarks ){
        // sort the measurements by id so the sparse insert is O(1)
        std::sort(lm.ProjResiduals.begin(), lm.ProjResiduals.end());
        for( const int id: lm.ProjResiduals ) {
            const ProjectionResidual& res = m_vProjResiduals[id];
//                std::cout << "      Adding jacobian cell for measurement " << pMeas->MeasurementId << " in landmark " << pMeas->LandmarkId << std::endl;
            m_Jl.insert(res.ResidualId,lm.OptId) = res.dZ_dX * res.W;
            m_Jlt.insert(lm.OptId,res.ResidualId) = res.dZ_dX.transpose() * res.W;
        }
    }
    dPortionSparse += Toc(dPortTime);
    // std::cout << "Jacobian calculation took " << Toc(dTime) << " seconds. transfer time " << dPortionTransfer << "s jacobian time " <<
                 // dPortionJac << "s sparse time " << dPortionSparse << "s" << std::endl;
}

// specializations
template class BundleAdjuster<double, ba::NOT_USED,9>;
 template class BundleAdjuster<double, 1,6>;
// template class BundleAdjuster<double, 3,9>;


}


