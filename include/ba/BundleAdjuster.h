#ifndef BUNDLEADUJSTER_H
#define BUNDLEADUJSTER_H

#include <sophus/se3.hpp>
#include <vector>
#include <calibu/Calibu.h>
#include <cholmod.h>
#include <Eigen/Sparse>
#include "SparseBlockMatrix.h"
#include "SparseBlockMatrixOps.h"
#include "Utils.h"
#include "Types.h"



namespace ba {

template< int LmSize, int PoseSize >
class BundleAdjuster
{
    typedef PoseT<LmSize> Pose;
    typedef LandmarkT<LmSize> Landmark;
    typedef ProjectionResidualT<LmSize> ProjectionResidual;
public:
    ///////////////////////////////////////////////////////////////////////////////////////////////
    BundleAdjuster() :
        m_Imu(Sophus::SE3d(),Eigen::Vector3d::Zero(),Eigen::Vector3d::Zero(),Eigen::Vector2d::Zero()) {}


    ///////////////////////////////////////////////////////////////////////////////////////////////
    void Init(const unsigned int uNumPoses,
              const unsigned int uNumMeasurements,
              const unsigned int uNumLandmarks = 0,
              const calibu::CameraRig *pRig = 0 )
    {
        // if LmSize == 0, there is no need for a camera rig or landmarks
        assert(pRig != 0 || LmSize == 0);
        assert(uNumLandmarks != 0 || LmSize == 0);

        m_uNumActivePoses = 0;
        m_uProjResidualOffset = 0;
        m_uBinaryResidualOffset = 0;
        m_uUnaryResidualOffset = 0;
        m_uImuResidualOffset = 0;
        if(pRig != 0){
            m_Rig = *pRig;
        }
        m_vLandmarks.reserve(uNumLandmarks);
        m_vProjResiduals.reserve(uNumMeasurements);
        m_vPoses.reserve(uNumPoses);

        // clear all arrays
        m_vPoses.clear();
        m_vProjResiduals.clear();
        m_vBinaryResiduals.clear();
        m_vUnaryResiduals.clear();
        m_vImuResiduals.clear();
        m_vLandmarks.clear();

    }    

    ///////////////////////////////////////////////////////////////////////////////////////////////
    unsigned int AddPose(const Sophus::SE3d& Twp, const bool bIsActive = true, const double dTime = -1)
    {
        Pose pose;
        pose.Time = dTime;
        pose.Twp = Twp;
        pose.IsActive = bIsActive;
        pose.Tsw.reserve(m_Rig.cameras.size());
        // assume equal distribution of measurements amongst poses
        pose.ProjResiduals.reserve(m_vProjResiduals.capacity()/m_vPoses.capacity());
        pose.Id = m_vPoses.size();
        if(bIsActive){
            pose.OptId = m_uNumActivePoses;
            m_uNumActivePoses++;
        }else{
            // the is active flag should be checked before reading this value, to see if the pose
            // is part of the optimization or not
            pose.OptId = 0;
        }

        m_vPoses.push_back(pose);        

        return pose.Id;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    unsigned int AddLandmark(const Eigen::Vector4d& Xw,const unsigned int uRefPoseId, const unsigned int uRefCamId = 0)
    {
        assert(uRefPoseId < m_vPoses.size());
        Landmark landmark;
        landmark.Xw = Xw;
        // assume equal distribution of measurements amongst landmarks
        landmark.ProjResiduals.reserve(m_vProjResiduals.capacity()/m_vLandmarks.capacity());
        landmark.OptId = m_vLandmarks.size();
        landmark.RefPoseId = uRefPoseId;
        landmark.RefCamId = uRefCamId;
        m_vLandmarks.push_back(landmark);
        return landmark.OptId;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    unsigned int AddUnaryConstraint(const unsigned int uPoseId,
                                    const Sophus::SE3d& Twp)
    {
        assert(uPoseId < m_vPoses.size());

        //now add this constraint to pose A
        UnaryResidual residual;
        residual.W = 1.0;
        residual.PoseId = uPoseId;
        residual.ResidualId = m_vUnaryResiduals.size();
        residual.ResidualOffset = m_uUnaryResidualOffset;
        residual.Twp = Twp;

        m_vUnaryResiduals.push_back(residual);
        m_uUnaryResidualOffset += 6;

        // we add this to both poses, as each one has a jacobian cell associated
        m_vPoses[uPoseId].UnaryResiduals.push_back(residual.ResidualId);
        return residual.ResidualId;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    unsigned int AddBinaryConstraint(const unsigned int uPoseAId,
                                     const unsigned int uPoseBId,
                                     const Sophus::SE3d& Tab)
    {
        assert(uPoseAId < m_vPoses.size());
        assert(uPoseBId < m_vPoses.size());

        //now add this constraint to pose A
        BinaryResidual residual;
        residual.W = 1.0;
        residual.PoseAId = uPoseAId;
        residual.PoseBId = uPoseBId;
        residual.ResidualId = m_vBinaryResiduals.size();
        residual.ResidualOffset = m_uBinaryResidualOffset;
        residual.Tab = Tab;

        m_vBinaryResiduals.push_back(residual);
        m_uBinaryResidualOffset += 6;

        // we add this to both poses, as each one has a jacobian cell associated
        m_vPoses[uPoseAId].BinaryResiduals.push_back(residual.ResidualId);
        m_vPoses[uPoseBId].BinaryResiduals.push_back(residual.ResidualId);
        return residual.ResidualId;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    unsigned int AddProjectionResidual(const Eigen::Vector2d z,
                                    const unsigned int uPoseId,
                                    const unsigned int uLandmarkId,
                                    const unsigned int uCameraId)
    {
        assert(uLandmarkId < m_vLandmarks.size());
        assert(uPoseId < m_vPoses.size());

        ProjectionResidual residual;
        residual.W = 1.0;
        residual.LandmarkId = uLandmarkId;
        residual.PoseId = uPoseId;
        residual.Z = z;
        residual.CameraId = uCameraId;
        residual.ResidualId = m_vProjResiduals.size();
        residual.ResidualOffset = m_uProjResidualOffset;

        m_vProjResiduals.push_back(residual);
        m_uProjResidualOffset += 2;

        m_vLandmarks[uLandmarkId].ProjResiduals.push_back(residual.ResidualId);
        m_vPoses[uPoseId].ProjResiduals.push_back(residual.ResidualId);

        return residual.ResidualId;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    unsigned int AddImuResidual(const unsigned int uPoseAId,
                                const unsigned int uPoseBId,
                                const std::vector<ba::ImuMeasurement>& vImuMeas)
    {
        assert(uPoseAId < m_vPoses.size());
        assert(uPoseBId < m_vPoses.size());
        // we must be using 9DOF poses for IMU residuals
        assert(PoseSize == 9);

        ImuResidual residual;
        residual.PoseAId = uPoseAId;
        residual.PoseBId = uPoseBId;
        residual.Measurements = vImuMeas;
        residual.ResidualId = m_vImuResiduals.size();
        residual.ResidualOffset = m_uImuResidualOffset;

        m_vImuResiduals.push_back(residual);
        m_uImuResidualOffset += 9;

        m_vPoses[uPoseAId].ImuResiduals.push_back(residual.ResidualId);
        return residual.ResidualId;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    void Solve(const unsigned int uMaxIter)
    {
        Eigen::IOFormat cleanFmt(2, 0, ", ", ";\n" , "" , "");
        // double dTime = Tic();
        // first build the jacobian and residual vector
        for( unsigned int kk = 0 ; kk < uMaxIter ; kk++){
            double dTime = Tic();
            _BuildProblem();
            std::cout << "Build problem took " << Toc(dTime) << " seconds." << std::endl;

            dTime = Tic();

            const unsigned int uNumPoses = m_uNumActivePoses;
            const unsigned int uNumLm = m_vLandmarks.size();
    //        const unsigned int uNumMeas = m_vMeasurements.size();
            // calculate bp and bl
            double dMatTime = Tic();
            Eigen::VectorXd bp(uNumPoses*PoseSize);
            Eigen::VectorXd bl;
            Eigen::SparseBlockMatrix< Eigen::Matrix<double,LmSize,LmSize> > V_inv(uNumLm,uNumLm);
            Eigen::VectorXd rhs_p(uNumPoses*PoseSize);
            Eigen::SparseBlockMatrix< Eigen::Matrix<double,LmSize,PoseSize> > Wt(uNumLm,uNumPoses);
            Eigen::MatrixXd S(uNumPoses*PoseSize,uNumPoses*PoseSize);
            std::cout << "  Rhs vector mult took " << Toc(dMatTime) << " seconds." << std::endl;

            dMatTime = Tic();

            // TODO: suboptimal, the matrices are symmetric. We should only multipl one half
            Eigen::SparseBlockMatrix< Eigen::Matrix<double,PoseSize,PoseSize> > U(uNumPoses,uNumPoses);
            U.setZero();
            bp.setZero();

            if( m_vProjResiduals.size() > 0 ){
                Eigen::SparseBlockMatrix< Eigen::Matrix<double,PoseSize,PoseSize> > Jprt_Jpr(uNumPoses,uNumPoses);
                Eigen::SparseBlockProduct(m_Jprt,m_Jpr,Jprt_Jpr);
                auto Temp = U;
                Eigen::SparseBlockAdd(Temp,Jprt_Jpr,U);

                Eigen::VectorXd Jprt_Rpr(uNumPoses*PoseSize);
                Eigen::SparseBlockVectorProductDenseResult(m_Jprt,m_Rpr,Jprt_Rpr);
                bp += Jprt_Rpr;
            }

            // add the contribution from the binary terms if any
            if( m_vBinaryResiduals.size() > 0 ) {
                Eigen::SparseBlockMatrix< Eigen::Matrix<double,PoseSize,PoseSize> > Jppt_Jpp(uNumPoses,uNumPoses);
                Eigen::SparseBlockProduct(m_Jppt,m_Jpp,Jppt_Jpp);
                auto Temp = U;
                Eigen::SparseBlockAdd(Temp,Jppt_Jpp,U);

                Eigen::VectorXd Jppt_Rpp(uNumPoses*PoseSize);
                Eigen::SparseBlockVectorProductDenseResult(m_Jppt,m_Rpp,Jppt_Rpp);
                bp += Jppt_Rpp;
            }

            // add the contribution from the unary terms if any
            if( m_vUnaryResiduals.size() > 0 ) {
                Eigen::SparseBlockMatrix< Eigen::Matrix<double,PoseSize,PoseSize> > Jut_Ju(uNumPoses,uNumPoses);
                Eigen::SparseBlockProduct(m_Jut,m_Ju,Jut_Ju);
                auto Temp = U;
                Eigen::SparseBlockAdd(Temp,Jut_Ju,U);

                Eigen::VectorXd Jut_Ru(uNumPoses*PoseSize);
                Eigen::SparseBlockVectorProductDenseResult(m_Jut,m_Ru,Jut_Ru);
                bp += Jut_Ru;

//                Eigen::LoadDenseFromSparse(U,S);
//                std::cout << "Dense S matrix is " << S.format(cleanFmt) << std::endl;
            }

            // add the contribution from the imu terms if any
            if( m_vImuResiduals.size() > 0 ) {
                Eigen::SparseBlockMatrix< Eigen::Matrix<double,PoseSize,PoseSize> > Jit_Ji(uNumPoses,uNumPoses);
                Eigen::SparseBlockProduct(m_Jit,m_Ji,Jit_Ji);
                auto Temp = U;
                Eigen::SparseBlockAdd(Temp,Jit_Ji,U);

                Eigen::VectorXd Jit_Ri(uNumPoses*PoseSize);
                Eigen::SparseBlockVectorProductDenseResult(m_Jit,m_Ri,Jit_Ri);
                bp += Jit_Ri;
//                Eigen::LoadDenseFromSparse(U,S);
//                std::cout << "Dense S matrix is " << S.format(cleanFmt) << std::endl;
            }

            if( LmSize > 0 && uNumLm > 0) {
                bl.resize(uNumLm*LmSize);
                Eigen::SparseBlockVectorProductDenseResult(m_Jlt,m_Rpr,bl);

                Eigen::SparseBlockMatrix< Eigen::Matrix<double,LmSize,LmSize> > V(uNumLm,uNumLm);
                Eigen::SparseBlockProduct(m_Jlt,m_Jl,V);

                // TODO this is really suboptimal, we should write a function to transpose the matrix
                Eigen::SparseBlockMatrix< Eigen::Matrix<double,PoseSize,LmSize> > W(uNumPoses,uNumLm);
                Eigen::SparseBlockProduct(m_Jprt,m_Jl,W);

                Eigen::SparseBlockProduct(m_Jlt,m_Jpr,Wt);

                std::cout << "  Outer produce took " << Toc(dMatTime) << " seconds." << std::endl;

                dMatTime = Tic();
                // calculate the inverse of the map hessian (it should be diagonal, unless a measurement is of more than
                // one landmark, which doesn't make sense)
                for(size_t ii = 0 ; ii < uNumLm ; ii++){
                    V_inv.coeffRef(ii,ii) = V.coeffRef(ii,ii).inverse();
                }
                std::cout << "  Inversion of V took " << Toc(dMatTime) << " seconds." << std::endl;

                dMatTime = Tic();
                // attempt to solve for the poses. W_V_inv is used later on, so we cache it
                Eigen::SparseBlockMatrix< Eigen::Matrix<double,PoseSize,LmSize> > W_V_inv(uNumPoses,uNumLm);
                Eigen::SparseBlockProduct(W,V_inv,W_V_inv);

                Eigen::SparseBlockMatrix< Eigen::Matrix<double,PoseSize,PoseSize> > WV_invWt(uNumPoses,uNumPoses);
                Eigen::SparseBlockProduct(W_V_inv,Wt,WV_invWt);

                // this in-place operation should be fine for subtraction
                Eigen::SparseBlockSubtractDenseResult(U,WV_invWt,S);

                // now form the rhs for the pose equations
                Eigen::VectorXd WV_inv_bl(uNumPoses*PoseSize);
                Eigen::SparseBlockVectorProductDenseResult(W_V_inv,bl,WV_inv_bl);

                rhs_p = bp - WV_inv_bl;
                std::cout << "  Rhs calculation took " << Toc(dMatTime) << " seconds." << std::endl;
            }else{
                Eigen::LoadDenseFromSparse(U,S);
                rhs_p = bp;
//                std::cout << "Dense S matrix is " << S.format(cleanFmt) << std::endl;
//                std::cout << "Dense rhs matrix is " << rhs_p.transpose().format(cleanFmt) << std::endl;
            }

            std::cout << "Setup took " << Toc(dTime) << " seconds." << std::endl;

            // now we have to solve for the pose constraints
            dTime = Tic();
            Eigen::VectorXd delta_p = uNumPoses == 0 ? Eigen::VectorXd() : S.ldlt().solve(rhs_p);            
            std::cout << "Cholesky solve of " << uNumPoses << " by " << uNumPoses << "matrix took " << Toc(dTime) << " seconds." << std::endl;

            if( uNumLm > 0) {
                dTime = Tic();
                Eigen::VectorXd delta_l;
                delta_l.resize(uNumLm*LmSize);
                Eigen::VectorXd Wt_delta_p;
                Wt_delta_p.resize(uNumLm*LmSize );
                Eigen::SparseBlockVectorProductDenseResult(Wt,delta_p,Wt_delta_p);
                Eigen::VectorXd rhs_l;
                rhs_l.resize(uNumLm*LmSize );
                rhs_l =  bl - Wt_delta_p;

                for(size_t ii = 0 ; ii < uNumLm ; ii++){
                    delta_l.block<LmSize,1>( ii*LmSize, 0 ).noalias() =  V_inv.coeff(ii,ii)*rhs_l.block<LmSize,1>(ii*LmSize,0);
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
//                     std::cout << "Pose delta for " << ii << " is " << delta_p.block<6,1>(m_vPoses[ii].OptId*6,0).transpose() << std::endl;
                    m_vPoses[ii].Twp *= Sophus::SE3d::exp(delta_p.block<6,1>(m_vPoses[ii].OptId*PoseSize,0));
                    // update the velocities if they are parametrized
                    if(PoseSize == 9){
                        m_vPoses[ii].V += delta_p.block<3,1>(m_vPoses[ii].OptId*PoseSize+6,0);
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
            lm.Xw = ba::MultHomogeneous(m_vPoses[lm.RefPoseId].GetTsw(lm.RefCamId,m_Rig).inverse(), lm.Xs);
//            return lm.Xw;
        }

            // std::cout << "Solve took " << Toc(dTime) << " seconds." << std::endl;
    }


    const ImuResidual& GetImuResidual(const unsigned int id) const { return m_vImuResiduals[id]; }
    const ImuCalibration& GetImuCalibration() const { return m_Imu; }
    const Sophus::SE3d& GetPose(const unsigned int id) const  { return m_vPoses[id].Twp; }
    // return the landmark in the world frame
    const Eigen::Vector4d& GetLandmark(const unsigned int id) const { return m_vLandmarks[id].Xw; }



private:
    void _BuildProblem()
    {
        Eigen::IOFormat cleanFmt(2, 0, ", ", ";\n" , "" , "");

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
//        m_Jki.setZero();
//        m_Jkit.setZero();
        m_Ri.setZero();

        m_Jl.setZero();
        m_Jlt.setZero();


        // used to store errors for robust norm calculation
        m_vErrors.reserve(uNumProjRes);
        m_vErrors.clear();


        double dPortionTransfer = 0, dPortionJac = 0, dPortionSparse = 0;

        // set all jacobians
        double dTime = Tic();
        for( ProjectionResidual& res : m_vProjResiduals ){
            double dPortTime = Tic();
            // calculate measurement jacobians

            // Tsw = T_cv * T_vw
            Landmark& lm = m_vLandmarks[res.LandmarkId];
            Pose& pose = m_vPoses[res.PoseId];
            Pose& refPose = m_vPoses[lm.RefPoseId];
            lm.Xs = ba::MultHomogeneous(refPose.GetTsw(lm.RefCamId,m_Rig) ,lm.Xw);
            const Sophus::SE3d parentTws = refPose.GetTsw(lm.RefCamId,m_Rig).inverse();

            const Eigen::Vector2d p = m_Rig.cameras[res.CameraId].camera.Transfer3D(pose.GetTsw(res.CameraId,m_Rig)*parentTws,
                                                                                    lm.Xs.template head<3>(),lm.Xs(3));
            res.Residual = res.Z - p;
//            std::cout << "Residual for meas " << meas.MeasurementId << " and landmark " << meas.LandmarkId << " with camera " << meas.CameraId << " is " << meas.Residual.transpose() << std::endl;

            // this array is used to calculate the robust norm
            m_vErrors.push_back(res.Residual.squaredNorm());

            dPortionTransfer += Toc(dPortTime);

            dPortTime = Tic();            
            const Eigen::Matrix<double,2,4> dTdP = m_Rig.cameras[res.CameraId].camera.dTransfer3D_dP(pose.GetTsw(res.CameraId,m_Rig)*parentTws,
                                                                                                     lm.Xs.template head<3>(),lm.Xs(3));
            res.dZ_dX = dTdP.block<2,LmSize>( 0, LmSize == 3 ? 0 : 3 );

            if( pose.IsActive ) {
                for(unsigned int ii=0; ii<6; ++ii){
                    const Eigen::Matrix<double,2,4> dTdP = m_Rig.cameras[res.CameraId].camera.dTransfer3D_dP(pose.GetTsw(res.CameraId,m_Rig),
                                                                                                             lm.Xw.template head<3>(),lm.Xw(3));
                    res.dZ_dP.template block<2,1>(0,ii) = dTdP * -Sophus::SE3::generator(ii) * lm.Xw;
                }
                //Eigen::Matrix<double,2,6> J_fd;
                //double dEps = 1e-6;
                //for(int ii = 0; ii < 6 ; ii++) {
                //    Eigen::Matrix<double,6,1> delta;
                //    delta.setZero();
                //    delta[ii] = dEps;
                //    Sophus::SE3d Tsw = (pose.Twp*Sophus::SE3d::exp(delta)*m_Rig.cameras[meas.CameraId].T_wc).inverse();
                //    const Eigen::Vector2d pPlus = m_Rig.cameras[meas.CameraId].camera.Transfer3D(Tsw,Xw.head(3),Xw[3]);
                //    delta[ii] = -dEps;
                //    Tsw = (pose.Twp*Sophus::SE3d::exp(delta)*m_Rig.cameras[meas.CameraId].T_wc).inverse();
                //    const Eigen::Vector2d pMinus = m_Rig.cameras[meas.CameraId].camera.Transfer3D(Tsw,Xw.head(3),Xw[3]);
                //    J_fd.col(ii) = (pPlus-pMinus)/(2*dEps);
                //}
                //std::cout << "J:" << meas.dZ_dP << std::endl;
                //std::cout << "J_fd:" << J_fd << std::endl;
            }


            dPortionJac += Toc(dPortTime);

            // set the residual in m_R which is dense
            m_Rpr.segment<ProjectionResidual::ResSize>(res.ResidualOffset) = res.Residual;
        }

        // build binary residual jacobians
        for( BinaryResidual& res : m_vBinaryResiduals ){
            const Sophus::SE3d& Twa = m_vPoses[res.PoseAId].Twp;
            const Sophus::SE3d& Twb = m_vPoses[res.PoseBId].Twp;
            res.dZ_dX1 = dLog_dX(Twa, res.Tab * Twb.inverse());
            // the negative sign here is because exp(x) is inside the inverse when we invert (Twb*exp(x)).inverse
            res.dZ_dX2 = -dLog_dX(Twa * res.Tab , Twb.inverse());

            // finite difference checking
//            Eigen::Matrix<double,6,6> J_fd;
//            double dEps = 1e-10;
//            for(int ii = 0; ii < 6 ; ii++) {
//                Eigen::Matrix<double,6,1> delta;
//                delta.setZero();
//                delta[ii] = dEps;
//                const Eigen::Vector6d pPlus = Sophus::SE3d::log(Twa*Sophus::SE3d::exp(delta) * res.Tab * Twb.inverse());
//                delta[ii] = -dEps;
//                const Eigen::Vector6d pMinus = Sophus::SE3d::log(Twa*Sophus::SE3d::exp(delta) * res.Tab * Twb.inverse());
//                J_fd.col(ii) = (pPlus-pMinus)/(2*dEps);
//            }
//            std::cout << "J:" << res.dZ_dX1 << std::endl;
//            std::cout << "J_fd:" << J_fd << std::endl;
            m_Rpp.segment<BinaryResidual::ResSize>(res.ResidualOffset) = (Twa*res.Tab*Twb.inverse()).log();
        }

        for( UnaryResidual& res : m_vUnaryResiduals ){
            const Sophus::SE3d& Twp = m_vPoses[res.PoseId].Twp;
            res.dZ_dX = dLog_dX(Twp, res.Twp.inverse());

            m_Ru.segment<UnaryResidual::ResSize>(res.ResidualOffset) = (Twp*res.Twp.inverse()).log();
        }

        for( ImuResidual& res : m_vImuResiduals ){
            // set up the initial pose for the integration
            const Eigen::Vector3d gravity = ImuCalibration::GetGravityVector(m_Imu.G);
            const Eigen::Matrix<double,3,2> dGravity = ImuCalibration::dGravity_dDirection(m_Imu.G);
            const Pose& poseA = m_vPoses[res.PoseAId];
            const Pose& poseB = m_vPoses[res.PoseBId];


//            ImuPose imuPose(poseA.Twp,poseA.V,Eigen::Vector3d::Zero(),poseA.Time);
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

            Eigen::Matrix<double,10,6> jb_q;
            ImuPose imuPose = ImuResidual::IntegrateResidual(poseA,res.Measurements,m_Imu.Bg,m_Imu.Ba,gravity,res.Poses,&jb_q   );
            double totalDt = res.Measurements.back().Time - res.Measurements.front().Time;
            const Sophus::SE3d Tab = poseA.Twp.inverse()*imuPose.Twp;
            const Sophus::SE3d& Twa = poseA.Twp;
            const Sophus::SE3d& Twb = poseB.Twp;

            // now given the poses, calculate the jacobians.
            // First subtract gravity, initial pose and velocity from the delta T and delta V
            Sophus::SE3d Tab_0 = imuPose.Twp;
            Tab_0.translation() -=(-gravity*0.5*powi(totalDt,2) + poseA.V*totalDt);   // subtract starting velocity and gravity
            Tab_0 = poseA.Twp.inverse() * Tab_0;                                       // subtract starting pose
            // Augment the velocity delta by subtracting effects of gravity
            Eigen::Vector3d Vab_0 = imuPose.V - poseA.V;
            Vab_0 += gravity*totalDt;
            // rotate the velocity delta so that it starts from orientation=Ident
            Vab_0 = poseA.Twp.so3().inverse() * Vab_0;

            // derivative with respect to the start pose
            res.dZ_dX1.setZero();
            res.dZ_dX2.setZero();
            res.dZ_dG.setZero();
            res.dZ_dB.setZero();

            Sophus::SE3d Tstar(Sophus::SO3d(),(poseA.V*totalDt - 0.5*gravity*powi(totalDt,2)));

            // calculate the derivative of the lie log with respect to the tangent plane at Twa
            const Eigen::Matrix<double,6,6> dLog  = dLog_dX(Tstar*Twa,Tab_0*Twb.inverse());
            const Eigen::Matrix<double,6,6> dLog_b  = dLog_dX(imuPose.Twp,Twb.inverse());

            // now add the log jacobians to the bias jacobian terms
            res.dZ_dB.block<3,6>(3,0) = dLog_dq((imuPose.Twp.so3() * Twb.so3().inverse()).unit_quaternion()) *
                                            dq1q2_dq1(Twb.so3().inverse().unit_quaternion()) * jb_q.block<4,6>(3,0);

//            const Sophus::SE3d se3 = imuPose.Twp*Twb.inverse();
//            Eigen::Vector3d upsilon_omega;
//            double theta;
//            upsilon_omega = Sophus::SO3d::logAndTheta(se3.so3(), &theta);

//              const Eigen::Matrix3d Omega = Sophus::SO3d::hat(upsilon_omega.template tail<3>());
//              const Eigen::Matrix3d V_inv =
//                  ( Eigen::Matrix3d::Identity() - 0.5*Omega + ( 1 - theta/(2*tan(theta/2))) / (theta*theta)*(Omega*Omega) );
//            std::cout << "Vinv " << std::endl << V_inv << std::endl;
//            upsilon_omega = V_inv*se3.translation();

//            Eigen::Matrix<double,6,6> temp;
//            temp.block<3,6>(0,0) = V_inv * jb_q.block<3,6>(0,0);
//            temp.block<3,6>(3,0) =dLog_dq((imuPose.Twp.so3() * Twb.so3().inverse()).unit_quaternion()) *
//                                    dq1q2_dq1(Twb.so3().inverse().unit_quaternion()) * jb_q.block<4,6>(3,0);
            Eigen::Matrix<double,6,7> dLog_dqx;
            for(int ii = 0 ; ii < 7 ; ii++){
                double dEps = 1e-9;
                Eigen::Matrix<double,7,1> epsVec = Eigen::Matrix<double,7,1>::Zero();
                epsVec[ii] += dEps;
                Sophus::SE3d Twp_imu_plus = imuPose.Twp;
                Twp_imu_plus.translation() += epsVec.head<3>();
                const Eigen::Vector4d coeffsPlus = Twp_imu_plus.unit_quaternion().coeffs() + epsVec.tail<4>();
                memcpy(Twp_imu_plus.so3().data(),coeffsPlus.data(),sizeof(double)*4);
                const Eigen::Vector6d resPlus = (Twp_imu_plus * Twb.inverse()).log();

                epsVec[ii] -= 2*dEps;
                Sophus::SE3d Twp_imu_minus = imuPose.Twp;
                Twp_imu_minus.translation() += epsVec.head<3>();
                const Eigen::Vector4d coeffsMinus = Twp_imu_minus.unit_quaternion().coeffs() + epsVec.tail<4>();
                memcpy(Twp_imu_minus.so3().data(),coeffsMinus.data(),sizeof(double)*4);
                const Eigen::Vector6d resMinus = (Twp_imu_minus * Twb.inverse()).log();
                dLog_dqx.col(ii) = (resPlus-resMinus)/(2*dEps);
            }
            std::cout << "dLog_dqx = " << std::endl << dLog_dqx << std::endl;
            res.dZ_dB.block<3,6>(0,0) = (dLog_dqx*jb_q.block<7,6>(0,0)).block<3,6>(0,0);
//            res.dZ_dB.block<3,6>(0,0) = dLog_b.block<3,3>(0,0) * imuPose.Twp.so3().inverse().matrix() * jb_q.block<3,6>(0,0) +
//                                        dLog_dq((imuPose.Twp.so3() * Twb.so3().inverse()).unit_quaternion()) * jb_q.block<4,6>(3,0);
            res.dZ_dB.block<3,6>(6,0) = jb_q.block<3,6>(7,0);

            res.dZ_dX1.block<6,6>(0,0) = dLog;
            // Twa^-1 is multiplied here as we need the velocity derivative in the frame of pose A, as the log is taken from this frame
            res.dZ_dX1.block<3,3>(0,6) = dLog.block<3,3>(0,0) * Twa.so3().inverse().matrix() * Eigen::Matrix3d::Identity()*totalDt;
            for( int ii = 0; ii < 3 ; ++ii ){
                res.dZ_dX1.block<3,1>(6,3+ii) = Twa.so3().matrix() * Sophus::SO3d::generator(ii) * Vab_0;
            }
            res.dZ_dX1.block<3,3>(6,6) = Eigen::Matrix3d::Identity();
            // the - sign is here because of the exp(-x) within the log
            res.dZ_dX2.block<6,6>(0,0) = -dLog_dX(imuPose.Twp,Twb.inverse());//-dLog_dX(Twa*Tab,Twb.inverse());
            res.dZ_dX2.block<3,3>(6,6) = -Eigen::Matrix3d::Identity();

            res.dZ_dG.block<3,2>(0,0) = dLog.block<3,3>(0,0) * Twa.so3().inverse().matrix() *
                                        -0.5*powi(totalDt,2)*Eigen::Matrix3d::Identity()*dGravity;
            res.dZ_dG.block<3,2>(6,0) = -totalDt*Eigen::Matrix3d::Identity()*dGravity;

            double dEps = 1e-12;
//            Eigen::Matrix<double,6,6> Jlog;
//            for(int ii = 0 ; ii < 6 ; ii++){
//                Eigen::Vector6d eps = Eigen::Vector6d::Zero();
//                eps[ii] += dEps;
//                const Eigen::Vector6d resPlus = Sophus::SE3d::log(Twa*Sophus::SE3d::exp(eps)*Tab*Twb.inverse());
//                eps[ii] -= 2*dEps;
//                const Eigen::Vector6d resMinus = Sophus::SE3d::log(Twa*Sophus::SE3d::exp(eps)*Tab*Twb.inverse());
//                Jlog.col(ii) = (resPlus-resMinus)/(2*dEps);
//            }

//            std::cout << "Jlog = [" << dLog_dX(Twa,Tab*Twb.inverse()) << "]" << std::endl;
//            std::cout << "Jlogf = [" << Jlog.format(cleanFmt) << "]" << std::endl;

            // verify using finite differences
            Eigen::Matrix<double,9,9> J_fd;
            J_fd.setZero();
            Eigen::Matrix<double,9,2> Jg_fd;
            Jg_fd.setZero();
            Eigen::Matrix<double,9,6> Jb_fd;
            Jb_fd.setZero();

            Eigen::Matrix<double,9,9>  dRi_dx2_fd;
            for(int ii = 0; ii < 9 ; ii++){
                Eigen::Matrix<double,9,1> epsVec = Eigen::Matrix<double,9,1>::Zero();
                epsVec[ii] += dEps;
                ImuPose y0_eps(poseB.Twp,poseB.V, Eigen::Vector3d::Zero(),0);
                y0_eps.Twp = y0_eps.Twp * Sophus::SE3d::exp(epsVec.head<6>());
                y0_eps.V += epsVec.tail<3>();
                Eigen::Matrix<double,9,1> r_plus;
                r_plus.head<6>() = (imuPose.Twp*y0_eps.Twp.inverse()).log();
                r_plus.tail<3>() = imuPose.V - y0_eps.V;



                epsVec[ii] -= 2*dEps;
                y0_eps = ImuPose(poseB.Twp,poseB.V, Eigen::Vector3d::Zero(),0);;
                y0_eps.Twp = y0_eps.Twp * Sophus::SE3d::exp(epsVec.head<6>());
                y0_eps.V += epsVec.tail<3>();
                Eigen::Matrix<double,9,1> r_minus;
                r_minus.head<6>() = (imuPose.Twp*y0_eps.Twp.inverse()).log();
                r_minus.tail<3>() = imuPose.V - y0_eps.V;

                dRi_dx2_fd.col(ii) = (r_plus-r_minus)/(2*dEps);
            }
            std::cout << "res.dZ_dX2= " << std::endl << res.dZ_dX2 << std::endl;
            std::cout << "dRi_dx2_fd = " << std::endl <<  dRi_dx2_fd << std::endl;
            std::cout << "res.dZ_dX2-dRi_dx2_fd = " << std::endl << res.dZ_dX2-dRi_dx2_fd << "norm: " << (res.dZ_dX2-dRi_dx2_fd).norm() <<  std::endl;


            for(int ii = 0 ; ii < 6 ; ii++){
                Eigen::Vector6d eps = Eigen::Vector6d::Zero();
                eps[ii] += dEps;
                Pose poseEps = poseA;
                poseEps.Twp = poseEps.Twp * Sophus::SE3d::exp(eps);
                std::vector<ImuPose> poses;
                const ImuPose imuPosePlus = ImuResidual::IntegrateResidual(poseEps,res.Measurements,m_Imu.Bg,m_Imu.Ba,gravity,poses);
                const Eigen::Vector6d dErrorPlus = (imuPosePlus.Twp * Twb.inverse()).log();
                const Eigen::Vector3d vErrorPlus = imuPosePlus.V - poseB.V;
                eps[ii] -= 2*dEps;
                poseEps = poseA;
                poseEps.Twp = poseEps.Twp * Sophus::SE3d::exp(eps);
                poses.clear();
                const ImuPose imuPoseMinus = ImuResidual::IntegrateResidual(poseEps,res.Measurements,m_Imu.Bg,m_Imu.Ba,gravity,poses);
                const Eigen::Vector6d dErrorMinus = (imuPoseMinus.Twp * Twb.inverse()).log();
                const Eigen::Vector3d vErrorMinus = imuPoseMinus.V - poseB.V;
                J_fd.col(ii).head<6>() = (dErrorPlus - dErrorMinus)/(2*dEps);
                J_fd.col(ii).tail<3>() = (vErrorPlus - vErrorMinus)/(2*dEps);
            }

            for(int ii = 0 ; ii < 3 ; ii++){
                Eigen::Vector3d eps = Eigen::Vector3d::Zero();
                eps[ii] += dEps;
                Pose poseEps = poseA;
                poseEps.V += eps;
                std::vector<ImuPose> poses;
                const ImuPose imuPosePlus = ImuResidual::IntegrateResidual(poseEps,res.Measurements,m_Imu.Bg,m_Imu.Ba,gravity,poses);
                const Eigen::Vector6d dErrorPlus = (imuPosePlus.Twp * Twb.inverse()).log();
//                std::cout << "Pose plus: " << imuPosePlus.Twp.matrix() << std::endl;
                const Eigen::Vector3d vErrorPlus = imuPosePlus.V - poseB.V;
                eps[ii] -= 2*dEps;
                poseEps = poseA;
                poseEps.V += eps;
                poses.clear();
                const ImuPose imuPoseMinus = ImuResidual::IntegrateResidual(poseEps,res.Measurements,m_Imu.Bg,m_Imu.Ba,gravity,poses);
                const Eigen::Vector6d dErrorMinus = (imuPoseMinus.Twp * Twb.inverse()).log();
//                std::cout << "Pose minus: " << imuPoseMinus.Twp.matrix() << std::endl;
                const Eigen::Vector3d vErrorMinus = imuPoseMinus.V - poseB.V;
                J_fd.col(ii+6).head<6>() = (dErrorPlus - dErrorMinus)/(2*dEps);
                J_fd.col(ii+6).tail<3>() = (vErrorPlus - vErrorMinus)/(2*dEps);
            }

            for(int ii = 0 ; ii < 2 ; ii++){
                Eigen::Vector2d eps = Eigen::Vector2d::Zero();
                eps[ii] += dEps;
                std::vector<ImuPose> poses;
                const ImuPose imuPosePlus = ImuResidual::IntegrateResidual(poseA,res.Measurements,m_Imu.Bg,m_Imu.Ba,m_Imu.GetGravityVector((Eigen::Vector2d)(m_Imu.G+eps)),poses);
                const Eigen::Vector6d dErrorPlus = (imuPosePlus.Twp * Twb.inverse()).log();
//                std::cout << "Pose plus: " << imuPosePlus.Twp.matrix() << std::endl;
                const Eigen::Vector3d vErrorPlus = imuPosePlus.V - poseB.V;
                eps[ii] -= 2*dEps;
                poses.clear();
                const ImuPose imuPoseMinus = ImuResidual::IntegrateResidual(poseA,res.Measurements,m_Imu.Bg,m_Imu.Ba,m_Imu.GetGravityVector((Eigen::Vector2d)(m_Imu.G+eps)),poses);
                const Eigen::Vector6d dErrorMinus = (imuPoseMinus.Twp * Twb.inverse()).log();
//                std::cout << "Pose minus: " << imuPoseMinus.Twp.matrix() << std::endl;
                const Eigen::Vector3d vErrorMinus = imuPoseMinus.V - poseB.V;
                Jg_fd.col(ii).head<6>() = (dErrorPlus - dErrorMinus)/(2*dEps);
                Jg_fd.col(ii).tail<3>() = (vErrorPlus - vErrorMinus)/(2*dEps);
            }

            Eigen::Vector6d biasVec;
            biasVec.head<3>() = m_Imu.Bg;
            biasVec.tail<3>() = m_Imu.Ba;
            for(int ii = 0 ; ii < 6 ; ii++){
                Eigen::Vector6d eps = Eigen::Vector6d::Zero();
                eps[ii] += dEps;
                std::vector<ImuPose> poses;
                const Eigen::Vector6d plusBiases = biasVec + eps;
                const ImuPose imuPosePlus = ImuResidual::IntegrateResidual(poseA,res.Measurements,plusBiases.head<3>(),plusBiases.tail<3>(),gravity,poses);
//                Eigen::Matrix<double,10,1> plusVec;
//                plusVec.head<3>() = imuPosePlus.Twp.translation();
//                plusVec.segment<4>(3) = imuPosePlus.Twp.so3().unit_quaternion().coeffs();
//                plusVec.tail<3>() = imuPosePlus.V;
                const Eigen::Vector6d dErrorPlus = (imuPosePlus.Twp * Twb.inverse()).log();
//                std::cout << "Pose plus: " << imuPosePlus.Twp.matrix() << std::endl;
                const Eigen::Vector3d vErrorPlus = imuPosePlus.V - poseB.V;

                eps[ii] -= 2*dEps;
                const Eigen::Vector6d minusBiases = biasVec + eps;
                poses.clear();
                const ImuPose imuPoseMinus = ImuResidual::IntegrateResidual(poseA,res.Measurements,minusBiases.head<3>(),minusBiases.tail<3>(),gravity,poses);
//                Eigen::Matrix<double,10,1> minusVec;
//                minusVec.head<3>() = imuPoseMinus.Twp.translation();
//                minusVec.segment<4>(3) = imuPoseMinus.Twp.so3().unit_quaternion().coeffs();
//                minusVec.tail<3>() = imuPoseMinus.V;
                const Eigen::Vector6d dErrorMinus = (imuPoseMinus.Twp * Twb.inverse()).log();
//                std::cout << "Pose minus: " << imuPoseMinus.Twp.matrix() << std::endl;
                const Eigen::Vector3d vErrorMinus = imuPoseMinus.V - poseB.V;
                Jb_fd.col(ii).head<6>() = (dErrorPlus - dErrorMinus)/(2*dEps);
                Jb_fd.col(ii).tail<3>() = (vErrorPlus - vErrorMinus)/(2*dEps);
//                Jb_fd.col(ii) = (plusVec-minusVec)/(2*dEps);
            }


            std::cout << "J = [" << std::endl << res.dZ_dX1 << "]" << std::endl;
            std::cout << "Jf = [" << std::endl << J_fd << "]" << std::endl;
            std::cout << "J-Jf = [" << std::endl << res.dZ_dX1-J_fd << "] norm = " << (res.dZ_dX1-J_fd).norm() << std::endl;

            std::cout << "Jg = [" << std::endl << res.dZ_dG << "]" << std::endl;
            std::cout << "Jgf = [" << std::endl << Jg_fd << "]" << std::endl;
            std::cout << "Jg-Jgf = [" << std::endl << res.dZ_dG-Jg_fd << "] norm = " << (res.dZ_dG-Jg_fd).norm() << std::endl;

            std::cout << "Jb = [" << std::endl << res.dZ_dB/*jb_q*/ << "]" << std::endl;
            std::cout << "Jbf = [" << std::endl << Jb_fd << "]" << std::endl;
            std::cout << "Jb-Jbf = [" << std::endl << res.dZ_dB-Jb_fd << "] norm = " << (res.dZ_dB-Jb_fd).norm() << std::endl;

            // now that we have the deltas with subtracted initial velocity, transform and gravity, we can construct the jacobian
            m_Ri.segment<6>(res.ResidualOffset) = (Twa*Tab,Twb.inverse()).log();
            m_Ri.segment<3>(res.ResidualOffset+6) = imuPose.V - poseB.V;
        }

        // get the sigma for robust norm calculation. This call is O(n) on average,
        // which is desirable over O(nlogn) sort
        if( m_vErrors.size() > 0 ){
            auto it = m_vErrors.begin()+std::floor(m_vErrors.size()/2);
            std::nth_element(m_vErrors.begin(),it,m_vErrors.end());
            const double dSigma = sqrt(*it);
            // See "Parameter Estimation Techniques: A Tutorial with Application to Conic
            // Fitting" by Zhengyou Zhang. PP 26 defines this magic number:
            const double c_huber = 1.2107*dSigma;

            // now go through the measurements and assign weights
            for( ProjectionResidual& res : m_vProjResiduals ){
                // calculate the huber norm weight for this measurement
                const double e = res.Residual.norm();
                // this is square rooted as normally we use JtWJ and JtWr, therefore in order to multiply
                // the weight directly into the jacobian/residual we must use the square root
                res.W = sqrt(e > c_huber ? c_huber/e : 1.0);
                m_Rpr.segment<ProjectionResidual::ResSize>(res.ResidualOffset) *= res.W;
            }
        }

        //TODO : The transpose insertions here are hideously expensive as they are not in order.
        // find a way around this.

        // here we sort the measurements and insert them per pose and per landmark, this will mean
        // each insert operation is O(1)
        double dPortTime = Tic();
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
                    const Eigen::Matrix<double,6,6>& dZ_dZ = res.PoseAId == pose.Id ? res.dZ_dX1 : res.dZ_dX2;
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
                    const Eigen::Matrix<double,9,9>& dZ_dZ = res.PoseAId == pose.Id ? res.dZ_dX1 : res.dZ_dX2;
                    m_Ji.insert(res.ResidualId,pose.OptId).setZero().template block<9,9>(0,0) = dZ_dZ * res.W;
                    m_Jit.insert(pose.OptId,res.ResidualId).setZero().template block<9,9>(0,0) = dZ_dZ.transpose() * res.W;
                    m_Jki.block<ImuResidual::ResSize,2>(res.ResidualId*ImuResidual::ResSize,0) = res.dZ_dG;
                    m_Jki.block<ImuResidual::ResSize,6>(res.ResidualId*ImuResidual::ResSize,2) = res.dZ_dB;
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
        std::cout << "Jacobian calculation took " << Toc(dTime) << " seconds. transfer time " << dPortionTransfer << "s jacobian time " <<
                     dPortionJac << "s sparse time " << dPortionSparse << "s" << std::endl;
    }

    // reprojection jacobians and residual
    Eigen::SparseBlockMatrix< Eigen::Matrix<double,ProjectionResidual::ResSize,PoseSize> > m_Jpr;
    Eigen::SparseBlockMatrix< Eigen::Matrix<double,PoseSize,ProjectionResidual::ResSize> > m_Jprt;
    // landmark jacobians
    Eigen::SparseBlockMatrix< Eigen::Matrix<double,ProjectionResidual::ResSize,LmSize> > m_Jl;
    Eigen::SparseBlockMatrix< Eigen::Matrix<double,LmSize,ProjectionResidual::ResSize> > m_Jlt;
    Eigen::VectorXd m_Rpr;

    // pose/pose jacobian for binary constraints
    Eigen::SparseBlockMatrix< Eigen::Matrix<double,BinaryResidual::ResSize,PoseSize> > m_Jpp;
    Eigen::SparseBlockMatrix< Eigen::Matrix<double,PoseSize,BinaryResidual::ResSize> > m_Jppt;
    Eigen::VectorXd m_Rpp;

    // pose/pose jacobian for unary constraints
    Eigen::SparseBlockMatrix< Eigen::Matrix<double,UnaryResidual::ResSize,PoseSize> > m_Ju;
    Eigen::SparseBlockMatrix< Eigen::Matrix<double,PoseSize,UnaryResidual::ResSize> > m_Jut;
    Eigen::VectorXd m_Ru;

    // imu jacobian
    Eigen::SparseBlockMatrix< Eigen::Matrix<double,ImuResidual::ResSize,PoseSize> > m_Ji;
    Eigen::SparseBlockMatrix< Eigen::Matrix<double,PoseSize,ImuResidual::ResSize> > m_Jit;

    // jacobian reserved for biases, gravity, etc
    // 2 for gravity, 6 for biases
    Eigen::MatrixXd m_Jki;
    Eigen::VectorXd m_Ri;



    unsigned int m_uNumActivePoses;
    unsigned int m_uBinaryResidualOffset;
    unsigned int m_uUnaryResidualOffset;
    unsigned int m_uProjResidualOffset;
    unsigned int m_uImuResidualOffset;
    calibu::CameraRig m_Rig;
    std::vector<Pose> m_vPoses;
    std::vector<Landmark> m_vLandmarks;
    std::vector<ProjectionResidual > m_vProjResiduals;
    std::vector<BinaryResidual> m_vBinaryResiduals;
    std::vector<UnaryResidual> m_vUnaryResiduals;
    std::vector<ImuResidual> m_vImuResiduals;
    std::vector<double> m_vErrors;

    ImuCalibration m_Imu;
};

static const int NOT_USED = 0;

// typedefs for convenience
typedef BundleAdjuster<ba::NOT_USED,9> GlobalInertialBundleAdjuster;
typedef BundleAdjuster<1,9> InverseDepthVisualInertialBundleAdjuster;
typedef BundleAdjuster<3,9> VisualInertialBundleAdjuster;

}





#endif // BUNDLEADUJSTER_H
