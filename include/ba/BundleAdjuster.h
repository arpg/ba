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
    Eigen::Matrix<double,9,1> GetPoseDerivative(const Sophus::SE3d& tTwb,const Eigen::Vector3d& tV_w,
                                                   const Eigen::Vector3d& tG_w, const ImuMeasurement& zStart,
                                                   const ImuMeasurement& zEnd, const Eigen::Vector3d& vBg,
                                                   const Eigen::Vector3d& vBa, const double dt)
    {
        double alpha = (zEnd.Time - (zStart.Time+dt))/(zEnd.Time - zStart.Time);
        Eigen::Vector3d zb = zStart.W*alpha + zEnd.W*(1.0-alpha);
        Eigen::Vector3d za = zStart.A*alpha + zEnd.A*(1.0-alpha);

        Eigen::Matrix<double,9,1> deriv;
        //derivative of position is velocity
        deriv.head<3>() = tV_w;
        //deriv.template segment<3>(3) = Sophus::SO3Group<T>::vee(tTwb.so3().matrix()*Sophus::SO3Group<T>::hat(zb));
        deriv.template segment<3>(3) = tTwb.so3().Adj()*(zb+vBg);
        deriv.template segment<3>(6) = tTwb.so3()*(za+vBa) - tG_w;
        return deriv;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    ImuPose IntegrateImu(ImuPose& pose, const ImuMeasurement& zStart,
                     const ImuMeasurement& zEnd, const Eigen::Vector3d& vBg,
                     const Eigen::Vector3d vBa,const Eigen::Vector3d dG)
    {
        //construct the state matrix
        double h = zEnd.Time - zStart.Time;
        if(h == 0){
            return pose;
        }

        Sophus::SE3d aug_Twv = pose.Twp;
        Eigen::Vector3d aug_V = pose.V;
        Eigen::Matrix<double,9,1> k1 = GetPoseDerivative(aug_Twv,aug_V,dG,zStart,zEnd,vBg,vBa,0);

        aug_Twv.translation() += k1.head<3>()*h;
        const Sophus::SO3d Rv2v1(Sophus::SO3d::exp(k1.segment<3>(3)*h));
        aug_Twv.so3() = Rv2v1*pose.Twp.so3();
        // do euler integration for now
        aug_V += k1.tail<3>()*h;

        //and now output the state
        pose.Twp = aug_Twv;
        pose.V = aug_V;
        pose.W = k1.segment<3>(3);
        pose.Time = zEnd.Time;
//        pose.m_dW = currentPose.m_dW;
//        pose.m_dTime = zEnd.m_dTime;
        return pose;
    }

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
        m_vPoses[uPoseId].UnaryResiduals.push_back(&m_vUnaryResiduals.back());
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
        m_vPoses[uPoseAId].BinaryResiduals.push_back(&m_vBinaryResiduals.back());
        m_vPoses[uPoseBId].BinaryResiduals.push_back(&m_vBinaryResiduals.back());
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

        m_vLandmarks[uLandmarkId].ProjResiduals.push_back(&m_vProjResiduals.back());
        m_vPoses[uPoseId].ProjResiduals.push_back(&m_vProjResiduals.back());

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

        m_vPoses[uPoseAId].ImuResiduals.push_back(&m_vImuResiduals.back());
        return residual.ResidualId;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    void Solve(const unsigned int uMaxIter)
    {
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
                Eigen::SparseBlockAdd(U,Jprt_Jpr,U);

                Eigen::VectorXd Jprt_Rpr(uNumPoses*PoseSize);
                Eigen::SparseBlockVectorProductDenseResult(m_Jprt,m_R.segment(m_vProjResiduals.front().ResidualOffset,m_vProjResiduals.size()*2),Jprt_Rpr);
                bp += Jprt_Rpr;
            }

            // add the contribution from the binary terms if any
            if( m_vBinaryResiduals.size() > 0 ) {
                Eigen::SparseBlockMatrix< Eigen::Matrix<double,PoseSize,PoseSize> > Jppt_Jpp(uNumPoses,uNumPoses);
                Eigen::SparseBlockProduct(m_Jppt,m_Jpp,Jppt_Jpp);
                Eigen::SparseBlockAdd(U,Jppt_Jpp,U);

                Eigen::VectorXd Jppt_Rpp(uNumPoses*PoseSize);
                Eigen::SparseBlockVectorProductDenseResult(m_Jppt,m_R.segment(m_vBinaryResiduals.front().ResidualOffset,m_vBinaryResiduals.size()*6),Jppt_Rpp);
                bp += Jppt_Rpp;
            }

            // add the contribution from the unary terms if any
            if( m_vUnaryResiduals.size() > 0 ) {
                Eigen::SparseBlockMatrix< Eigen::Matrix<double,PoseSize,PoseSize> > Jut_Ju(uNumPoses,uNumPoses);
                Eigen::SparseBlockProduct(m_Jut,m_Ju,Jut_Ju);
                Eigen::SparseBlockAdd(U,Jut_Ju,U);

                Eigen::VectorXd Jut_Ru(uNumPoses*PoseSize);
                Eigen::SparseBlockVectorProductDenseResult(m_Jut,m_R.segment(m_vUnaryResiduals.front().ResidualOffset,m_vUnaryResiduals.size()*6),Jut_Ru);
                bp += Jut_Ru;
            }

            if( LmSize > 0 && uNumLm > 0) {
                bl.resize(uNumLm*LmSize);
                Eigen::SparseBlockVectorProductDenseResult(m_Jlt,m_R,bl);

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
                    // std::cout << "Pose delta for " << ii << " is " << delta_p.block<6,1>(m_vPoses[ii].OptId*6,0).transpose() << std::endl;
                    m_vPoses[ii].Twp *= Sophus::SE3d::exp(delta_p.block<6,1>(m_vPoses[ii].OptId*6,0));
                    // clear the vector of Tsw values as they will need to be recalculated
                    m_vPoses[ii].Tsw.clear();
                }
                // else{
                //  std::cout << " Pose " << ii << " is inactive." << std::endl;
                //  }
            }
            std::cout << "BA iteration " << kk <<  " error: " << m_R.norm() << std::endl;
        }

        // update the global position of the landmarks from the sensor position
        for (Landmark& lm : m_vLandmarks) {
            lm.Xw = MultHomogeneous(m_vPoses[lm.RefPoseId].GetTsw(lm.RefCamId,m_Rig).inverse(), lm.Xs);
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
        // resize as needed
        const unsigned int uNumPoses = m_uNumActivePoses;
        const unsigned int uNumLm = m_vLandmarks.size();
        const unsigned int uNumProjRes = m_vProjResiduals.size();
        const unsigned int uNumBinRes = m_vBinaryResiduals.size();
        const unsigned int uNumUnRes = m_vUnaryResiduals.size();
        const unsigned int uNumImuRes = m_vImuResiduals.size();

        m_Jpr.resize(uNumProjRes,uNumPoses);
        m_Jprt.resize(uNumPoses,uNumProjRes);

        m_Jpp.resize(uNumBinRes,uNumPoses);
        m_Jppt.resize(uNumPoses,uNumBinRes);

        m_Ju.resize(uNumUnRes,uNumPoses);
        m_Jut.resize(uNumPoses,uNumUnRes);

        m_Jl.resize(uNumProjRes,uNumLm);
        m_Jlt.resize(uNumLm,uNumProjRes);

        // calculate the total number of residuals
        m_R.resize(uNumProjRes*2+uNumBinRes*6+uNumUnRes*6+uNumImuRes*9,1);


        // these calls remove all the blocks, but KEEP allocated memory as long as the object is alive
        m_Jpr.setZero();
        m_Jprt.setZero();

        m_Jpp.setZero();
        m_Jppt.setZero();

        m_Jpp.setZero();
        m_Jppt.setZero();

        m_Ju.setZero();
        m_Jut.setZero();

        m_Jl.setZero();
        m_Jlt.setZero();

        m_R.setZero();

        // used to store errors for robust norm calculation
        m_vErrors.reserve(uNumProjRes);
        m_vErrors.clear();


        double dPortionTransfer = 0, dPortionJac = 0, dPortionSparse = 0;
        unsigned int uPrevBlockResId = 0;

        // set all jacobians
        double dTime = Tic();
        for( ProjectionResidual& res : m_vProjResiduals ){
            double dPortTime = Tic();
            // calculate measurement jacobians

            // Tsw = T_cv * T_vw
            Landmark& lm = m_vLandmarks[res.LandmarkId];
            Pose& pose = m_vPoses[res.PoseId];
            Pose& refPose = m_vPoses[lm.RefPoseId];
            lm.Xs = MultHomogeneous(refPose.GetTsw(lm.RefCamId,m_Rig) ,lm.Xw);
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
            res.ResidualOffset = res.ResidualOffset+uPrevBlockResId;
            m_R.segment<2>(res.ResidualOffset) = res.Residual;
        }
        uPrevBlockResId += m_vProjResiduals.size()*2;

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
            m_R.segment<6>(res.ResidualOffset+uPrevBlockResId) = (Twa*res.Tab*Twb.inverse()).log();
        }
        uPrevBlockResId += m_vBinaryResiduals.size()*6;

        for( UnaryResidual& res : m_vUnaryResiduals ){
            const Sophus::SE3d& Twp = m_vPoses[res.PoseId].Twp;
            res.dZ_dX = dLog_dX(Twp, res.Twp.inverse());

            m_R.segment<6>(res.ResidualOffset+uPrevBlockResId) = (Twp*res.Twp.inverse()).log();
        }
        uPrevBlockResId += m_vUnaryResiduals.size()*6;

        for( ImuResidual& res : m_vImuResiduals ){
            // set up the initial pose for the integration
            const Eigen::Vector3d gravity = ImuCalibration::GetGravityVector(m_Imu.G);
            const Eigen::Matrix<double,3,2> dGravity = ImuCalibration::dGravity_dDirection(m_Imu.G);
            const Pose& startPose = m_vPoses[res.PoseAId];
            double totalDt = 0;
            ImuPose imuPose(startPose.Twp,startPose.V,Eigen::Vector3d::Zero(),startPose.Time);
            ImuMeasurement* pPrevMeas = 0;
            res.Poses.clear();
            res.Poses.reserve(res.Measurements.size()+1);
            res.Poses.push_back(imuPose);

            // integrate forward in time, and retain all the poses
            for(ImuMeasurement& meas : res.Measurements){
                if(pPrevMeas != 0){
                    totalDt += meas.Time - pPrevMeas->Time;
                    imuPose = IntegrateImu(imuPose,*pPrevMeas,meas,m_Imu.Bg,m_Imu.Ba,gravity);
                    res.Poses.push_back(imuPose);
                }
                pPrevMeas = &meas;
            }
            const ImuPose& endPose = res.Poses.back();
            const Sophus::SE3d Tab = startPose.Twp.inverse()*endPose.Twp;
            const Sophus::SE3d& Twa = startPose.Twp;
            const Sophus::SE3d& Twb = endPose.Twp;

            // now given the poses, calculate the jacobians.
            // First subtract gravity, initial pose and velocity from the delta T and delta V
            Sophus::SE3d Tab_0 = endPose.Twp;
            Eigen::Vector3d Vab_0 = endPose.V;            
            Tab_0.translation() -=(-gravity*0.5*powi(totalDt,2) + startPose.V*totalDt);   // subtract starting velocity and gravity
            Tab_0 = startPose.Twp.inverse() * Tab_0;                                       // subtract starting pose
            // Augment the velocity delta by subtracting effects of gravity
            Vab_0 = endPose.V - startPose.V;
            Vab_0 += gravity*totalDt;
            // rotate the velocity delta so that it starts from orientation=Ident
            Vab_0 = startPose.Twp.so3().inverse() * Vab_0;

            // derivative with respect to the start pose
            res.dZ_dX1.setZero();
            res.dZ_dX2.setZero();
            res.dZ_dG.setZero();

            res.dZ_dX1.block<6,6>(0,0) = dLog_dX(Twa,Tab*Twb.inverse());
            res.dZ_dX1.block<3,3>(0,6) = Eigen::Matrix3d::Identity()*totalDt;
            for( int ii = 0; ii < 3 ; ++ii ){
                res.dZ_dX1.block<3,1>(6,3+ii) = Twa.so3().matrix() * Sophus::SO3d::generator(ii) * Vab_0;
            }
            res.dZ_dX1.block<3,3>(6,6) = Eigen::Matrix3d::Identity();
            // the - sign is here because of the exp(-x) within the log
            res.dZ_dX2.block<6,6>(0,0) = -dLog_dX(Twa*Tab,Twb.inverse());

            res.dZ_dG.block<3,2>(0,0) = -0.5*powi(totalDt,2)*Eigen::Matrix3d::Identity()*dGravity;
            res.dZ_dG.block<3,2>(6,0) = -totalDt*Eigen::Matrix3d::Identity()*dGravity;

            // now that we have the deltas with subtracted initial velocity, transform and gravity, we can construt the jacobian

            //            m_R.segment<9>(uMaxResId) = (Twp*res.Twp.inverse());
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
                m_R.segment<2>(res.ResidualOffset) *= res.W;
            }
        }

        // here we sort the measurements and insert them per pose and per landmark, this will mean
        // each insert operation is O(1)
        double dPortTime = Tic();
        for( Pose& pose : m_vPoses ){
            if( pose.IsActive ) {
                // sort the measurements by id so the sparse insert is O(1)
                std::sort(pose.ProjResiduals.begin(), pose.ProjResiduals.end(),
                    [](const ProjectionResidual * pA, const ProjectionResidual * pB) -> bool { return pA->ResidualId < pB->ResidualId;  });

                for( ProjectionResidual* pRes: pose.ProjResiduals ) {
                    // insert the jacobians into the sparse matrices
                    m_Jpr.insert(pRes->ResidualId,pose.OptId).template block<2,6>(0,0) = pRes->dZ_dP * pRes->W;
                    m_Jprt.insert(pose.OptId,pRes->ResidualId).template block<6,2>(0,0) = pRes->dZ_dP.transpose() * pRes->W;
                }

                // add the pose/pose constraints
                std::sort(pose.BinaryResiduals.begin(), pose.BinaryResiduals.end(),
                    [](const BinaryResidual * pA, const BinaryResidual * pB) -> bool { return pA->ResidualId < pB->ResidualId;  });
                for( BinaryResidual* pRes: pose.BinaryResiduals ) {
                    const Eigen::Matrix<double,6,6>& dZ_dZ = pRes->PoseAId == pose.Id ? pRes->dZ_dX1 : pRes->dZ_dX2;
                    m_Jpp.insert(pRes->ResidualId,pose.OptId).template block<6,6>(0,0) = dZ_dZ * pRes->W;
                    m_Jppt.insert(pose.OptId,pRes->ResidualId).template block<6,6>(0,0) = dZ_dZ.transpose() * pRes->W;
                }

                // add the unary constraints
                std::sort(pose.UnaryResiduals.begin(), pose.UnaryResiduals.end(),
                    [](const UnaryResidual * pA, const UnaryResidual * pB) -> bool { return pA->ResidualId < pB->ResidualId;  });
                for( UnaryResidual* pRes: pose.UnaryResiduals ) {
                    m_Ju.insert(pRes->ResidualId,pose.OptId).template block<6,6>(0,0) = pRes->dZ_dX * pRes->W;
                    m_Jut.insert(pose.OptId,pRes->ResidualId).template block<6,6>(0,0) = pRes->dZ_dX.transpose() * pRes->W;
                }
            }
        }

        for( Landmark& lm : m_vLandmarks ){
            // sort the measurements by id so the sparse insert is O(1)
            std::sort(lm.ProjResiduals.begin(), lm.ProjResiduals.end(),
                [](const ProjectionResidual * pA, const ProjectionResidual * pB) -> bool { return pA->ResidualId < pB->ResidualId; });

            for( ProjectionResidual* pRes: lm.ProjResiduals ) {
//                std::cout << "      Adding jacobian cell for measurement " << pMeas->MeasurementId << " in landmark " << pMeas->LandmarkId << std::endl;
                m_Jl.insert(pRes->ResidualId,lm.OptId) = pRes->dZ_dX * pRes->W;
                m_Jlt.insert(lm.OptId,pRes->ResidualId) = pRes->dZ_dX.transpose() * pRes->W;
            }
        }
        dPortionSparse += Toc(dPortTime);
        std::cout << "Jacobian calculation took " << Toc(dTime) << " seconds. transfer time " << dPortionTransfer << "s jacobian time " <<
                     dPortionJac << "s sparse time " << dPortionSparse << "s" << std::endl;
    }

    // reprojection jacobians
    Eigen::SparseBlockMatrix< Eigen::Matrix<double,2,PoseSize> > m_Jpr;
    Eigen::SparseBlockMatrix< Eigen::Matrix<double,PoseSize,2> > m_Jprt;

    // pose/pose jacobian for binary constraints
    Eigen::SparseBlockMatrix< Eigen::Matrix<double,PoseSize,PoseSize> > m_Jpp;
    Eigen::SparseBlockMatrix< Eigen::Matrix<double,PoseSize,PoseSize> > m_Jppt;

    // pose/pose jacobian for unary constraints
    Eigen::SparseBlockMatrix< Eigen::Matrix<double,PoseSize,PoseSize> > m_Ju;
    Eigen::SparseBlockMatrix< Eigen::Matrix<double,PoseSize,PoseSize> > m_Jut;

    // landmark jacobians
    Eigen::SparseBlockMatrix< Eigen::Matrix<double,2,LmSize> > m_Jl;
    Eigen::SparseBlockMatrix< Eigen::Matrix<double,LmSize,2> > m_Jlt;

    // imu jacobian
    Eigen::SparseBlockMatrix< Eigen::Matrix<double,PoseSize,PoseSize> > m_Ji;
    Eigen::SparseBlockMatrix< Eigen::Matrix<double,PoseSize,PoseSize> > m_Jit;

    // gravity jacobian
    Eigen::SparseBlockMatrix< Eigen::Matrix<double,2,1> > m_Jg;
    Eigen::SparseBlockMatrix< Eigen::Matrix<double,1,2> > m_Jgt;

    Eigen::VectorXd m_R;

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
