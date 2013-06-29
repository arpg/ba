#ifndef BUNDLEADUJSTER_H
#define BUNDLEADUJSTER_H

#include <sophus/se3.hpp>
#include <vector>
#include <calibu/Calibu.h>
#include <cholmod.h>
#include <Eigen/Sparse>
#include "SparseBlockMatrix.h"
#include "SparseBlockMatrixOps.h"
#include <sys/time.h>
#include <time.h>
#include "Utils.h"
#include "InterpolationBuffer.h"

inline double Tic() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec  + 1e-6 * (tv.tv_usec);
}

inline double Toc(double dTic) {
    return Tic() - dTic;
}

namespace Eigen
{
    typedef Matrix<double,6,1> Vector6d;
    typedef Matrix<double,9,1> Vector9d;
}

namespace Sophus
{
    // this function implements d vee(log(A * exp(x) * B) ) / dx , which is in R^{6x6}
    inline Eigen::Matrix<double,6,6> dLog_dX(const Sophus::SE3d& A,const Sophus::SE3d& B)
    {
        const Eigen::Vector6d d_2 = Sophus::SE3d::log(A*B)/2;
        const double d1 = d_2[3], d2 = d_2[4], d3 = d_2[5], dx = d_2[0], dy = d_2[1], dz = d_2[2];
        // this is using the 2nd order cbh expansion, to evaluate (I + 0.5 [Adj*x, log(AB)])*Adj
        // refer to the thesis by Hauke Strasdat, Appendix 3.
        return (Eigen::Matrix<double, 6, 6>() <<
                  1,  d3, -d2,   0,  dz, -dy,
                -d3,   1,  d1, -dz,   0,  dx,
                 d2, -d1,   1,  dy, -dx,   0,
                  0,   0,   0,   1,  d3, -d2,
                  0,   0,   0, -d3,   1,  d1,
                  0,   0,   0,  d2, -d1,   1
               ).finished() * A.Adj();
    }

    inline std::vector<Eigen::Matrix<double,3,3> > dLog_dR(const Eigen::Matrix3d R)
    {
        std::vector<Eigen::Matrix<double,3,3> > vRes(3);
        const double s1 = R(0)/2 + R(4)/2 + R(8)/2 - 0.5;
        const double s2 = - (R(5) - R(7))/(4*(powi(s1,2) - 1)) - (s1*acos(s1)*(R(5) - R(7)))/(4*pow(1 - powi(s1,2),3.0/2.0));
        const double s3 = acos(s1)/(2*sqrt(1 - powi(s1,2)));
        vRes[0] << s2, 0, 0, 0, s2, s3, 0, -s3, s2;

        const double s4 = s1; // r0/2 + R(4)/2 + R(8)/2 - 1/2
        const double s5 = (R(2) - R(6))/(4*(powi(s4,2) - 1)) + (s4*acos(s4)*(R(2) - R(6)))/(4*pow(1 - powi(s4,2),3.0/2.0));
        const double s6 = ( 1/sqrt(1 - powi(s4,2)) )*acos(s4)*0.5;
        vRes[1] << s5, 0, -s6, 0, s5, 0, s6, 0, s5;

        const double s7 = s1; // r0/2 + R(4)/2 + R(8)/2 - 1/2;
        const double s8 = -(R(1) - R(3))/(4*(powi(s7,2) - 1)) - (s7*acos(s7)*(R(1) - R(3)))/(4*pow(1 - powi(s7,2),3.0/2.0));
        const double s9 = acos(s7)/(2*sqrt(1 - powi(s7,2)));
        vRes[2] << s8, s9, 0, -s9, s8, 0, 0, 0, s8;

        return vRes;
    }
}

template< int LmSize, int PoseSize >
class BundleAdjuster
{
public:
    struct ImuPose
    {
        Sophus::SE3d Twp;
        Eigen::Vector3d V;
        Eigen::Vector3d W;
        double Time;
    };

    struct ImuMeasurement
    {
        ImuMeasurement(const Eigen::Vector3d& w,const Eigen::Vector3d& a): W(w), A(a) {}
        Eigen::Vector3d W;
        Eigen::Vector3d A;
        double Time;
        ImuMeasurement operator*(const double &rhs) {
            return ImuMeasurement( W*rhs, A*rhs );
        }
        ImuMeasurement operator+(const ImuMeasurement &rhs) {
            return ImuMeasurement( W+rhs.W, A+rhs.A );
        }
    };

    struct UnaryResidual
    {
        unsigned int PoseId;
        unsigned int ResidualId;
        unsigned int ResidualOffset;
        double       W;
        Sophus::SE3d Twp;
        Eigen::Matrix<double,6,6> dZ_dX;
        Eigen::Vector6d Residual;
    };

    struct BinaryResidual
    {
        unsigned int PoseAId;
        unsigned int PoseBId;
        unsigned int ResidualId;
        unsigned int ResidualOffset;
        double       W;
        Sophus::SE3d Tab;
        Eigen::Matrix<double,6,6> dZ_dX1;
        Eigen::Matrix<double,6,6> dZ_dX2;
        Eigen::Vector6d Residual;
    };

    struct ProjectionResidual
    {
        Eigen::Vector2d Z;
        unsigned int PoseId;
        unsigned int LandmarkId;
        unsigned int CameraId;
        unsigned int ResidualId;
        unsigned int ResidualOffset;
        double       W;

        Eigen::Matrix<double,2,LmSize> dZ_dX;
        Eigen::Matrix<double,2,6> dZ_dP;
        Eigen::Vector2d Residual;
    };

    struct ImuResidual
    {
        unsigned int PoseAId;
        unsigned int PoseBId;
        unsigned int ResidualId;
        unsigned int ResidualOffset;
        double       W;
        Sophus::SE3d Tab;
        Eigen::Matrix<double,6,6> dZ_dX1;
        Eigen::Matrix<double,6,6> dZ_dX2;
        Eigen::Vector9d Residual;
    };

    InterpolationBuffer<ImuMeasurement,double> m_ImuBuffer;

    struct Pose
    {
        Sophus::SE3d T_wp;
        Eigen::Vector3d V;
        bool IsActive;
        unsigned int Id;
        unsigned int OptId;
        std::vector<ProjectionResidual*> ProjResiduals;
        std::vector<BinaryResidual*> BinaryResiduals;
        std::vector<UnaryResidual*> UnaryResiduals;
        std::vector<Sophus::SE3d> T_sw;

        // imu data associated with this pose
        std::vector<ImuMeasurement> ImuMeasurements;
        std::vector<ImuPose> ImuPoses;

        const Sophus::SE3d& GetTsw(const unsigned int camId, const calibu::CameraRig& rig)
        {
            while(T_sw.size() <= camId ){
              T_sw.push_back( (T_wp*rig.cameras[T_sw.size()].T_wc).inverse());
            }
            return T_sw[camId];
        }
    };



    struct Landmark
    {
        Eigen::Vector4d X_s;
        Eigen::Vector4d X_w;
        std::vector<ProjectionResidual*> ProjResiduals;
        unsigned int OptId;
        unsigned int RefPoseId;
        unsigned int RefCamId;
    };

    ///////////////////////////////////////////////////////////////////////////////////////////////
    BundleAdjuster() {}

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
    Pose IntegrateImu(ImuPose& pose, const ImuMeasurement& zStart,
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
        aug_Twv.so3() = Rv2v1*pose.T_wp.so3();
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
    void Init(const calibu::CameraRig &rig, const unsigned int uNumPoses,
                                   const unsigned int uNumLandmarks,
                                   const unsigned int uNumMeasurements)
    {
        m_uNumActivePoses = 0;
        m_uProjResidualOffset = 0;
        m_uBinaryResidualOffset = 0;
        m_uUnaryResidualOffset = 0;
        m_Rig = rig;
        m_vLandmarks.reserve(uNumLandmarks);
        m_vProjResiduals.reserve(uNumMeasurements);
        m_vPoses.reserve(uNumPoses);

        // clear all arrays
        m_vPoses.clear();
        m_vProjResiduals.clear();
        m_vBinaryResiduals.clear();
        m_vLandmarks.clear();

        AddImuResidual(0,1);
    }    

    ///////////////////////////////////////////////////////////////////////////////////////////////
    unsigned int AddPose(const Sophus::SE3d& T_wp, const bool bIsActive = true)
    {
        Pose pose;
        pose.T_wp = T_wp;
        pose.IsActive = bIsActive;
        pose.T_sw.reserve(m_Rig.cameras.size());
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
    unsigned int AddLandmark(const Eigen::Vector4d& X_w,const unsigned int uRefPoseId, const unsigned int uRefCamId = 0)
    {
        assert(uRefPoseId < m_vPoses.size());
        Landmark landmark;
        landmark.X_w = X_w;
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
        UnaryResidual meas;
        meas.W = 1.0;
        meas.PoseId = uPoseId;
        meas.ResidualId = m_vUnaryResiduals.size();
        meas.ResidualOffset = m_uUnaryResidualOffset;
        meas.Twp = Twp;

        m_vUnaryResiduals.push_back(meas);
        m_uUnaryResidualOffset += 6;

        // we add this to both poses, as each one has a jacobian cell associated
        m_vPoses[uPoseId].BinaryResiduals.push_back(&m_vUnaryResiduals.back());
        return meas.ResidualId;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    unsigned int AddBinaryConstraint(const unsigned int uPoseAId,
                                     const unsigned int uPoseBId,
                                     const Sophus::SE3d& Tab)
    {
        assert(uPoseAId < m_vPoses.size());
        assert(uPoseBId < m_vPoses.size());

        //now add this constraint to pose A
        BinaryResidual meas;
        meas.W = 1.0;
        meas.PoseAId = uPoseAId;
        meas.PoseBId = uPoseBId;
        meas.ResidualId = m_vBinaryResiduals.size();
        meas.ResidualOffset = m_uBinaryResidualOffset;
        meas.Tab = Tab;

        m_vBinaryResiduals.push_back(meas);
        m_uBinaryResidualOffset += 6;

        // we add this to both poses, as each one has a jacobian cell associated
        m_vPoses[uPoseAId].BinaryResiduals.push_back(&m_vBinaryResiduals.back());
        m_vPoses[uPoseBId].BinaryResiduals.push_back(&m_vBinaryResiduals.back());
        return meas.ResidualId;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    unsigned int AddProjectionResidual(const Eigen::Vector2d z,
                                    const unsigned int uPoseId,
                                    const unsigned int uLandmarkId,
                                    const unsigned int uCameraId)
    {
        assert(uLandmarkId < m_vLandmarks.size());
        assert(uPoseId < m_vPoses.size());

        ProjectionResidual meas;
        meas.W = 1.0;
        meas.LandmarkId = uLandmarkId;
        meas.PoseId = uPoseId;
        meas.Z = z;
        meas.CameraId = uCameraId;
        meas.ResidualId = m_vProjResiduals.size();
        meas.ResidualOffset = m_uProjResidualOffset;

        m_vProjResiduals.push_back(meas);
        m_uProjResidualOffset += 2;

        m_vLandmarks[uLandmarkId].ProjResiduals.push_back(&m_vProjResiduals.back());
        m_vPoses[uPoseId].ProjResiduals.push_back(&m_vProjResiduals.back());

        return meas.ResidualId;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    unsigned int AddImuResidual(const unsigned int uPoseAId,
                                const unsigned int uPoseBId)
    {
        // we must be using 9DOF poses for IMU
        assert(PoseSize == 9);
        m_ImuBuffer.AddElement(ImuMeasurement(Eigen::Vector3d(),Eigen::Vector3d()));
        ImuMeasurement data = m_ImuBuffer.GetElement(0.1);
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
            Eigen::VectorXd bp(uNumPoses*6);
            Eigen::VectorXd bl(uNumLm*LmSize);
            Eigen::SparseBlockVectorProductDenseResult(m_Jpt,m_R,bp);
            Eigen::SparseBlockMatrix< Eigen::Matrix<double,LmSize,LmSize> > V_inv(uNumLm,uNumLm);
            Eigen::VectorXd rhs_p(uNumPoses*6);
            Eigen::SparseBlockMatrix< Eigen::Matrix<double,LmSize,6> > Wt(uNumLm,uNumPoses);
            Eigen::MatrixXd S(uNumPoses*6,uNumPoses*6);
            std::cout << "  Rhs vector mult took " << Toc(dMatTime) << " seconds." << std::endl;

            dMatTime = Tic();

            // TODO: suboptimal, the matrices are symmetric. We should only multipl one half
            Eigen::SparseBlockMatrix< Eigen::Matrix<double,6,6> > U(uNumPoses,uNumPoses);
            Eigen::SparseBlockProduct(m_Jpt,m_Jp,U);

            // add the contribution from the binary terms if any
            if( m_vBinaryResiduals.size() > 0 ) {
                Eigen::SparseBlockMatrix< Eigen::Matrix<double,6,6> > Jppt_Jpp(uNumPoses,uNumPoses);
                Eigen::SparseBlockProduct(m_Jppt,m_Jpp,Jppt_Jpp);
                Eigen::SparseBlockAdd(U,Jppt_Jpp,U);
            }

            // add the contribution from the unary terms if any
            if( m_vUnaryResiduals.size() > 0 ) {
                Eigen::SparseBlockMatrix< Eigen::Matrix<double,6,6> > Jut_Ju(uNumPoses,uNumPoses);
                Eigen::SparseBlockProduct(m_Jut,m_Ju,Jut_Ju);
                Eigen::SparseBlockAdd(U,Jut_Ju,U);
            }

            if( uNumLm > 0) {
                Eigen::SparseBlockVectorProductDenseResult(m_Jlt,m_R,bl);

                Eigen::SparseBlockMatrix< Eigen::Matrix<double,LmSize,LmSize> > V(uNumLm,uNumLm);
                Eigen::SparseBlockProduct(m_Jlt,m_Jl,V);

                // TODO this is really suboptimal, we should write a function to transpose the matrix
                Eigen::SparseBlockMatrix< Eigen::Matrix<double,6,LmSize> > W(uNumPoses,uNumLm);
                Eigen::SparseBlockProduct(m_Jpt,m_Jl,W);

                Eigen::SparseBlockProduct(m_Jlt,m_Jp,Wt);

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
                Eigen::SparseBlockMatrix< Eigen::Matrix<double,6,LmSize> > W_V_inv(uNumPoses,uNumLm);
                Eigen::SparseBlockProduct(W,V_inv,W_V_inv);

                Eigen::SparseBlockMatrix< Eigen::Matrix<double,6,6> > WV_invWt(uNumPoses,uNumPoses);
                Eigen::SparseBlockProduct(W_V_inv,Wt,WV_invWt);

                // this in-place operation should be fine for subtraction
                Eigen::SparseBlockSubtractDenseResult(U,WV_invWt,S);

                // now form the rhs for the pose equations
                Eigen::VectorXd WV_inv_bl(uNumPoses*6);
                Eigen::SparseBlockVectorProductDenseResult(W_V_inv,bl,WV_inv_bl);

                rhs_p = bp - WV_inv_bl;
                std::cout << "  Rhs calculation took " << Toc(dMatTime) << " seconds." << std::endl;
            }else{
                Eigen::LoadDenseFromSparse(U,S);
            }

            std::cout << "Setup took " << Toc(dTime) << " seconds." << std::endl;

            // now we have to solve for the pose constraints
            dTime = Tic();
            Eigen::VectorXd delta_p = uNumPoses == 0 ? Eigen::VectorXd() : S.ldlt().solve(rhs_p);
            Eigen::VectorXd delta_l( uNumLm*LmSize );
            std::cout << "Cholesky solve of " << uNumPoses << " by " << uNumPoses << "matrix took " << Toc(dTime) << " seconds." << std::endl;

            if( uNumLm > 0) {
                dTime = Tic();
                Eigen::VectorXd Wt_delta_p( uNumLm*LmSize );
                Eigen::SparseBlockVectorProductDenseResult(Wt,delta_p,Wt_delta_p);
                Eigen::VectorXd rhs_l( uNumLm*LmSize );
                rhs_l =  bl - Wt_delta_p;

                for(size_t ii = 0 ; ii < uNumLm ; ii++){
                    delta_l.block<LmSize,1>( ii*LmSize, 0 ).noalias() =  V_inv.coeff(ii,ii)*rhs_l.block<LmSize,1>(ii*LmSize,0);
                }

                // update the landmarks
                for (size_t ii = 0 ; ii < uNumLm ; ii++){
                    if(LmSize == 1){
                        m_vLandmarks[ii].X_s.template tail<LmSize>() += delta_l.template segment<LmSize>(m_vLandmarks[ii].OptId*LmSize);
                    }else{
                        m_vLandmarks[ii].X_s.template head<LmSize>() += delta_l.template segment<LmSize>(m_vLandmarks[ii].OptId*LmSize);
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
                    m_vPoses[ii].T_wp *= Sophus::SE3d::exp(delta_p.block<6,1>(m_vPoses[ii].OptId*6,0));
                    // clear the vector of Tsw values as they will need to be recalculated
                    m_vPoses[ii].T_sw.clear();
                }
                // else{
                //  std::cout << " Pose " << ii << " is inactive." << std::endl;
                //  }
            }
            std::cout << "BA iteration " << kk <<  " error: " << m_R.norm() << std::endl;
        }
            // std::cout << "Solve took " << Toc(dTime) << " seconds." << std::endl;
    }


    const Sophus::SE3d& GetPose(const unsigned int id) const  { return m_vPoses[id].T_wp; }
    // return the landmark in the world frame
    const Eigen::Vector4d& GetLandmark(const unsigned int id)
    {
        Landmark& lm = m_vLandmarks[id];
        lm.X_w = Sophus::MultHomogeneous(m_vPoses[lm.RefPoseId].GetTsw(lm.RefCamId,m_Rig).inverse(), lm.X_s);
        return lm.X_w;
    }


private:
    void _BuildProblem()
    {
        // resize as needed
        const unsigned int uNumPoses = m_uNumActivePoses;
        const unsigned int uNumLm = m_vLandmarks.size();
        const unsigned int uNumProjMeas = m_vProjResiduals.size();
        const unsigned int uNumBinMeas = m_vBinaryResiduals.size();
        const unsigned int uNumUnMeas = m_vUnaryResiduals.size();

        m_Jp.resize(uNumProjMeas,uNumPoses);
        m_Jpt.resize(uNumPoses,uNumProjMeas);

        m_Jpp.resize(uNumBinMeas,uNumPoses);
        m_Jppt.resize(uNumPoses,uNumBinMeas);

        m_Ju.resize(uNumUnMeas,uNumPoses);
        m_Jut.resize(uNumPoses,uNumUnMeas);

        m_Jl.resize(uNumProjMeas,uNumLm);
        m_Jlt.resize(uNumLm,uNumProjMeas);

        m_R.resize(uNumProjMeas*2,1);


        // these calls remove all the blocks, but KEEP allocated memory as long as the object is alive
        m_Jp.setZero();
        m_Jpt.setZero();

        m_Jpp.setZero();
        m_Jppt.setZero();

        m_Jpp.setZero();
        m_Jppt.setZero();

        m_Ju.setZero();
        m_Jut.setZero();

        m_R.setZero();

        // used to store errors for robust norm calculation
        m_vErrors.reserve(uNumProjMeas);
        m_vErrors.clear();


        double dPortionTransfer = 0, dPortionJac = 0, dPortionSparse = 0;

        // set all jacobians
        double dTime = Tic();
        for( ProjectionResidual& res : m_vProjResiduals ){
            double dPortTime = Tic();
            // calculate measurement jacobians

            // T_sw = T_cv * T_vw
            Landmark& lm = m_vLandmarks[res.LandmarkId];
            Pose& pose = m_vPoses[res.PoseId];
            Pose& refPose = m_vPoses[lm.RefPoseId];
            lm.X_s = Sophus::MultHomogeneous(refPose.GetTsw(lm.RefCamId,m_Rig) ,lm.X_w);
            const Sophus::SE3d parentTws = refPose.GetTsw(lm.RefCamId,m_Rig).inverse();

            const Eigen::Vector2d p = m_Rig.cameras[res.CameraId].camera.Transfer3D(pose.GetTsw(res.CameraId,m_Rig)*parentTws,
                                                                                    lm.X_s.template head<3>(),lm.X_s(3));
            res.Residual = res.Z - p;
//            std::cout << "Residual for meas " << meas.MeasurementId << " and landmark " << meas.LandmarkId << " with camera " << meas.CameraId << " is " << meas.Residual.transpose() << std::endl;

            // this array is used to calculate the robust norm
            m_vErrors.push_back(res.Residual.squaredNorm());

            dPortionTransfer += Toc(dPortTime);

            dPortTime = Tic();            
            const Eigen::Matrix<double,2,4> dTdP = m_Rig.cameras[res.CameraId].camera.dTransfer3D_dP(pose.GetTsw(res.CameraId,m_Rig)*parentTws,
                                                                                                     lm.X_s.template head<3>(),lm.X_s(3));
            res.dZ_dX = dTdP.block<2,LmSize>( 0, LmSize == 3 ? 0 : 3 );

            if( pose.IsActive ) {
                for(unsigned int ii=0; ii<6; ++ii){
                    const Eigen::Matrix<double,2,4> dTdP = m_Rig.cameras[res.CameraId].camera.dTransfer3D_dP(pose.GetTsw(res.CameraId,m_Rig),
                                                                                                             lm.X_w.template head<3>(),lm.X_w(3));
                    res.dZ_dP.template block<2,1>(0,ii) = dTdP * -Sophus::SE3::generator(ii) * lm.X_w;
                }
                //Eigen::Matrix<double,2,6> J_fd;
                //double dEps = 1e-6;
                //for(int ii = 0; ii < 6 ; ii++) {
                //    Eigen::Matrix<double,6,1> delta;
                //    delta.setZero();
                //    delta[ii] = dEps;
                //    Sophus::SE3d Tsw = (pose.T_wp*Sophus::SE3d::exp(delta)*m_Rig.cameras[meas.CameraId].T_wc).inverse();
                //    const Eigen::Vector2d pPlus = m_Rig.cameras[meas.CameraId].camera.Transfer3D(Tsw,X_w.head(3),X_w[3]);
                //    delta[ii] = -dEps;
                //    Tsw = (pose.T_wp*Sophus::SE3d::exp(delta)*m_Rig.cameras[meas.CameraId].T_wc).inverse();
                //    const Eigen::Vector2d pMinus = m_Rig.cameras[meas.CameraId].camera.Transfer3D(Tsw,X_w.head(3),X_w[3]);
                //    J_fd.col(ii) = (pPlus-pMinus)/(2*dEps);
                //}
                //std::cout << "J:" << meas.dZ_dP << std::endl;
                //std::cout << "J_fd:" << J_fd << std::endl;
            }


            dPortionJac += Toc(dPortTime);

            // set the residual in m_R which is dense
            m_R.segment<2>(res.ResidualOffset) = res.Residual;
        }

        // build binary residual jacobians
        for( BinaryResidual& res : m_vBinaryResiduals ){
            const Sophus::SE3d& Twa = m_vPoses[res.PoseAId].T_wp;
            const Sophus::SE3d& Twb = m_vPoses[res.PoseBId].T_wp;
            res.dZ_dX1 = Sophus::dLog_dX(Twa, res.Tab * Twb.inverse());
            // the negative sign here is because exp(x) is inside the inverse when we invert (Twb*exp(x)).inverse
            res.dZ_dX2 = -Sophus::dLog_dX(Twa * res.Tab , Twb.inverse());

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
        }

        for( UnaryResidual& res : m_vUnaryResiduals ){
            const Sophus::SE3d& Twp = m_vPoses[res.PoseId].T_wp;
            res.dZ_dX = Sophus::dLog_dX(Twp, res.Twp.inverse());
        }

        // get the sigma for robust norm calculation. This call is O(n) on average,
        // which is desirable over O(nlogn) sort
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
                    m_Jp.insert(pRes->ResidualId,pose.OptId).template block<2,6>(0,0) = pRes->dZ_dP * pRes->W;
                    m_Jpt.insert(pose.OptId,pRes->ResidualId).template block<6,2>(0,0) = pRes->dZ_dP.transpose() * pRes->W;
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
    Eigen::SparseBlockMatrix< Eigen::Matrix<double,2,PoseSize> > m_Jp;
    Eigen::SparseBlockMatrix< Eigen::Matrix<double,PoseSize,2> > m_Jpt;

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
    std::vector<ProjectionResidual> m_vProjResiduals;
    std::vector<BinaryResidual> m_vBinaryResiduals;
    std::vector<UnaryResidual> m_vUnaryResiduals;
    std::vector<ImuResidual> m_vImuResiduals;
    std::vector<double> m_vErrors;
};







#endif // BUNDLEADUJSTER_H
