#ifndef BUNDLEADUJSTER_H
#define BUNDLEADUJSTER_H

#include <sophus/se3.hpp>
#include <vector>
#include <calibu/Calibu.h>
//#include <cholmod.h>
#include <Eigen/Sparse>
#include "SparseBlockMatrix.h"
#include "SparseBlockMatrixOps.h"
#include "CeresCostFunctions.h"
#include "Utils.h"
#include "Types.h"


namespace ba {

template< typename Scalar=double,int LmSize=1, int PoseSize=6, int CalibSize=8 >
class BundleAdjuster
{
    static const unsigned int kLmDim = LmSize;
    static const unsigned int kPoseDim = PoseSize;
    static const unsigned int kCalibDim = CalibSize;

    typedef PoseT<Scalar> Pose;
    typedef LandmarkT<Scalar,LmSize> Landmark;
    typedef ProjectionResidualT<Scalar,LmSize> ProjectionResidual;
    typedef ImuMeasurementT<Scalar> ImuMeasurement;
    typedef UnaryResidualT<Scalar> UnaryResidual;
    typedef BinaryResidualT<Scalar> BinaryResidual;
    typedef ImuResidualT<Scalar, PoseSize, PoseSize> ImuResidual;
    typedef ImuCalibrationT<Scalar> ImuCalibration;
    typedef ImuPoseT<Scalar> ImuPose;

    typedef Eigen::Matrix<Scalar,2,1> Vector2t;
    typedef Eigen::Matrix<Scalar,3,1> Vector3t;
    typedef Eigen::Matrix<Scalar,4,1> Vector4t;
    typedef Eigen::Matrix<Scalar,6,1> Vector6t;
    typedef Eigen::Matrix<Scalar,7,1> Vector7t;
    typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1> VectorXt;
    typedef Eigen::Matrix<Scalar,3,3> Matrix3t;
    typedef Sophus::SE3Group<Scalar> SE3t;

public:
    ///////////////////////////////////////////////////////////////////////////////////////////////
    BundleAdjuster() :
        m_Imu(SE3t(),Vector3t::Zero(),Vector3t::Zero(),Vector2t::Zero()),
        m_bEnableTranslation(false), m_dTotalTvsChange(0),m_dTvsTransPrior(1.0),m_dTvsRotPrior(1.0) {}


    ///////////////////////////////////////////////////////////////////////////////////////////////
    void Init(const unsigned int uNumPoses,
              const unsigned int uNumMeasurements,
              const unsigned int uNumLandmarks = 0,
              const calibu::CameraRigT<Scalar> *pRig = 0 )
    {
        // if LmSize == 0, there is no need for a camera rig or landmarks
        assert(pRig != 0 || LmSize == 0);
        assert(uNumLandmarks != 0 || LmSize == 0);

        m_uNumActivePoses = 0;
        m_uNumActiveLandmakrs = 0;
        m_uProjResidualOffset = 0;
        m_uBinaryResidualOffset = 0;
        m_uUnaryResidualOffset = 0;
        m_uImuResidualOffset = 0;
        if(pRig != 0){
            m_Rig = *pRig;
            m_Imu.t_vs = m_Rig.cameras[0].T_wc;
            m_dLastTvs = m_Imu.t_vs;
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
    unsigned int AddPose(const SE3t& Twp, const bool bIsActive = true, const double dTime = -1)
    {
        return AddPose( Twp, Sophus::SE3d(), VectorXt(5).setZero(), Vector3t::Zero(), Vector6t::Zero(), bIsActive, dTime);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    unsigned int AddPose(const SE3t& Twp, const SE3t& Tvs, const VectorXt camParams, const Vector3t& V, const Vector6t& B, const bool bIsActive = true, const double dTime = -1)
    {
        Pose pose;
        pose.time = dTime;
        pose.t_wp = Twp;
        pose.t_vs = Tvs;
        pose.v_w = V;
        pose.b = B;
        pose.cam_params = camParams;
        pose.is_active = bIsActive;
        pose.t_sw.reserve(m_Rig.cameras.size());
        // assume equal distribution of measurements amongst poses
        pose.proj_residuals.reserve(m_vProjResiduals.capacity()/m_vPoses.capacity());
        pose.id = m_vPoses.size();
        if(bIsActive){
            pose.opt_id = m_uNumActivePoses;
            m_uNumActivePoses++;
        }else{
            // the is active flag should be checked before reading this value, to see if the pose
            // is part of the optimization or not
            pose.opt_id = UINT_MAX;
        }

        m_vPoses.push_back(pose);        
        // std::cout << "Addeded pose with IsActive= " << pose.IsActive << ", Id = " << pose.Id << " and OptId = " << pose.OptId << std::endl;

        return pose.id;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    unsigned int AddLandmark(const Vector4t& Xw,const unsigned int uRefPoseId, const unsigned int uRefCamId, const bool bIsActive)
    {
        assert(uRefPoseId < m_vPoses.size());
        Landmark landmark;
        landmark.x_w = Xw;
        // assume equal distribution of measurements amongst landmarks
        landmark.proj_residuals.reserve(m_vProjResiduals.capacity()/m_vLandmarks.capacity());
        landmark.ref_pose_id = uRefPoseId;
        landmark.ref_cam_id = uRefCamId;
        landmark.is_active = bIsActive;
        landmark.id = m_vLandmarks.size();
        if(bIsActive){
            landmark.opt_id = m_uNumActiveLandmakrs;
            m_uNumActiveLandmakrs++;
        }else{
            // the is active flag should be checked before reading this value, to see if the pose
            // is part of the optimization or not
            landmark.opt_id = UINT_MAX;
        }

        m_vLandmarks.push_back(landmark);
         //std::cout << "Adding landmark with Xw = [" << Xw.transpose() << "], refPoseId " << uRefPoseId << ", uRefCamId " << uRefCamId << ", OptId " << landmark.OptId << std::endl;
        return landmark.id;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    unsigned int AddUnaryConstraint(const unsigned int uPoseId,
                                    const SE3t& Twp)
    {
        assert(uPoseId < m_vPoses.size());

        //now add this constraint to pose A
        UnaryResidual residual;
        residual.weight = 1.0;
        residual.pose_id = uPoseId;
        residual.residual_id = m_vUnaryResiduals.size();
        residual.residual_offset = m_uUnaryResidualOffset;
        residual.t_wp = Twp;

        m_vUnaryResiduals.push_back(residual);
        m_uUnaryResidualOffset += UnaryResidual::kResSize;

        // we add this to both poses, as each one has a jacobian cell associated
        m_vPoses[uPoseId].unary_residuals.push_back(residual.residual_id);
        return residual.residual_id;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    unsigned int AddBinaryConstraint(const unsigned int uPoseAId,
                                     const unsigned int uPoseBId,
                                     const SE3t& Tab)
    {
        assert(uPoseAId < m_vPoses.size());
        assert(uPoseBId < m_vPoses.size());

        //now add this constraint to pose A
        BinaryResidual residual;
        residual.weight = 1.0;
        residual.x1_id = uPoseAId;
        residual.x2_id = uPoseBId;
        residual.residual_id = m_vBinaryResiduals.size();
        residual.residual_offset = m_uBinaryResidualOffset;
        residual.t_ab = Tab;

        m_vBinaryResiduals.push_back(residual);
        m_uBinaryResidualOffset += BinaryResidual::kResSize;

        // we add this to both poses, as each one has a jacobian cell associated
        m_vPoses[uPoseAId].binary_residuals.push_back(residual.residual_id);
        m_vPoses[uPoseBId].binary_residuals.push_back(residual.residual_id);
        return residual.residual_id;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    unsigned int AddProjectionResidual(const Vector2t z,
                                    const unsigned int uMeasPoseId,
                                    const unsigned int uLandmarkId,
                                    const unsigned int uCameraId,
                                    const Scalar dWeight = 1.0)
    {
        assert(uLandmarkId < m_vLandmarks.size());
        assert(uMeasPoseId < m_vPoses.size());

        ProjectionResidual residual;
        residual.weight = dWeight;
        residual.landmark_id = uLandmarkId;
        residual.x_meas_id = uMeasPoseId;
        residual.x_ref_id = m_vLandmarks[uLandmarkId].ref_pose_id;
        residual.z = z;
        residual.cam_id = uCameraId;
        residual.residual_id = m_vProjResiduals.size();
        residual.residual_offset = m_uProjResidualOffset;

        Landmark& lm = m_vLandmarks[uLandmarkId];
        // set the reference measurement
        if(uMeasPoseId == residual.x_ref_id && uCameraId == lm.ref_cam_id){
            lm.z_ref = z;
        }

        // this prevents adding measurements of the landmark in the privileged frame in which
        // it was first seen, as with inverse depth, the error would always be zero.
        // however, if 3dof parametrization of landmarks is used, we add all measurements
        if(uMeasPoseId != residual.x_ref_id || uCameraId != lm.ref_cam_id || LmSize != 1){
            m_vPoses[uMeasPoseId].proj_residuals.push_back(residual.residual_id);
            m_vLandmarks[uLandmarkId].proj_residuals.push_back(residual.residual_id);
            if(LmSize == 1){
                m_vPoses[residual.x_ref_id].proj_residuals.push_back(residual.residual_id);
            }
        }else{
            // we should not add this residual
             return -1;
        }

        m_vProjResiduals.push_back(residual);
        m_uProjResidualOffset += ProjectionResidual::kResSize;

        return residual.residual_id;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    unsigned int AddImuResidual(const unsigned int uPoseAId,
                                const unsigned int uPoseBId,
                                const std::vector<ImuMeasurement>& vImuMeas,
                                const Scalar dWeight = 1.0)
    {
        assert(uPoseAId < m_vPoses.size());
        assert(uPoseBId < m_vPoses.size());
        // we must be using 9DOF poses for IMU residuals
        //assert(PoseSize == 9);

        ImuResidual residual;
        residual.weight = dWeight;
        residual.pose1_id = uPoseAId;
        residual.pose2_id = uPoseBId;
        residual.measurements = vImuMeas;
        residual.residual_id = m_vImuResiduals.size();
        residual.residual_offset = m_uImuResidualOffset;

        m_vImuResiduals.push_back(residual);
        m_uImuResidualOffset += ImuResidual::kResSize;

        m_vPoses[uPoseAId].inertial_residuals.push_back(residual.residual_id);
        m_vPoses[uPoseBId].inertial_residuals.push_back(residual.residual_id);
        return residual.residual_id;
    }
    void Solve(const unsigned int uMaxIter);

    bool IsTranslationEnabled() { return m_bEnableTranslation; }
    unsigned int GetNumPoses() const { return m_vPoses.size(); }
    const ImuResidual& GetImuResidual(const unsigned int id) const { return m_vImuResiduals[id]; }
    const ImuCalibration& GetImuCalibration() const { return m_Imu; }
    void SetImuCalibration(const ImuCalibration& calib) { m_Imu = calib; }
    const Pose& GetPose(const unsigned int id) const  { return m_vPoses[id]; }
    // return the landmark in the world frame
    const Vector4t& GetLandmark(const unsigned int id) const { return m_vLandmarks[id].x_w; }

private:
    void _ApplyUpdate(const VectorXt &delta_p, const VectorXt &delta_l, const VectorXt &deltaCalib, const bool bRollback);
    void _EvaluateResiduals();
    void BuildProblem();

    // reprojection jacobians and residual
    Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,ProjectionResidual::kResSize,PoseSize> > m_Jpr;
    Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,PoseSize,ProjectionResidual::kResSize> > m_Jprt;
    // landmark jacobians
    Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,ProjectionResidual::kResSize,LmSize> > m_Jl;
    Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,LmSize,ProjectionResidual::kResSize> > m_Jlt;
    VectorXt m_Rpr;

    // pose/pose jacobian for binary constraints
    Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,BinaryResidual::kResSize,PoseSize> > m_Jpp;
    Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,PoseSize,BinaryResidual::kResSize> > m_Jppt;
    VectorXt m_Rpp;

    // pose/pose jacobian for unary constraints
    Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,UnaryResidual::kResSize,PoseSize> > m_Ju;
    Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,PoseSize,UnaryResidual::kResSize> > m_Jut;
    VectorXt m_Ru;

    // imu jacobian
    Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,ImuResidual::kResSize,PoseSize> > m_Ji;
    Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,PoseSize,ImuResidual::kResSize> > m_Jit;

    Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,ImuResidual::kResSize,CalibSize> > m_Jki;
    Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,CalibSize,ImuResidual::kResSize> > m_Jkit;

    Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,ProjectionResidual::kResSize,CalibSize> > m_Jkpr;
    Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,CalibSize,ProjectionResidual::kResSize> > m_Jkprt;


    VectorXt m_Ri;

    bool m_bEnableTranslation;
    double m_dTotalTvsChange;
    SE3t m_dLastTvs;
    double m_dProjError;
    double m_dBinaryError;
    double m_dUnaryError;
    double m_dImuError;
    double m_dTvsTransPrior;
    double m_dTvsRotPrior;
    unsigned int m_uNumActivePoses;
    unsigned int m_uNumActiveLandmakrs;
    unsigned int m_uBinaryResidualOffset;
    unsigned int m_uUnaryResidualOffset;
    unsigned int m_uProjResidualOffset;
    unsigned int m_uImuResidualOffset;
    calibu::CameraRigT<Scalar> m_Rig;
    std::vector<Pose> m_vPoses;
    std::vector<Landmark> m_vLandmarks;
    std::vector<ProjectionResidual > m_vProjResiduals;
    std::vector<BinaryResidual> m_vBinaryResiduals;
    std::vector<UnaryResidual> m_vUnaryResiduals;
    std::vector<ImuResidual> m_vImuResiduals;
    std::vector<Scalar> m_vErrors;

    ImuCalibration m_Imu;

};

static const int NOT_USED = 0;

// typedefs for convenience
template< typename Scalar >
using GlobalInertialBundleAdjuster = BundleAdjuster<Scalar, ba::NOT_USED,9>;
template< typename Scalar >
using InverseDepthVisualInertialBundleAdjuster = BundleAdjuster<Scalar, 1,9>;
template< typename Scalar >
using VisualInertialBundleAdjuster = BundleAdjuster<Scalar, 3,9>;

}





#endif // BUNDLEADUJSTER_H
