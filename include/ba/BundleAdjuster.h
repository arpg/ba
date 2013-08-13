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

template< typename Scalar=double,int LmSize=1, int PoseSize=6 >
class BundleAdjuster
{
    typedef PoseT<Scalar> Pose;
    typedef LandmarkT<Scalar,LmSize> Landmark;
    typedef ProjectionResidualT<Scalar,LmSize> ProjectionResidual;
    typedef ImuMeasurementT<Scalar> ImuMeasurement;
    typedef UnaryResidualT<Scalar> UnaryResidual;
    typedef BinaryResidualT<Scalar> BinaryResidual;
    typedef ImuResidualT<Scalar> ImuResidual;
    typedef ImuCalibrationT<Scalar> ImuCalibration;
    typedef ImuPoseT<Scalar> ImuPose;

    typedef Eigen::Matrix<Scalar,2,1> Vector2t;
    typedef Eigen::Matrix<Scalar,3,1> Vector3t;
    typedef Eigen::Matrix<Scalar,4,1> Vector4t;
    typedef Eigen::Matrix<Scalar,6,1> Vector6t;
    typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1> VectorXt;
    typedef Eigen::Matrix<Scalar,3,3> Matrix3t;
    typedef Sophus::SE3Group<Scalar> SE3t;

public:
    ///////////////////////////////////////////////////////////////////////////////////////////////
    BundleAdjuster() :
        m_Imu(SE3t(),Vector3t::Zero(),Vector3t::Zero(),Vector2t::Zero()) {}


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
    unsigned int AddPose(const SE3t& Twp, const bool bIsActive = true, const double dTime = -1)
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
    unsigned int AddLandmark(const Vector4t& Xw,const unsigned int uRefPoseId, const unsigned int uRefCamId = 0)
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
         //std::cout << "Adding landmark with Xw = [" << Xw.transpose() << "], refPoseId " << uRefPoseId << ", uRefCamId " << uRefCamId << ", OptId " << landmark.OptId << std::endl;
        return landmark.OptId;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    unsigned int AddUnaryConstraint(const unsigned int uPoseId,
                                    const SE3t& Twp)
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
                                     const SE3t& Tab)
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
    unsigned int AddProjectionResidual(const Vector2t z,
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
                                const std::vector<ImuMeasurement>& vImuMeas)
    {
        assert(uPoseAId < m_vPoses.size());
        assert(uPoseBId < m_vPoses.size());
        // we must be using 9DOF poses for IMU residuals
        assert(PoseSize == 9);

        ImuResidual residual;
        residual.W = 1.0;
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
    void Solve(const unsigned int uMaxIter);

    const ImuResidual& GetImuResidual(const unsigned int id) const { return m_vImuResiduals[id]; }
    const ImuCalibration& GetImuCalibration() const { return m_Imu; }
    const SE3t& GetPose(const unsigned int id) const  { return m_vPoses[id].Twp; }
    // return the landmark in the world frame
    const Vector4t& GetLandmark(const unsigned int id) const { return m_vLandmarks[id].Xw; }



private:
    void _BuildProblem();

    // reprojection jacobians and residual
    Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,ProjectionResidual::ResSize,PoseSize> > m_Jpr;
    Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,PoseSize,ProjectionResidual::ResSize> > m_Jprt;
    // landmark jacobians
    Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,ProjectionResidual::ResSize,LmSize> > m_Jl;
    Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,LmSize,ProjectionResidual::ResSize> > m_Jlt;
    VectorXt m_Rpr;

    // pose/pose jacobian for binary constraints
    Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,BinaryResidual::ResSize,PoseSize> > m_Jpp;
    Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,PoseSize,BinaryResidual::ResSize> > m_Jppt;
    VectorXt m_Rpp;

    // pose/pose jacobian for unary constraints
    Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,UnaryResidual::ResSize,PoseSize> > m_Ju;
    Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,PoseSize,UnaryResidual::ResSize> > m_Jut;
    VectorXt m_Ru;

    // imu jacobian
    Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,ImuResidual::ResSize,PoseSize> > m_Ji;
    Eigen::SparseBlockMatrix< Eigen::Matrix<Scalar,PoseSize,ImuResidual::ResSize> > m_Jit;

    // jacobian reserved for biases, gravity, etc
    // 2 for gravity, 6 for biases
    Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> m_Jki;
    VectorXt m_Ri;



    unsigned int m_uNumActivePoses;
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
