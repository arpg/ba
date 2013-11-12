#ifndef BUNDLEADUJSTER_H
#define BUNDLEADUJSTER_H

#include <sophus/se3.hpp>
#include <vector>
#include <calibu/Calibu.h>
//#include <cholmod.h>

#ifdef _X
#undef _X  // Weirdest undef ever. Defined in ctype.h. Conflicts in Eigen.
#endif

#include <Eigen/Sparse>
#include "SparseBlockMatrix.h"
#include "SparseBlockMatrixOps.h"
#include "CeresCostFunctions.h"
#include "Utils.h"
#include "Types.h"


namespace ba {

template<typename Scalar>
using BlockMat = Eigen::SparseBlockMatrix<Scalar>;

template<typename Scalar=double,int LmSize=1, int PoseSize=6, int CalibSize=8>
class BundleAdjuster
{
  static const unsigned int kPrPoseDim = 6;
  static const unsigned int kLmDim = LmSize;
  static const unsigned int kPoseDim = PoseSize;
  static const unsigned int kCalibDim = CalibSize;  

  typedef PoseT<Scalar> Pose;
  typedef LandmarkT<Scalar,LmSize> Landmark;
  typedef ProjectionResidualT<Scalar,LmSize> ProjectionResidual;
  typedef ImuMeasurementT<Scalar>     ImuMeasurement;
  typedef UnaryResidualT<Scalar> UnaryResidual;
  typedef BinaryResidualT<Scalar> BinaryResidual;
  typedef ImuResidualT<Scalar, kPoseDim, kPoseDim> ImuResidual;
  typedef ImuCalibrationT<Scalar> ImuCalibration;
  typedef ImuPoseT<Scalar> ImuPose;

  typedef Eigen::Matrix<Scalar,2,1> Vector2t;
  typedef Eigen::Matrix<Scalar,3,1> Vector3t;
  typedef Eigen::Matrix<Scalar,4,1> Vector4t;
  typedef Eigen::Matrix<Scalar,6,1> Vector6t;
  typedef Eigen::Matrix<Scalar,7,1> Vector7t;
  typedef Eigen::Matrix<Scalar,9,1> Vector9t;
  typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1> VectorXt;
  typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> MatrixXt;
  typedef Eigen::Matrix<Scalar,3,3> Matrix3t;
  typedef Sophus::SE3Group<Scalar> SE3t;

  struct Delta
  {
    VectorXt delta_p;
    VectorXt delta_l;
  };


public:
  static const bool kVelInState = (kPoseDim >= 9);
  static const bool kBiasInState = (kPoseDim >= 15);
  static const bool kTvsInState = (kPoseDim >= 21);
  static const bool kGravityInCalib = (kCalibDim >= 2);
  static const bool kTvsInCalib = (kCalibDim >= 8);
  static const bool kCamParamsInCalib = (kCalibDim > 8);

  ////////////////////////////////////////////////////////////////////////////
  BundleAdjuster() :
    imu_(SE3t(),Vector3t::Zero(),Vector3t::Zero(),Vector2t::Zero()),
    translation_enabled_(kCalibDim > 15 ? false : true),
    total_tvs_change_(0),
    tvs_trans_prior_(1.0),
    tvs_rot_prior_(1.0)
  {}


  ////////////////////////////////////////////////////////////////////////////
  void Init(const unsigned int num_poses,
            const unsigned int num_measurements,
            const unsigned int num_landmarks = 0,
            const calibu::CameraRigT<Scalar> *rig = 0,
            const Scalar trust_region_size = 1.0)
  {
    // if LmSize == 0, there is no need for a camera rig or landmarks
    assert(rig != 0 || LmSize == 0);
    assert(num_landmarks != 0 || LmSize == 0);

    // set the initial trust region size
    trust_region_size_ = trust_region_size;

    root_pose_id_ = 0;
    num_active_poses_ = 0;
    num_active_landmarks_ = 0;
    proj_residual_offset = 0;
    binary_residual_offset_ = 0;
    unary_residual_offset_ = 0;
    inertial_residual_offset_ = 0;
    if (rig != 0) {
      rig_ = *rig;
      imu_.t_vs = rig_.cameras[0].T_wc;
      last_tvs_ = imu_.t_vs;
    }
    landmarks_.reserve(num_landmarks);
    proj_residuals_.reserve(num_measurements);
    poses_.reserve(num_poses);

    // clear all arrays
    poses_.clear();
    proj_residuals_.clear();
    binary_residuals_.clear();
    unary_residuals_.clear();
    inertial_residuals_.clear();
    landmarks_.clear();
  }

  ////////////////////////////////////////////////////////////////////////////
  void SetGravity(const Vector3t& g){
    if (kGravityInCalib) {
      const Vector3t new_g_norm = g.normalized();
      const double p = asin(new_g_norm[1]);
      const double q = acos(std::min(1.0,std::max(-1.0,-new_g_norm[2]/cos(p))));
      imu_.g << p, q;
    }else {
      imu_.g_vec = g;
    }
  }

  ////////////////////////////////////////////////////////////////////////////
  unsigned int AddPose(const SE3t& t_wp,
                       const bool is_active = true,
                       const double time = -1)
  {
    return AddPose(t_wp, SE3t(), VectorXt(5).setZero(),
                   Vector3t::Zero(), Vector6t::Zero(), is_active, time);
  }

  ////////////////////////////////////////////////////////////////////////////
  unsigned int AddPose(const SE3t& t_wp, const SE3t& t_vs,
                       const VectorXt cam_params, const Vector3t& v_w,
                       const Vector6t& b, const bool is_active = true,
                       const double time = -1)
  {
    Pose pose;
    pose.time = time;
    pose.t_wp = t_wp;
    pose.t_vs = t_vs;
    pose.v_w = v_w;
    pose.b = b;
    pose.cam_params = cam_params;
    pose.is_active = is_active;
    pose.is_param_mask_used = false;
    pose.t_sw.reserve(rig_.cameras.size());

    pose.id = poses_.size();
    if (is_active) {
      pose.opt_id = num_active_poses_;
      num_active_poses_++;
    } else {
      // the is active flag should be checked before reading this value,
      //to see if the pose is part of the optimization or not
      pose.opt_id = UINT_MAX;
    }

    poses_.push_back(pose);
    // std::cout << "Addeded pose with IsActive= " << pose.IsActive <<
    // ", Id = " << pose.Id << " and OptId = " << pose.OptId << std::endl;

    return pose.id;
  }

  ////////////////////////////////////////////////////////////////////////////
  unsigned int AddLandmark(const Vector4t& x_w,
                           const unsigned int ref_pose_id,
                           const unsigned int ref_cam_id,
                           const bool is_active)
  {
    assert(ref_pose_id < poses_.size());
    Landmark landmark;
    landmark.x_w = x_w;
    // assume equal distribution of measurements amongst landmarks
    landmark.proj_residuals.reserve(
          proj_residuals_.capacity()/landmarks_.capacity());

    landmark.ref_pose_id = ref_pose_id;
    landmark.ref_cam_id = ref_cam_id;
    landmark.is_active = is_active;
    landmark.is_reliable = true;
    landmark.id = landmarks_.size();

    poses_[ref_pose_id].landmarks.push_back(landmark.id);
    // std::cout << "Adding landmark id " << landmark.id << " to pose " <<
    //                poses_[ref_pose_id].id << " with x_w: " <<
    //               landmark.x_w.transpose() << std::endl;

    if (is_active) {
      landmark.opt_id = num_active_landmarks_;
      num_active_landmarks_++;
    } else {
      // the is active flag should be checked before reading this value,
      //to see if the pose is part of the optimization or not
      landmark.opt_id = UINT_MAX;
    }

    landmarks_.push_back(landmark);
    //std::cout << "Adding landmark with Xw = [" << Xw.transpose() <<
    // "], refPoseId " << uRefPoseId << ", uRefCamId " << uRefCamId <<
    // ", OptId " << landmark.OptId << std::endl;
    return landmark.id;
  }

  ////////////////////////////////////////////////////////////////////////////
  unsigned int AddUnaryConstraint(const unsigned int pos_id,
                                  const SE3t& t_wp)
  {
    assert(pos_id < poses_.size());

    //now add this constraint to pose A
    UnaryResidual residual;
    residual.orig_weight = 1.0;
    residual.pose_id = pos_id;
    residual.residual_id = unary_residuals_.size();
    residual.residual_offset = unary_residual_offset_;
    residual.t_wp = t_wp;

    unary_residuals_.push_back(residual);
    unary_residual_offset_ += UnaryResidual::kResSize;

    // we add this to both poses, as each one has a jacobian cell associated
    poses_[pos_id].unary_residuals.push_back(residual.residual_id);
    return residual.residual_id;
  }

  ////////////////////////////////////////////////////////////////////////////
  unsigned int AddBinaryConstraint(const unsigned int pose1_id,
                                   const unsigned int pose2_id,
                                   const SE3t& t_ab)
  {
    assert(pose1_id < poses_.size());
    assert(pose2_id < poses_.size());

    //now add this constraint to pose A
    BinaryResidual residual;
    residual.orig_weight = 1.0;
    residual.x1_id = pose1_id;
    residual.x2_id = pose2_id;
    residual.residual_id = binary_residuals_.size();
    residual.residual_offset = binary_residual_offset_;
    residual.t_ab = t_ab;

    binary_residuals_.push_back(residual);
    binary_residual_offset_ += BinaryResidual::kResSize;

    // we add this to both poses, as each one has a jacobian cell associated
    poses_[pose1_id].binary_residuals.push_back(residual.residual_id);
    poses_[pose2_id].binary_residuals.push_back(residual.residual_id);

    return residual.residual_id;
  }

  ////////////////////////////////////////////////////////////////////////////
  unsigned int AddProjectionResidual(const Vector2t z,
                                     const unsigned int meas_pose_id,
                                     const unsigned int landmark_id,
                                     const unsigned int cam_id,
                                     const Scalar weight = 1.0)
  {
    assert(landmark_id < landmarks_.size());
    assert(meas_pose_id < poses_.size());

    ProjectionResidual residual;
    residual.orig_weight = weight;
    residual.landmark_id = landmark_id;
    residual.x_meas_id = meas_pose_id;
    residual.x_ref_id = landmarks_[landmark_id].ref_pose_id;
    residual.z = z;
    residual.cam_id = cam_id;
    residual.residual_id = proj_residuals_.size();
    residual.residual_offset = proj_residual_offset;

    Landmark& lm = landmarks_[landmark_id];
    // set the reference measurement
    if(meas_pose_id == residual.x_ref_id && cam_id == lm.ref_cam_id){
      lm.z_ref = z;
    }

    // this prevents adding measurements of the landmark in the privileged
    // frame in which it was first seen, as with inverse depth, the error
    // would always be zero. however, if 3dof parametrization of landmarks
    // is used, we add all measurements
    const unsigned int res_id = residual.residual_id;
    const bool diff_poses = meas_pose_id != residual.x_ref_id;
    if (diff_poses || cam_id != lm.ref_cam_id || LmSize != 1) {
      landmarks_[landmark_id].proj_residuals.push_back(res_id);
      if (diff_poses || LmSize  != 1) {
        poses_[meas_pose_id].proj_residuals.push_back(res_id);
        if (LmSize == 1) {
          poses_[residual.x_ref_id].proj_residuals.push_back(res_id);
        }
      }
    } else {
      // we should not add this residual
      return -1;
    }

    proj_residuals_.push_back(residual);
    proj_residual_offset += ProjectionResidual::kResSize;

    return residual.residual_id;
  }

  ////////////////////////////////////////////////////////////////////////////
  unsigned int AddImuResidual(const unsigned int pose1_id,
                              const unsigned int pose2_id,
                              const std::vector<ImuMeasurement>& imu_meas,
                              const Scalar weight = 1.0)
  {
    assert(pose1_id < poses_.size());
    assert(pose2_id < poses_.size());
    // we must be using 9DOF poses for IMU residuals
    //assert(kPoseDim == 9);

    ImuResidual residual;
    residual.orig_weight = weight;
    residual.pose1_id = pose1_id;
    residual.pose2_id = pose2_id;
    residual.measurements = imu_meas;
    residual.residual_id = inertial_residuals_.size();
    residual.residual_offset = inertial_residual_offset_;

    inertial_residuals_.push_back(residual);
    inertial_residual_offset_ += ImuResidual::kResSize;

    poses_[pose1_id].inertial_residuals.push_back(residual.residual_id);
    poses_[pose2_id].inertial_residuals.push_back(residual.residual_id);
    return residual.residual_id;
  }

  void Solve(const unsigned int uMaxIter,
             const Scalar gn_damping = 1.0,
             const bool error_increase_allowed = false,
             const bool use_dogleg = true);

  void SetRootPoseId(const unsigned int id) { root_pose_id_ = id; }

  bool IsTranslationEnabled() { return translation_enabled_; }
  unsigned int GetNumPoses() const { return poses_.size(); }
  unsigned int GetNumImuResiduals() const { return inertial_residuals_.size(); }
  unsigned int GetNumProjResiduals() const { return proj_residuals_.size(); }

  const ImuResidual& GetImuResidual(const unsigned int id)
  const { return inertial_residuals_[id]; }

  const ImuCalibration& GetImuCalibration() const { return imu_; }
  void SetImuCalibration(const ImuCalibration& calib) { imu_ = calib; }
  const Pose& GetPose(const unsigned int id) const  { return poses_[id]; }

  // return the landmark in the world frame
  const Vector4t& GetLandmark(const unsigned int id)  
  const { return landmarks_[id].x_w; }
  bool IsLandmarkReliable(const unsigned int id)
  const { return landmarks_[id].is_reliable; }

  void GetErrors( Scalar &proj_error,
                  Scalar &unary_error,
                  Scalar &binary_error,
                  Scalar &inertial_error)
  {
    proj_error = proj_error_;
    unary_error = unary_error_;
    binary_error = binary_error_;
    inertial_error = inertial_error_;
  }

private:
  bool SolveInternal(VectorXt rhs_p_sc, const Scalar gn_damping,
                     const bool error_increase_allowed, const bool use_dogleg);

  bool _Test_dImuResidual_dX(
      const Pose &pose1, const Pose &pose2,const ImuPose &imu_pose,
      const ImuResidual &res, const Vector3t gravity,
      const Eigen::Matrix<Scalar,7,6>& dse3_dx1,
      const Eigen::Matrix<Scalar,10,6>& dt_db);

  void CalculateGn(const VectorXt& rhs_p, VectorXt& delta_gn);
  void GetLandmarkDelta(
      const VectorXt &delta_p, const VectorXt &rhs_l,
      const BlockMat<Eigen::Matrix<Scalar, kLmDim, kLmDim> > &vi,
      const BlockMat<Eigen::Matrix<Scalar, kLmDim, kPrPoseDim> > &jt_l_j_pr,
      const uint32_t num_poses, const uint32_t num_lm, VectorXt &delta_l);

  void ApplyUpdate(const Delta& delta, const bool bRollback,
                   const Scalar damping = 1.0);
  void EvaluateResiduals(
      double* proj_error = nullptr, double* binary_error = nullptr,
      double* unary_error = nullptr, double* inertial_error = nullptr);
  void BuildProblem();

  ImuCalibration imu_;

  // reprojection jacobians and residual
  BlockMat<Eigen::Matrix<Scalar, ProjectionResidual::kResSize, kPrPoseDim>>
                                                                        j_pr_;
  BlockMat<Eigen::Matrix<Scalar, kPrPoseDim, ProjectionResidual::kResSize>>
                                                                        jt_pr;

  // landmark jacobians
  BlockMat<Eigen::Matrix<Scalar, ProjectionResidual::kResSize, LmSize>> j_l_;
  // BlockMat<Eigen::Matrix<Scalar, LmSize, ProjectionResidual::kResSize>>
  // jt_l_;

  VectorXt r_pr_;

  // pose/pose jacobian for binary constraints
  BlockMat<Eigen::Matrix<Scalar, BinaryResidual::kResSize, kPoseDim>> j_pp_;
  BlockMat<Eigen::Matrix<Scalar, kPoseDim, BinaryResidual::kResSize>> jt_pp_;
  VectorXt r_pp_;

  // pose/pose jacobian for unary constraints
  BlockMat<Eigen::Matrix<Scalar, UnaryResidual::kResSize, kPoseDim>> j_u_;
  BlockMat<Eigen::Matrix<Scalar, kPoseDim, UnaryResidual::kResSize>> jt_u_;
  VectorXt r_u_;

  // imu jacobian
  BlockMat<Eigen::Matrix<Scalar, ImuResidual::kResSize, kPoseDim>> j_i_;
  BlockMat<Eigen::Matrix<Scalar, kPoseDim, ImuResidual::kResSize>> jt_i_;

  BlockMat<Eigen::Matrix<Scalar, ImuResidual::kResSize, CalibSize>> j_ki_;
  BlockMat<Eigen::Matrix<Scalar, CalibSize, ImuResidual::kResSize>> jt_ki_;

  BlockMat<Eigen::Matrix<Scalar, ProjectionResidual::kResSize, CalibSize>>
                                                                        j_kpr_;
  BlockMat<Eigen::Matrix<Scalar, CalibSize, ProjectionResidual::kResSize>>
                                                                        jt_kpr_;

  BlockMat<Eigen::Matrix<Scalar, kPoseDim, kPoseDim>> u_;
  BlockMat<Eigen::Matrix<Scalar, kLmDim, kLmDim>> vi_;
  BlockMat<Eigen::Matrix<Scalar, kLmDim, kPrPoseDim>> jt_l_j_pr_;
  BlockMat<Eigen::Matrix<Scalar, kPrPoseDim, kLmDim>> jt_pr_j_l_;

  VectorXt rhs_p_;
  VectorXt rhs_l_;
  VectorXt r_i_;  
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> s_;
  Eigen::SparseMatrix<Scalar> s_sparse_;
  Scalar trust_region_size_;

  bool translation_enabled_;
  bool is_param_mask_used_; 
  bool do_sparse_solve_;
  bool do_last_pose_cov_;
  double total_tvs_change_;
  SE3t last_tvs_;
  double proj_error_;
  double binary_error_;
  double unary_error_;
  double inertial_error_;
  double tvs_trans_prior_;
  double tvs_rot_prior_;
  unsigned int root_pose_id_;
  unsigned int num_active_poses_;
  unsigned int num_active_landmarks_;
  unsigned int binary_residual_offset_;
  unsigned int unary_residual_offset_;
  unsigned int proj_residual_offset;
  unsigned int inertial_residual_offset_;
  calibu::CameraRigT<Scalar> rig_;
  std::vector<Pose> poses_;
  std::vector<Landmark> landmarks_;
  std::vector<ProjectionResidual > proj_residuals_;
  std::vector<BinaryResidual> binary_residuals_;
  std::vector<UnaryResidual> unary_residuals_;
  std::vector<ImuResidual> inertial_residuals_;
  std::vector<Scalar> errors_;
  Eigen::Matrix<Scalar,kPoseDim+1,kPoseDim+1> last_pose_cov_;

};

static const int NOT_USED = 0;

// typedefs for convenience
template< typename Scalar >
using GlobalInertialBundleAdjuster = BundleAdjuster<Scalar, ba::NOT_USED,15, 2>;
template< typename Scalar >
using InverseDepthVisualInertialBundleAdjuster = BundleAdjuster<Scalar, 1,9>;
template< typename Scalar >
using VisualInertialBundleAdjuster = BundleAdjuster<Scalar, 3,9>;

}





#endif // BUNDLEADUJSTER_H
