#ifndef BUNDLEADUJSTER_H
#define BUNDLEADUJSTER_H

#include <sophus/se3.hpp>
#include <vector>
#include <Eigen/StdVector>
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
#include <calibu/cam/camera_crtp_interop.h>
// #ifdef ENABLE_TESTING
#include "BundleAdjusterTest.h"
// Only used for matrix square root.
#include <unsupported/Eigen/MatrixFunctions>
// #endif

namespace ba {
template<typename Scalar>
using BlockMat = Eigen::SparseBlockMatrix<Scalar>;

template<typename Scalar>
using aligned_vector = std::vector<Scalar, Eigen::aligned_allocator<Scalar>>;

template<typename Scalar=double>
struct SolutionSummary
{
  uint32_t num_proj_residuals;
  uint32_t num_inertial_residuals;
  uint32_t num_cond_proj_residuals;
  uint32_t num_cond_inertial_residuals;
  Scalar cond_proj_error;
  Scalar cond_inertial_error;
  Scalar proj_error_;
  Scalar inertial_error;

  Scalar delta_norm;
  Scalar pre_solve_norm;
  Scalar post_solve_norm;

  Eigen::MatrixXd calibration_marginals;
};

template<typename Scalar=double>
struct Options
{
  Scalar trust_region_size = 1.0;
  Scalar gyro_sigma = IMU_GYRO_SIGMA;
  Scalar accel_sigma = IMU_ACCEL_SIGMA;
  Scalar gyro_bias_sigma = IMU_GYRO_BIAS_SIGMA;
  Scalar accel_bias_sigma = IMU_ACCEL_BIAS_SIGMA;

  // Outlier thresholds
  Scalar projection_outlier_threshold = 1.0;

  // Exit thresholds
  Scalar error_change_threshold = 0.01;
  Scalar param_change_threshold = 1e-3;

  uint32_t dogleg_max_inner_iterations = 100;
  bool use_dogleg = true;
  bool use_triangular_matrices = true;
  bool use_sparse_solver = true;
  bool write_reduced_camera_matrix = false;
  bool calculate_calibration_marginals = false;

  // Initialization.
  bool regularize_biases_in_batch = true;

  // Robust norms.
  bool use_robust_norm_for_proj_residuals = true;
  bool use_robust_norm_for_inertial_residuals = false;
};



template<typename Scalar=double,int LmSize=1, int PoseSize=6, int CalibSize=0>
class BundleAdjuster
{
public:
  static const uint32_t kPrPoseDim = 6;
  static const uint32_t kLmDim = LmSize;
  static const uint32_t kPoseDim = PoseSize;
  static const uint32_t kCalibDim = CalibSize;

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
    VectorXt delta_k;
    VectorXt delta_l;
  };

  static const bool kVelInState = (kPoseDim >= 9);
  static const bool kBiasInState = (kPoseDim >= 15);
  static const bool kTvsInState = (kPoseDim >= 21);
  static const bool kGravityInCalib = (kCalibDim >= 2);
  static const bool kTvsInCalib = (kCalibDim >= 8);
  static const bool kCamParamsInCalib = (kCalibDim > 0);

  ////////////////////////////////////////////////////////////////////////////
  BundleAdjuster() :
    imu_(SE3t(),Vector3t::Zero(),Vector3t::Zero(),Vector2t::Zero()),
    translation_enabled_(kCalibDim > 15 ? false : true),
    total_tvs_change_(0)
  {}


  ////////////////////////////////////////////////////////////////////////////
  void Init(const Options<Scalar>& options,
            uint32_t num_poses = 0,
            uint32_t num_measurements = 0,
            uint32_t num_landmarks = 0,
            const SE3t& t_vs = SE3t())
  {
    // If any of these are zero, they will destroy some of the predictive maths
    // further ahead, so set them to a default value.
    num_poses = std::max(1u, num_poses);
    num_measurements = std::max(1u, num_measurements);
    num_landmarks = std::max(1u, num_landmarks);

    // if LmSize == 0, there is no need for a camera rig or landmarks
    assert(num_landmarks != 0 || LmSize == 0);

    options_ = options;
    trust_region_size_ = options_.trust_region_size;
    root_pose_id_ = 0;
    num_active_poses_ = 0;
    num_active_landmarks_ = 0;
    proj_residual_offset = 0;
    binary_residual_offset_ = 0;
    unary_residual_offset_ = 0;
    inertial_residual_offset_ = 0;

    imu_.t_vs = t_vs;
    last_tvs_ = imu_.t_vs;
    imu_.r = ((Eigen::Matrix<Scalar, 6, 1>() <<
              powi(options.gyro_sigma, 2),
              powi(options.gyro_sigma, 2),
              powi(options.gyro_sigma, 2),
              powi(options.accel_sigma, 2),
              powi(options.accel_sigma, 2),
              powi(options.accel_sigma, 2)).finished().asDiagonal());

    imu_.r_b <<
              powi(options.gyro_bias_sigma, 2),
              powi(options.gyro_bias_sigma, 2),
              powi(options.gyro_bias_sigma, 2),
              powi(options.accel_bias_sigma, 2),
              powi(options.accel_bias_sigma, 2),
              powi(options.accel_bias_sigma, 2);

    landmarks_.reserve(num_landmarks);
    proj_residuals_.reserve(num_measurements);
    poses_.reserve(num_poses);

    // Delete the cameras that we own.
    for (size_t ii = 0; ii < rig_.cameras_.size() ; ++ii) {
      if (camera_owned_[ii] == true) {
        delete rig_.cameras_[ii];
      }
    }
    rig_.cameras_.clear();

    // clear all arrays
    rig_.Clear();
    poses_.clear();
    proj_residuals_.clear();
    binary_residuals_.clear();
    unary_residuals_.clear();
    inertial_residuals_.clear();
    landmarks_.clear();

    conditioning_inertial_residuals_.clear();
    conditioning_proj_residuals_.clear();
  }

  ////////////////////////////////////////////////////////////////////////////
  /// \brief Sets the direction and magnitude of gravity.
  /// \param g the 3d gravity vector.
  ///
  void SetGravity(const Vector3t& g){
    if (kGravityInCalib) {
      const Vector3t new_g_norm = g.normalized();
      const Scalar p = asin(new_g_norm[1]);
      const Scalar q = acos(std::min(1.0,std::max(-1.0,-new_g_norm[2]/cos(p))));
      imu_.g << p, q;
    }else {
      imu_.g_vec = g;
    }
  }


  ////////////////////////////////////////////////////////////////////////////
  uint32_t AddCamera(const calibu::CameraModelInterfaceT<Scalar>& cam_param,
                     const SE3t& cam_pose)
  {
    rig_.AddCamera(calibu::CreateFromOldCamera<Scalar>(cam_param), cam_pose);
    // Signal that we own this camera so we delete it.
    camera_owned_.resize(rig_.cameras_.size());
    camera_owned_.back() = true;
    return rig_.cameras_.size()-1;
  }

  ////////////////////////////////////////////////////////////////////////////
  uint32_t AddCamera(calibu::CameraInterface<Scalar>* cam,
                     const SE3t& cam_pose)
  {
    rig_.AddCamera(cam, cam_pose);
    // Signal that we don't own this camera so we don't delete it.
    camera_owned_.resize(rig_.cameras_.size());
    camera_owned_.back() = false;
    return rig_.cameras_.size();
  }


  ////////////////////////////////////////////////////////////////////////////
  uint32_t AddPose(const SE3t& t_wp,
                       const bool is_active = true,
                       const double time = -1,
                       const int external_id = -1)
  {
    return AddPose(t_wp, SE3t(), VectorXt(5).setZero(),
                   Vector3t::Zero(), Vector6t::Zero(), is_active, time,
                   external_id);
  }

  ////////////////////////////////////////////////////////////////////////////
  /// \brief Adds a pose to the optimization, the id of the pose can later be
  /// used to form constraints.
  /// \param t_wv is the SE3 pose of the vejhicle
  /// \param t_vs is the vehicle to sensor extrinsics calibration (if required)
  /// which can be set to identity if not needed.
  /// \param cam_params is the vector of camera intrinsics used in calibration
  /// which can be empty if not needed.
  /// \param v_w is the 3D velocity vector.
  /// \param b is the 6d gyro/imu bias vector.
  /// \param is_active defines whether or not this pose is active in the
  /// optimization.
  /// \param time timestamp for this pose
  /// \param external_id external id used for bookkeeping outside of ba
  /// \return the internal optimization id used to form constraints
  ///
  uint32_t AddPose(const SE3t& t_wv, const SE3t& t_vs,
                       const VectorXt cam_params, const Vector3t& v_w,
                       const Vector6t& b, const bool is_active = true,
                       const double time = -1, const int external_id = -1)
  {
    Pose pose;
    pose.external_id = external_id;
    pose.time = time;
    pose.t_wp = t_wv;
    pose.t_vs = t_vs;
    pose.v_w = v_w;
    pose.b = b;
    pose.cam_params = cam_params;
    pose.is_active = is_active;
    pose.is_param_mask_used = false;
    pose.t_sw.reserve(rig_.cameras_.size());

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
  uint32_t AddLandmark(const Vector4t& x_w,
                           const uint32_t ref_pose_id,
                           const uint32_t ref_cam_id,
                           const bool is_active,
                           const int external_id = -1)
  {
    assert(ref_pose_id < poses_.size());
    Landmark landmark;
    landmark.external_id = external_id;
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
  /// \brief Adds a unary constraint to a pose. This unary constraint is
  /// calcualted as log(t_wv * t_wv'.inverse())
  /// \param pose_id of the pose to which this constraint applies
  /// \param t_wv is the world to vehicle transform that forms this constraint
  /// \param covariance the 6x6 covariance in the residual space
  /// \return
  ///
  uint32_t AddUnaryConstraint(const uint32_t pose_id,
                                  const SE3t& t_wv,
                                  Eigen::Matrix<Scalar, UnaryResidual::kResSize,
                                  UnaryResidual::kResSize> covariance)
  {
    assert(pose_id < poses_.size());

    //now add this constraint to pose A
    UnaryResidual residual;
    residual.orig_weight = 1.0;
    residual.pose_id = pose_id;
    residual.residual_id = unary_residuals_.size();
    residual.residual_offset = unary_residual_offset_;
    residual.t_wp = t_wv;
    residual.cov_inv = covariance.inverse();
    residual.cov_inv_sqrt = residual.cov_inv.sqrt();

    unary_residuals_.push_back(residual);
    unary_residual_offset_ += UnaryResidual::kResSize;

    // we add this to both poses, as each one has a jacobian cell associated
    poses_[pose_id].unary_residuals.push_back(residual.residual_id);
    return residual.residual_id;
  }

  ////////////////////////////////////////////////////////////////////////////
  uint32_t AddBinaryConstraint(const uint32_t pose1_id,
                               const uint32_t pose2_id,
                               const SE3t& t_12,
                               Scalar weight = 1.0)
  {
    const Eigen::Matrix<Scalar, BinaryResidual::kResSize,
      UnaryResidual::kResSize> covariance =
        Eigen::Matrix<Scalar, BinaryResidual::kResSize,
              UnaryResidual::kResSize>::Identity();
    return AddBinaryConstraint(pose1_id, pose2_id, t_12, covariance, weight);
  }

  ////////////////////////////////////////////////////////////////////////////
  uint32_t AddBinaryConstraint(const uint32_t pose1_id,
                               const uint32_t pose2_id,
                               const SE3t& t_12,
                               Eigen::Matrix<Scalar, BinaryResidual::kResSize,
                               UnaryResidual::kResSize> covariance,
                               Scalar weight = 1.0)
  {
    assert(pose1_id < poses_.size());
    assert(pose2_id < poses_.size());

    //now add this constraint to pose A
    BinaryResidual residual;
    residual.orig_weight = weight;
    residual.x1_id = pose1_id;
    residual.x2_id = pose2_id;
    residual.residual_id = binary_residuals_.size();
    residual.residual_offset = binary_residual_offset_;
    residual.t_12 = t_12;
    residual.cov_inv = covariance.inverse();
    residual.cov_inv_sqrt = residual.cov_inv.sqrt();

    binary_residuals_.push_back(residual);
    binary_residual_offset_ += BinaryResidual::kResSize;

    // we add this to both poses, as each one has a jacobian cell associated
    poses_[pose1_id].binary_residuals.push_back(residual.residual_id);
    poses_[pose2_id].binary_residuals.push_back(residual.residual_id);

    return residual.residual_id;
  }

  ////////////////////////////////////////////////////////////////////////////
  uint32_t AddProjectionResidual(const Vector2t z,
                                     const uint32_t meas_pose_id,
                                     const uint32_t landmark_id,
                                     const uint32_t cam_id,
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
    const uint32_t res_id = residual.residual_id;
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

    if (poses_[residual.x_ref_id].is_active == false &&
        poses_[residual.x_meas_id].is_active == true) {
      conditioning_proj_residuals_.push_back(residual.residual_id);
    }

    return residual.residual_id;
  }

  ////////////////////////////////////////////////////////////////////////////
  uint32_t AddImuResidual(const uint32_t pose1_id,
                              const uint32_t pose2_id,
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

    if (poses_[pose1_id].is_active == false &&
        poses_[pose2_id].is_active == true) {
      conditioning_inertial_residuals_.push_back(residual.residual_id);
    }

    return residual.residual_id;
  }

  void Solve(const uint32_t uMaxIter,
             const Scalar gn_damping = 1.0,
             const bool error_increase_allowed = false);

  void SetRootPoseId(const uint32_t id) { root_pose_id_ = id; }

  bool IsTranslationEnabled() { return translation_enabled_; }
  uint32_t GetNumPoses() const { return poses_.size(); }
  uint32_t GetNumImuResiduals() const { return inertial_residuals_.size(); }
  uint32_t GetNumProjResiduals() const { return proj_residuals_.size(); }

  const ImuResidual& GetImuResidual(const uint32_t id)
  const { return inertial_residuals_[id]; }

  const ImuCalibration& GetImuCalibration() const { return imu_; }
  void SetImuCalibration(const ImuCalibration& calib) { imu_ = calib; }
  const ProjectionResidual& GetProjectionResidual(uint32_t id) const
  {
    return proj_residuals_[id];
  }

  const Pose& GetPose(const uint32_t id) const
  {
    if (id >= poses_.size()) {
      std::cerr << "Attempted to get pose with id " << id << " from BA. "
                << " when poses_.size() is only " << poses_.size()
                << "Aborting..." << std::endl;
      throw 0;
    }
    return poses_[id];
  }

  // return the landmark in the world frame
  const Landmark& GetLandmarkObj(const uint32_t id) const
    { return landmarks_[id]; }
  const Vector4t& GetLandmark(const uint32_t id) const
    { return landmarks_[id].x_w; }
  bool IsLandmarkReliable(const uint32_t id) const
  { return landmarks_[id].is_reliable; }
  double LandmarkOutlierRatio(const uint32_t id) const;

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

  const SolutionSummary<Scalar>& GetSolutionSummary() { return summary_; }
  Options<Scalar>& options() { return options_; }
  const calibu::Rig<Scalar>& rig() { return rig_; }

private:
  bool SolveInternal(VectorXt rhs_p_sc, const Scalar gn_damping,
                     const bool error_increase_allowed, const bool use_dogleg);

  void CalculateGn(const VectorXt& rhs_p, Delta &delta);
  void GetLandmarkDelta(const Delta& delta, const uint32_t num_poses,
                        const uint32_t num_lm, VectorXt &delta_l);

  void ApplyUpdate(const Delta& delta, const bool bRollback,
                   const Scalar damping = 1.0);
  void EvaluateResiduals(
      Scalar* proj_error = nullptr, Scalar* binary_error = nullptr,
      Scalar* unary_error = nullptr, Scalar* inertial_error = nullptr);
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

  VectorXt r_i_;
  BlockMat<Eigen::Matrix<Scalar, ImuResidual::kResSize, CalibSize>> j_ki_;
  BlockMat<Eigen::Matrix<Scalar, CalibSize, ImuResidual::kResSize>> jt_ki_;

  BlockMat<Eigen::Matrix<Scalar, ProjectionResidual::kResSize, CalibSize>>
                                                                        j_kpr_;
  BlockMat<Eigen::Matrix<Scalar, CalibSize, ProjectionResidual::kResSize>>
                                                                        jt_kpr_;

  BlockMat<Eigen::Matrix<Scalar, kPoseDim, kPoseDim>> u_;
  BlockMat<Eigen::Matrix<Scalar, kLmDim, kLmDim>> vi_;
  BlockMat<Eigen::Matrix<Scalar, kLmDim, kPrPoseDim>> jt_l_j_pr_;
  BlockMat< Eigen::Matrix<Scalar, kLmDim, CalibSize>> jt_l_j_kpr_;
  BlockMat<Eigen::Matrix<Scalar, kPrPoseDim, kLmDim>> jt_pr_j_l_;


  VectorXt rhs_p_;
  VectorXt rhs_k_;
  VectorXt rhs_l_;
  VectorXt r_pi_;

  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> s_;
  Eigen::SparseMatrix<Scalar> s_sparse_;
  Scalar trust_region_size_;

  bool translation_enabled_;
  bool is_param_mask_used_;
  Scalar total_tvs_change_;
  SE3t last_tvs_;
  Scalar proj_error_;
  Scalar binary_error_;
  Scalar unary_error_;
  Scalar inertial_error_;
  uint32_t root_pose_id_;
  uint32_t num_active_poses_;
  uint32_t num_active_landmarks_;
  uint32_t binary_residual_offset_;
  uint32_t unary_residual_offset_;
  uint32_t proj_residual_offset;
  uint32_t inertial_residual_offset_;
  // calibu::CameraRigT<Scalar> rig_;
  calibu::Rig<Scalar> rig_;
  std::vector<bool> camera_owned_;
  std::vector<Pose> poses_;
  std::vector<Landmark> landmarks_;
  std::vector<ProjectionResidual > proj_residuals_;
  std::vector<uint32_t> conditioning_proj_residuals_;
  std::vector<uint32_t> conditioning_inertial_residuals_;
  std::vector<BinaryResidual> binary_residuals_;
  std::vector<UnaryResidual> unary_residuals_;
  std::vector<ImuResidual> inertial_residuals_;
  std::vector<Scalar> errors_;
  Eigen::Matrix<Scalar,kPoseDim+1,kPoseDim+1> last_pose_cov_;

  SolutionSummary<Scalar> summary_;
  Options<Scalar> options_;
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
