/*
 This file is part of the BA Project.

 Copyright (C) 2013 George Washington University,
 Nima Keivan,
 Steven Lovegrove,
 Gabe Sibley

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

#ifndef VICALIBRATOR_H
#define VICALIBRATOR_H

// Compile against new ceres covarience interface.
//#define CALIBU_CERES_COVAR

#include <thread>
#include <mutex>
#include <memory>
#include <system_error>

#include <sophus/se3.hpp>

#include <calibu/cam/camera_crtp.h>
#include <calibu/cam/camera_xml.h>
#include <calibu/calib/CostFunctionAndParams.h>

#include <ceres/ceres.h>

#ifdef CALIBU_CERES_COVAR
#include <ceres/covariance.h>
#endif // CALIBU_CERES_COVAR

#include "LocalParamSe3.h"
#include <calibu/cam/camera_models_crtp.h>
#include <calibu/cam/camera_rig.h>
#include <calibu/calib/ReprojectionCostFunctor.h>
#include <calibu/calib/CostFunctionAndParams.h>

#include "Types.h"
#include "InterpolationBuffer.h"
#include "CeresCostFunctions.h"

namespace ba {

// Tie together a single camera and its position relative to the IMU.
struct CameraAndPose {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  CameraAndPose(const std::shared_ptr<calibu::CameraInterface<double>> camera,
                const Sophus::SE3d& T_ck)
      : camera(camera), T_ck(T_ck) {}

  std::shared_ptr<calibu::CameraInterface<double>> camera;
  Sophus::SE3d T_ck;
};

class ViCalibrator {
 public:

  /// Construct empty calibration object.
  ViCalibrator()
      : is_running_(false),
        fix_intrinsics_(false),
        loss_func_(new ceres::SoftLOneLoss(0.5), ceres::TAKE_OWNERSHIP),
        imu_(Sophus::SE3d(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
             Eigen::Vector2d::Zero()),
        imu_loss_func_(0.2),
        num_imu_residuals_(0) {
    prob_options_.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    prob_options_.local_parameterization_ownership =
        ceres::DO_NOT_TAKE_OWNERSHIP;
    prob_options_.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;

    solver_options_.num_threads = 4;
    solver_options_.update_state_every_iteration = true;
    solver_options_.max_num_iterations = 10;

    Clear();
  }

  ~ViCalibrator() {
  }

  /// Write XML file containing configuration of camera rig.
  void WriteCameraModels(const std::string filename) {
    std::shared_ptr<calibu::Rig<double>> rig(new calibu::Rig<double>);

    for (size_t c = 0; c < camera_.size(); ++c) {
      // Rdfrobotics.inverse is multiplied so that T_ck d oes not bake the
      // robotics (imu) to vision coordinate transform d
      rig.Add(
          camera_[c]->camera,
          camera_[c]->T_ck.inverse()
              * Sophus::SE3d(calibu::RdfRobotics.inverse(),
                             Eigen::Vector3d::Zero()));
    }

    WriteXmlRig(filename, rig);
  }

  /// Clear all cameras / constraints
  void Clear() {
    Stop();
    t_wk_.clear();
    camera_.clear();
    proj_costs_.clear();
    imu_costs_.clear();
    mse_ = 0;
    num_imu_residuals_ = 0;
    is_bias_active_ = true;
    is_inertial_active_ = true;
    is_visual_active_ = true;
    optimize_rotation_only_ = true;
  }

  /// Start optimisation thread to modify intrinsic / extrinsic parameters
  void Start() {
    if (!is_running_) {
      should_run = true;
      thread_ = std::thread(std::bind(&ViCalibrator::SolveThread, this));
    } else {
      std::cerr << "Already Running." << std::endl;
    }
  }

  /// Stop optimisation thread
  void Stop() {
    if (is_running_) {
      should_run = false;
      try {
        thread_.join();
      } catch (std::system_error) {
        // thread already died.
      }
    }
  }

  /// Add camera to sensor rig. The returned ID should be used when adding
  /// measurements for this camera
  int AddCamera(const std::shared_ptr<calibu::CameraInterface<double>> cam,
                const Sophus::SE3d& t_ck = Sophus::SE3d()) {
    int id = camera_.size();
    camera_.push_back(
        calibu::make_unique < CameraAndPose > (cam, t_ck));
    camera_.back()->camera->SetIndex(id);
    return id;
  }

  /// Set whether intrinsics should be 'fixed' and left unchanged by the
  /// minimization.
  void FixCameraIntrinsics(bool v = true) {
    fix_intrinsics_ = v;
  }

  /// Add frame to optimiser. The returned ID should be used when adding
  /// target measurements for a given moment in time. Measurements given
  /// for any camera for a given frame are assumed to be simultaneous, with
  /// camera extrinsics equal between all cameras for each frame.
  int AddFrame(const Sophus::SE3d t_wk, const double time) {
    update_mutex_.lock();
    int id = t_wk_.size();
    ba::ImuPoseT<double> pose(t_wk, Eigen::Vector3d::Zero(),
                              Eigen::Vector3d::Zero(), time);

    t_wk_.push_back(calibu::make_unique<ba::ImuPoseT<double>>(pose));
    update_mutex_.unlock();

    return id;
  }

  /// Add imu measurements
  void AddImuMeasurements(const Eigen::Vector3d& gyro,
                          const Eigen::Vector3d& accel, const double time) {
    imu_buffer_.AddElement(ba::ImuMeasurementT<double>(gyro, accel, time));
  }

  /// Add observation p_c of 3D feature P_w from 'camera' for 'frame'
  /// 'camera' and 'frame' id's can be obtained by calls to AddCamera and
  /// AddFrame respectively.
  void AddObservation(size_t frame, size_t camera_id,
                      const Eigen::Vector3d& p_w, const Eigen::Vector2d& p_c,
                      const double time) {
    update_mutex_.lock();

    // Ensure index is valid
    while (NumFrames() < frame) {
      AddFrame(Sophus::SE3d(), time);
    }

    if (NumCameras() < camera_id) {
      throw std::runtime_error("Bad camera index. Add all cameras first.");
    }

    // new camera pose to bundle adjust

    calibu::CameraAndPose& cp = *camera_[camera_id];
    Sophus::SE3d& t_wk = t_wk_[frame]->t_wp;

    // Create cost function
    calibu::CostFunctionAndParams* cost = new calibu::CostFunctionAndParams();

    std::shared_ptr<calibu::CameraInterface<double>> interface = cp.camera;

    // Allocate and assign the correct cost function. Lifetimes are
    // handled by Calibu.
    if (dynamic_cast<calibu::FovCamera<double>*>( interface.get())) {  // NOLINT
      cost->Cost() = new ceres::AutoDiffCostFunction<
          ImuReprojectionCostFunctor<calibu::FovCamera<double>>, 2,
          Sophus::SE3d::num_parameters, Sophus::SO3d::num_parameters, 3,
          calibu::FovCamera<double>::NumParams>(
          new ImuReprojectionCostFunctor<calibu::FovCamera<double>>(p_w, p_c));

    } else if (dynamic_cast<calibu::Poly2Camera<double>*>(
                   interface.get())) {  // NOLINT
      cost->Cost() = new ceres::AutoDiffCostFunction<
          ImuReprojectionCostFunctor<calibu::Poly2Camera<double>>, 2,
          Sophus::SE3d::num_parameters, Sophus::SO3d::num_parameters, 3,
          calibu::Poly2Camera<double>::NumParams>(
          new ImuReprojectionCostFunctor<calibu::Poly2Camera<double>>(p_w, p_c));

    } else if (dynamic_cast<calibu::Poly3Camera<double>*>( interface.get())) {  // NOLINT
      cost->Cost() = new ceres::AutoDiffCostFunction<
          ImuReprojectionCostFunctor<calibu::Poly3Camera<double>>, 2,
          Sophus::SE3d::num_parameters, Sophus::SO3d::num_parameters, 3,
          calibu::Poly3Camera<double>::NumParams>(
          new ImuReprojectionCostFunctor<calibu::Poly3Camera<double>>(p_w, p_c));

    } else if (dynamic_cast<calibu::KannalaBrandtCamera<double>*>( interface.get())) {
      cost->Cost() = new ceres::AutoDiffCostFunction<
          ImuReprojectionCostFunctor<calibu::KannalaBrandtCamera<double>>, 2,
          Sophus::SE3d::num_parameters, Sophus::SO3d::num_parameters, 3,
          calibu::KannalaBrandtCamera<double>::NumParams>(
          new ImuReprojectionCostFunctor<calibu::KannalaBrandtCamera<double>>(p_w, p_c));
    } else if (dynamic_cast<calibu::LinearCamera<double>*>( interface.get())) {
      cost->Cost() = new ceres::AutoDiffCostFunction<
          ImuReprojectionCostFunctor<calibu::LinearCamera<double>>, 2,
          Sophus::SE3d::num_parameters, Sophus::SO3d::num_parameters, 3,
          calibu::LinearCamera<double>::NumParams>(
          new ImuReprojectionCostFunctor<calibu::LinearCamera<double>>(p_w, p_c));

    } else {
      LOG(FATAL) << "Don't know how to optimize CameraModel: "
                 << interface->Type();
    }

    cost->Params() = std::vector<double*> { t_wk.data(), cp.T_ck.so3().data(),
        cp.T_ck.translation().data(), cp.camera->GetParams().data() };
    cost->Loss() = &loss_func_;
    proj_costs_.push_back(
        std::unique_ptr < calibu::CostFunctionAndParams > (cost));

    update_mutex_.unlock();
  }

  /// Return number of synchronised camera rig frames
  size_t NumFrames() const {
    return t_wk_.size();
  }

  /// Return pose of camera rig frame i
  Sophus::SE3d& GetFrame(size_t i) {
    return t_wk_[i]->t_wp;
  }

  /// Return number of cameras in camera rig
  size_t NumCameras() const {
    return camera_.size();
  }

  /// Return camera i of camera rig
  CameraAndPose& GetCamera(size_t i) {
    return *camera_[i];
  }

  /// Return current Mean Square reprojection Error - the objective function
  /// being minimised by optimisation.
  double MeanSquareError() const {
    return mse_;
  }

  //////////////////////////////////////////////////////////////////////////////
  const std::vector<ba::ImuPoseT<double>> GetIntegrationPoses(
      const unsigned int id) {
    std::vector<ba::ImuPoseT<double> > poses;
    if (id >= 0 && id <= t_wk_.size() - 1) {
      const ba::ImuPoseT<double>& prev_pose = *(t_wk_[id]);
      const ba::ImuPoseT<double>& pose = *(t_wk_[id + 1]);

      // get all the imu measurements between the two poses
      std::vector<ba::ImuMeasurementT<double>> measurements = imu_buffer_
          .GetRange(prev_pose.time, pose.time);

      if (measurements.empty() == false) {
        ba::ImuResidualT<double>::IntegrateResidual(
            prev_pose, measurements, imu_.b_g, imu_.b_a,
            ba::GetGravityVector<double>(imu_.g), poses);
      }
    }
    return poses;
  }

  /// Print summary of calibration
  void PrintResults() {
    LOG(INFO) << "------------------------------------------" << std::endl;
    for (size_t c = 0; c < cameras_.size(); ++c) {
      LOG(INFO) << "Camera: " << c << std::endl;
      LOG(INFO) << cameras_[c]->camera->GetParams().transpose() << std::endl;
      LOG(INFO) << cameras_[c]->T_ck.matrix();
      LOG(INFO) << std::endl;
    }
  }

 protected:

  void SetupProblem(ceres::Problem& problem) {
    update_mutex_.lock();

    // Add parameters
    for (size_t c = 0; c < camera_.size(); ++c) {
      problem.AddParameterBlock(camera_[c]->T_ck.so3().data(), 4,
                                &local_param_so3_);

      problem.AddParameterBlock(camera_[c]->T_ck.translation().data(), 3);
      // we don't do this anymore due to inertial constraints
      if (c == 0) {
        if (is_inertial_active_ == false) {
          problem.SetParameterBlockConstant(camera_[c]->T_ck.so3().data());
          problem.SetParameterBlockConstant(
              camera_[c]->T_ck.translation().data());
        } else {
          problem.SetParameterBlockVariable(camera_[c]->T_ck.so3().data());
          if (optimize_rotation_only_ == true) {
            problem.SetParameterBlockConstant(
                camera_[c]->T_ck.translation().data());
          } else {
            problem.SetParameterBlockVariable(
                camera_[c]->T_ck.translation().data());
          }
        }
      }

      if (fix_intrinsics_) {
        problem.AddParameterBlock(camera_[c]->camera->GetParams().data(),
                                  camera_[c]->camera->NumParams());

        problem.SetParameterBlockConstant(camera_[c]->camera->GetParams().data());
      }
    }

    for (size_t p = 0; p < t_wk_.size(); ++p) {
      problem.AddParameterBlock(t_wk_[p]->t_wp.data(), 7, &local_param_se3_);

      // add an imu cost residual if we have not yet for this frame
      if (p >= num_imu_residuals_) {

        // get all the imu measurements between these two poses, and add
        // them to a vector
        if (p > 0) {
          std::vector<ba::ImuMeasurementT<double>> measurements = imu_buffer_
              .GetRange(t_wk_[p - 1]->time, t_wk_[p]->time);

          if (measurements.empty() == false) {
            calibu::CostFunctionAndParams* cost =
                new calibu::CostFunctionAndParams();

            cost->Cost() =
                new ceres::AutoDiffCostFunction<
                    ba::SwitchedFullImuCostFunction<double>, 9, 7, 7, 3, 3, 2,
                    3, 3>(
                    new ba::SwitchedFullImuCostFunction<double>(
                        measurements, 500.0, &optimize_rotation_only_));

            cost->Loss() = NULL;

            cost->Params() = std::vector<double*> { t_wk_[p]->t_wp.data(),
                t_wk_[p - 1]->t_wp.data(), t_wk_[p]->v_w.data(), t_wk_[p - 1]
                    ->v_w.data(), imu_.g.data(), imu_.b_g.data(),
                imu_.b_a.data() };
            imu_costs_.push_back(
                std::unique_ptr < calibu::CostFunctionAndParams > (cost));
          }
        }
        num_imu_residuals_++;
      }
    }

    // Add costs
    if (is_visual_active_ == true) {
      for (size_t c = 0; c < proj_costs_.size(); ++c) {
        calibu::CostFunctionAndParams& cost = *proj_costs_[c];
        problem.AddResidualBlock(cost.Cost(), cost.Loss(), cost.Params());
      }
    }

    if (is_inertial_active_ == true) {
      for (size_t c = 0; c < imu_costs_.size(); ++c) {
        calibu::CostFunctionAndParams& cost = *imu_costs_[c];
        problem.AddResidualBlock(cost.Cost(), cost.Loss(), cost.Params());
      }

      // only do this once
      if (is_bias_active_ == false) {
        std::cout << "Setting bias terms to constant... " << std::endl;
        problem.SetParameterBlockConstant(imu_.b_g.data());
        problem.SetParameterBlockConstant(imu_.b_a.data());
      }
    }
    update_mutex_.unlock();
  }

  void SolveThread() {
    is_running_ = true;
    while (should_run) {
      ceres::Problem problem(prob_options_);
      SetupProblem(problem);

      // Crank optimisation
      if (problem.NumResiduals() > 0) {
        try {
          ceres::Solver::Summary summary;
          ceres::Solve(solver_options_, &problem, &summary);
          std::cout << summary.BriefReport() << std::endl;
          mse_ = summary.final_cost / summary.num_residuals;
          if (summary.termination_type != ceres::NO_CONVERGENCE) {
            if (is_inertial_active_ == false) {
              is_inertial_active_ = true;
            } else {
              if (optimize_rotation_only_ == true) {
                std::cout << "Finished optimizing rotations. Activating T_ck "
                    "translation optimization..." << std::endl;
                optimize_rotation_only_ = false;
              } else {
                is_bias_active_ = true;
                std::cout << "Activating bias terms... " << std::endl;
                problem.SetParameterBlockVariable(imu_.b_g.data());
                problem.SetParameterBlockVariable(imu_.b_a.data());
              }
            }
          }
          std::cout << "Frames: " << t_wk_.size() << "; Observations: "
              << summary.num_residuals << "; mse: " << mse_ << std::endl;

          std::cout << "Bg= " << imu_.b_g.transpose() << std::endl;
          std::cout << "Ba= " << imu_.b_a.transpose() << std::endl;
          std::cout << "G= " << imu_.g.transpose() << std::endl;
        } catch (std::exception e) {
          std::cerr << e.what() << std::endl;
        }
      }
    }
    is_running_ = false;
  }

  std::mutex update_mutex_;
  std::thread thread_;
  bool should_run;
  bool is_running_;

  bool fix_intrinsics_;

  std::vector<std::unique_ptr<ba::ImuPoseT<double>> > t_wk_;
  std::vector<std::unique_ptr<CameraAndPose> > camera_;
  std::vector<std::unique_ptr<calibu::CostFunctionAndParams> > proj_costs_;
  std::vector<std::unique_ptr<calibu::CostFunctionAndParams> > imu_costs_;

  ceres::Problem::Options prob_options_;
  ceres::Solver::Options solver_options_;

  ceres::LossFunctionWrapper loss_func_;
  LocalParamSe3 local_param_se3_;
  LocalParamSo3 local_param_so3_;
  ba::InterpolationBufferT<ba::ImuMeasurementT<double>, double> imu_buffer_;
  ba::ImuCalibrationT<double> imu_;
  ceres::CauchyLoss imu_loss_func_;
  double imu_cauchy_norm_;
  unsigned int num_imu_residuals_;
  bool is_bias_active_;
  bool is_inertial_active_;
  bool is_visual_active_;
  bool optimize_rotation_only_;

  double mse_;
};

}

#endif

