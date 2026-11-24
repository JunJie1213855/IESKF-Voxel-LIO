#ifndef LIO_IESKF_H_
#define LIO_IESKF_H_

#include <string>

#include "lio/VoxelMap.h"

namespace lio {
constexpr int kStateDim = 23;
using ErrorVector = Eigen::Matrix<float, kStateDim, 1>;
using Covariance = Eigen::Matrix<float, kStateDim, kStateDim>;
using NoiseCovariance = Eigen::Matrix<float, 12, 12>;
// 名义状态与 23 维误差状态分开：两个旋转各用 3 维右扰动，重力方向仅 2 维。
struct State {
  Vec3 position = Vec3::Zero();
  Mat3 rotation = Mat3::Identity();
  Mat3 extrinsic_rotation = Mat3::Identity();
  Vec3 extrinsic_translation = Vec3::Zero();
  Vec3 velocity = Vec3::Zero();
  Vec3 gyro_bias = Vec3::Zero();
  Vec3 accel_bias = Vec3::Zero();
  // 重力在世界系，模长固定；二维误差表示方向变化（单位 rad）。
  Vec3 gravity = Vec3(0, 0, -9.81f);
};
struct FilterConfig {
  int max_iterations = 4;
  int min_effective_points = 30;
  float rotation_convergence_rad = 0.001f;
  float translation_convergence_m = 0.001f;
  bool rematch_each_iteration = true;
  std::string measurement_model = "constant";
  float plane_residual_std = 0.1f;
  float lidar_range_std = 0.04f;
  float lidar_angle_std_deg = 0.1f;
  float residual_sigma_gate = 3;
  float min_residual_std = 0.01f;
  float max_plane_distance = 0.3f;
  bool estimate_extrinsic = false;
  float gyro_noise_density = 0.01f;
  float accel_noise_density = 0.1f;
  float gyro_bias_random_walk = 0.0001f;
  float accel_bias_random_walk = 0.001f;
  void Validate() const;
};
struct UpdateResult {
  bool accepted = false;
  bool converged = false;
  int iterations = 0;
  int effective_points = 0;
  float mean_abs_residual = 0;
  float pose_information_ratio = 0;
  std::string reason;
};
State IntegrateImu(const State& state, const ImuSample& midpoint, float dt);
Covariance TransitionJacobian(const State& state, const ImuSample& midpoint,
                              float dt);
State BoxPlus(const State& state, const ErrorVector& delta);
ErrorVector BoxMinus(const State& state, const State& reference);
Covariance PriorJacobian(const State& state, const State& reference);
Eigen::Matrix<float, 1, kStateDim> PlaneJacobian(const State& state,
                                                 const Vec3& lidar_point,
                                                 const Vec3& normal);
Vec3 WorldPoint(const State& state, const Vec3& lidar_point);
// 状态与协方差只由估计线程读写，ROS 层通过快照获得结果。
class Ieskf {
 public:
  explicit Ieskf(const FilterConfig& config = {});
  void Reset(const State& state = State(), int64_t timestamp_ns = 0);
  // 单步状态与协方差传播；绝对时间由 ImuProcessing 精确推进。
  bool Predict(const ImuSample& midpoint, float dt);
  // 失败时保持传播先验不变，成功时只提交一次后验。
  UpdateResult Update(const PointCloud& cloud, const VoxelMap& map);
  const State& GetState() const { return state_; }
  const Covariance& GetCovariance() const { return covariance_; }
  int64_t Timestamp() const { return timestamp_ns_; }
  void SetTimestamp(int64_t time) { timestamp_ns_ = time; }
  std::vector<MapPoint> MapPoints(const PointCloud& cloud) const;

 private:
  FilterConfig config_;
  State state_;
  Covariance covariance_;
  int64_t timestamp_ns_ = 0;
};
}  // namespace lio
#endif
