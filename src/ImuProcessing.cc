#include "lio/ImuProcessing.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <Eigen/Geometry>
namespace lio {
void ImuConfig::Validate() const {
  for (float v : {gravity_magnitude, init_duration_sec, init_max_gyro_std,
                  init_max_accel_std, init_max_gyro_mean, max_gap_sec})
    if (!std::isfinite(v) || v <= 0)
      throw std::invalid_argument("invalid IMU scale");
  if (max_gap_sec > 0.1f || init_min_samples < 2 ||
      !gravity_direction.allFinite() || gravity_direction.norm() < 1e-6f ||
      !extrinsic_translation.allFinite() || !extrinsic_rotation.allFinite() ||
      !(extrinsic_rotation.transpose() * extrinsic_rotation)
           .isApprox(Mat3::Identity(), 1e-4f) ||
      std::abs(extrinsic_rotation.determinant() - 1) > 1e-4f)
    throw std::invalid_argument("invalid IMU/extrinsic parameters");
}
ImuProcessing::ImuProcessing(const ImuConfig& c) : config_(c) { c.Validate(); }
void ImuProcessing::Reset() {
  count_ = 0;
  last_init_ns_ = first_init_ns_ = -1;
  mean_acc_.setZero();
  mean_gyro_.setZero();
  m2_acc_.setZero();
  m2_gyro_.setZero();
  trajectory_.clear();
}
bool ImuProcessing::Initialize(const MeasurementGroup& m, Ieskf* filter) {
  // 同步组含重复的边界样本，用 last_init_ns_ 去重，避免初始化重复统计。
  for (const auto& sample : m.imu) {
    if (sample.timestamp_ns <= last_init_ns_ ||
        sample.timestamp_ns > m.scan.end_ns)
      continue;
    if (last_init_ns_ >= 0 &&
        SecondsBetween(sample.timestamp_ns, last_init_ns_) >
            config_.max_gap_sec)
      Reset();
    if (sample.gyro.norm() > config_.init_max_gyro_mean ||
        std::abs(sample.accel.norm() - config_.gravity_magnitude) >
            0.2f * config_.gravity_magnitude) {
      Reset();
      last_init_ns_ = sample.timestamp_ns;
      continue;
    }
    if (count_ == 0) first_init_ns_ = sample.timestamp_ns;
    last_init_ns_ = sample.timestamp_ns;
    ++count_;
    const Vec3 da = sample.accel - mean_acc_, dg = sample.gyro - mean_gyro_;
    mean_acc_ += da / static_cast<float>(count_);
    mean_gyro_ += dg / static_cast<float>(count_);
    m2_acc_ += da.cwiseProduct(sample.accel - mean_acc_);
    m2_gyro_ += dg.cwiseProduct(sample.gyro - mean_gyro_);
    if (count_ >= 10 &&
        ((m2_acc_ / (count_ - 1)).cwiseMax(0).cwiseSqrt().maxCoeff() >
             config_.init_max_accel_std ||
         (m2_gyro_ / (count_ - 1)).cwiseMax(0).cwiseSqrt().maxCoeff() >
             config_.init_max_gyro_std)) {
      const int64_t last = last_init_ns_;
      Reset();
      last_init_ns_ = last;
    }
  }
  if (count_ < config_.init_min_samples ||
      SecondsBetween(last_init_ns_, first_init_ns_) < config_.init_duration_sec)
    return false;
  // 静止比力与重力反向；显式设定零航向规范，避免倒置 IMU
  // 近反平行时航向任意跳变。
  State s;
  s.gravity =
      config_.gravity_direction.normalized() * config_.gravity_magnitude;
  const Vec3 gravity_body = -mean_acc_.normalized();
  const float roll = std::atan2(-gravity_body.y(), -gravity_body.z());
  const float pitch = std::asin(std::clamp(gravity_body.x(), -1.0f, 1.0f));
  const Mat3 reference =
      Eigen::Quaternionf::FromTwoVectors(-Vec3::UnitZ(), s.gravity.normalized())
          .toRotationMatrix();
  s.rotation = reference * ExpSO3(Vec3(0, pitch, 0)) * ExpSO3(Vec3(roll, 0, 0));
  s.gyro_bias = mean_gyro_;
  s.extrinsic_rotation = config_.extrinsic_rotation;
  s.extrinsic_translation = config_.extrinsic_translation;
  filter->Reset(s, m.scan.end_ns);
  return true;
}
bool ImuProcessing::Propagate(const MeasurementGroup& m, Ieskf* filter) {
  trajectory_.clear();
  const int64_t begin = filter->Timestamp(), end = m.scan.end_ns;
  if (m.imu.size() < 2 || begin > m.scan.start_ns || begin >= end ||
      m.imu.front().timestamp_ns > begin || m.imu.back().timestamp_ns < end)
    return false;
  // 在修改滤波器之前检查整段覆盖，避免遇到 IMU 缺口后留下半帧传播结果。
  for (size_t i = 1; i < m.imu.size(); ++i) {
    if (m.imu[i].timestamp_ns <= m.imu[i - 1].timestamp_ns ||
        !m.imu[i].accel.allFinite() || !m.imu[i].gyro.allFinite())
      return false;
    if (m.imu[i].timestamp_ns > begin && m.imu[i - 1].timestamp_ns < end &&
        SecondsBetween(m.imu[i].timestamp_ns, m.imu[i - 1].timestamp_ns) >
            config_.max_gap_sec)
      return false;
  }
  std::vector<int64_t> times{begin, end};
  if (m.scan.start_ns > begin) times.push_back(m.scan.start_ns);
  for (const auto& sample : m.imu)
    if (sample.timestamp_ns > begin && sample.timestamp_ns < end)
      times.push_back(sample.timestamp_ns);
  std::sort(times.begin(), times.end());
  times.erase(std::unique(times.begin(), times.end()), times.end());
  // 将扫描起点也作为分段边界，扫描开始前的运动仍需从上一帧连续传播。
  size_t interval = 0;
  const auto input_at = [&](int64_t time) {
    while (interval + 1 < m.imu.size() - 1 &&
           m.imu[interval + 1].timestamp_ns < time)
      ++interval;
    const auto& a = m.imu[interval];
    const auto& b = m.imu[interval + 1];
    const float alpha = SecondsBetween(time, a.timestamp_ns) /
                        SecondsBetween(b.timestamp_ns, a.timestamp_ns);
    ImuSample u;
    u.timestamp_ns = time;
    u.accel = a.accel + alpha * (b.accel - a.accel);
    u.gyro = a.gyro + alpha * (b.gyro - a.gyro);
    return u;
  };
  for (size_t k = 0; k + 1 < times.size(); ++k) {
    const ImuSample a = input_at(times[k]), b = input_at(times[k + 1]);
    ImuSample u;
    u.accel = 0.5f * (a.accel + b.accel);
    u.gyro = 0.5f * (a.gyro + b.gyro);
    const float dt = SecondsBetween(times[k + 1], times[k]);
    const State& s = filter->GetState();
    ImuPose pose;
    pose.timestamp_ns = times[k];
    pose.rotation = s.rotation;
    pose.position = s.position;
    pose.velocity = s.velocity;
    pose.angular_velocity = u.gyro - s.gyro_bias;
    pose.acceleration_world = s.rotation *
                                  ExpSO3(pose.angular_velocity * (0.5f * dt)) *
                                  (u.accel - s.accel_bias) +
                              s.gravity;
    trajectory_.push_back(pose);
    if (!filter->Predict(u, dt)) return false;
    filter->SetTimestamp(times[k + 1]);
  }
  const State& s = filter->GetState();
  ImuPose last;
  last.timestamp_ns = end;
  last.rotation = s.rotation;
  last.position = s.position;
  last.velocity = s.velocity;
  trajectory_.push_back(last);
  return true;
}
PointCloud ImuProcessing::Deskew(const LidarScan& scan,
                                 const State& end) const {
  if (trajectory_.size() < 2 ||
      trajectory_.front().timestamp_ns > scan.start_ns ||
      trajectory_.back().timestamp_ns != scan.end_ns)
    throw std::runtime_error("deskew trajectory does not cover scan");
  PointCloud cloud = scan.points;
  std::stable_sort(cloud.begin(), cloud.end(),
                   [](const Point& a, const Point& b) {
                     return a.relative_time_ns < b.relative_time_ns;
                   });
  // “反向”只表示倒序查找轨迹区间，滤波器不做反向积分。
  size_t interval = trajectory_.size() - 2;
  for (auto it = cloud.rbegin(); it != cloud.rend(); ++it) {
    const int64_t time = scan.start_ns + it->relative_time_ns;
    if (time < scan.start_ns || time > scan.end_ns)
      throw std::runtime_error("point time outside scan");
    while (interval > 0 && time < trajectory_[interval].timestamp_ns)
      --interval;
    const auto& pose = trajectory_[interval];
    const float dt = SecondsBetween(time, pose.timestamp_ns);
    const Mat3 r = pose.rotation * ExpSO3(pose.angular_velocity * dt);
    const Vec3 p = pose.position + pose.velocity * dt +
                   0.5f * pose.acceleration_world * dt * dt;
    // L_i -> I_i -> W -> I_end -> L_end，同时补偿平移、旋转和外参杆臂。
    it->position = end.extrinsic_rotation.transpose() *
                   (end.rotation.transpose() *
                        (r * (end.extrinsic_rotation * it->position +
                              end.extrinsic_translation) +
                         p - end.position) -
                    end.extrinsic_translation);
  }
  return cloud;
}
}  // namespace lio
