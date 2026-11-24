#ifndef LIO_IMU_PROCESSING_H_
#define LIO_IMU_PROCESSING_H_

#include "lio/Ieskf.h"

namespace lio {
struct ImuConfig {
  float gravity_magnitude = 9.81f;
  Vec3 gravity_direction = Vec3(0, 0, -1);
  float init_duration_sec = 1, init_max_gyro_std = 0.02f,
        init_max_accel_std = 0.2f, init_max_gyro_mean = 0.1f;
  int init_min_samples = 100;
  float max_gap_sec = 0.05f;
  Mat3 extrinsic_rotation = Mat3::Identity();
  Vec3 extrinsic_translation = Vec3::Zero();
  void Validate() const;
};
struct ImuPose {
  int64_t timestamp_ns = 0;
  Mat3 rotation = Mat3::Identity();
  Vec3 position = Vec3::Zero();
  Vec3 velocity = Vec3::Zero();
  Vec3 angular_velocity = Vec3::Zero();
  Vec3 acceleration_world = Vec3::Zero();
};
class ImuProcessing {
 public:
  explicit ImuProcessing(const ImuConfig& config = {});
  void Reset();
  // 返回 true 表示静止窗口满足要求并已设置滤波器初始状态。
  bool Initialize(const MeasurementGroup& group, Ieskf* filter);
  // 保存前向轨迹；遇到未覆盖边界或 IMU 缺口时拒绝传播。
  bool Propagate(const MeasurementGroup& group, Ieskf* filter);
  // 输出扫描结束时刻 LiDAR 坐标系中的普通点。
  PointCloud Deskew(const LidarScan& scan, const State& end_state) const;
  const std::vector<ImuPose>& Trajectory() const { return trajectory_; }

 private:
  ImuConfig config_;
  int count_ = 0;
  int64_t last_init_ns_ = -1;
  int64_t first_init_ns_ = -1;
  Vec3 mean_acc_ = Vec3::Zero();
  Vec3 mean_gyro_ = Vec3::Zero();
  Vec3 m2_acc_ = Vec3::Zero();
  Vec3 m2_gyro_ = Vec3::Zero();
  std::vector<ImuPose> trajectory_;
};
}  // namespace lio
#endif
