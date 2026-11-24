#ifndef LIO_PREPROCESS_H_
#define LIO_PREPROCESS_H_

#include <condition_variable>
#include <deque>
#include <mutex>
#include <vector>

#include "lio/LieUtils.h"

namespace lio {
// 点坐标为 float32；相对时间保持整数纳秒直到短时间积分。
struct Point {
  Vec3 position = Vec3::Zero();
  float intensity = 0;
  int64_t relative_time_ns = 0;
};
using PointCloud = std::vector<Point>;
struct ImuSample {
  int64_t timestamp_ns = 0;
  Vec3 accel = Vec3::Zero();
  Vec3 gyro = Vec3::Zero();
};
struct LidarScan {
  int64_t start_ns = 0;
  int64_t end_ns = 0;
  PointCloud points;
};
// 仅裁剪不超过 1 ms 的相邻扫描重叠；不对已过去的时间反向传播。
size_t TrimScanBefore(int64_t timestamp_ns, LidarScan* scan);
struct MeasurementGroup {
  LidarScan scan;
  std::vector<ImuSample> imu;
  bool reset = false;
};
struct PreprocessConfig {
  float min_range = 0.5f;
  float max_range = 150;
  int point_stride = 3;
  float scan_voxel_size = 0.5f;
  int max_imu_samples = 4000;
  int max_lidar_scans = 20;
  void Validate() const;
};
struct BufferStats {
  uint64_t imu_dropped = 0, lidar_dropped = 0, resets = 0;
};
class Preprocess {
 public:
  explicit Preprocess(const PreprocessConfig& config = {});
  bool AddImu(ImuSample sample);
  bool AddLidar(LidarScan scan);
  bool WaitMeasurements(MeasurementGroup* group);
  bool SyncMeasurements(MeasurementGroup* group);
  PointCloud Downsample(const PointCloud& cloud) const;
  void Stop();
  BufferStats Stats() const;

 private:
  bool SyncLocked(MeasurementGroup* group);
  void ResetLocked();
  PreprocessConfig config_;
  mutable std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<ImuSample> imu_;
  std::deque<LidarScan> lidar_;
  int64_t last_imu_ns_ = -1;
  int64_t last_lidar_ns_ = -1;
  bool stopped_ = false;
  bool reset_pending_ = false;
  BufferStats stats_;
};
}  // namespace lio
#endif
