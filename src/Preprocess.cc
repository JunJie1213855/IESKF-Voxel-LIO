#include "lio/Preprocess.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <tuple>
namespace lio {
size_t TrimScanBefore(int64_t timestamp_ns, LidarScan* scan) {
  if (timestamp_ns <= scan->start_ns) return 0;
  constexpr int64_t kMaxScanOverlapNs = 1000000;
  const int64_t overlap = timestamp_ns - scan->start_ns;
  if (overlap > kMaxScanOverlapNs || timestamp_ns >= scan->end_ns) {
    throw std::invalid_argument(
        "scan overlap exceeds 1 ms or has no new interval");
  }
  const size_t original = scan->points.size();
  scan->points.erase(std::remove_if(scan->points.begin(), scan->points.end(),
                                    [overlap](const Point& point) {
                                      return point.relative_time_ns < overlap;
                                    }),
                     scan->points.end());
  for (auto& point : scan->points) point.relative_time_ns -= overlap;
  scan->start_ns = timestamp_ns;
  return original - scan->points.size();
}
void PreprocessConfig::Validate() const {
  if (!std::isfinite(min_range) || !std::isfinite(max_range) ||
      !std::isfinite(scan_voxel_size) || min_range < 0 ||
      max_range <= min_range || point_stride < 1 || scan_voxel_size <= 0 ||
      max_imu_samples < 2 || max_lidar_scans < 1)
    throw std::invalid_argument("invalid preprocessing parameters");
}
Preprocess::Preprocess(const PreprocessConfig& c) : config_(c) { c.Validate(); }
void Preprocess::ResetLocked() {
  imu_.clear();
  lidar_.clear();
  last_imu_ns_ = last_lidar_ns_ = -1;
  // 回调只标记复位；滤波器和地图由估计线程统一清理。
  reset_pending_ = true;
  ++stats_.resets;
}
bool Preprocess::AddImu(ImuSample s) {
  if (s.timestamp_ns < 0 || !s.accel.allFinite() || !s.gyro.allFinite())
    return false;
  std::lock_guard<std::mutex> lock(mutex_);
  if (last_imu_ns_ >= 0 && s.timestamp_ns < last_imu_ns_) ResetLocked();
  if (s.timestamp_ns == last_imu_ns_) return false;
  last_imu_ns_ = s.timestamp_ns;
  imu_.push_back(s);
  if (imu_.size() > static_cast<size_t>(config_.max_imu_samples)) {
    imu_.pop_front();
    ++stats_.imu_dropped;
  }
  ready_.notify_one();
  return true;
}
bool Preprocess::AddLidar(LidarScan s) {
  if (s.start_ns < 0 || s.end_ns <= s.start_ns || s.points.empty())
    return false;
  std::stable_sort(s.points.begin(), s.points.end(),
                   [](const Point& a, const Point& b) {
                     return a.relative_time_ns < b.relative_time_ns;
                   });
  PointCloud selected;
  selected.reserve(s.points.size() / config_.point_stride + 1);
  for (size_t i = 0; i < s.points.size(); ++i) {
    const auto& p = s.points[i];
    if (p.relative_time_ns < 0 || p.relative_time_ns > s.end_ns - s.start_ns ||
        !p.position.allFinite() || !std::isfinite(p.intensity))
      continue;
    const float r = p.position.norm();
    if (i % config_.point_stride == 0 && r >= config_.min_range &&
        r <= config_.max_range)
      selected.push_back(p);
  }
  if (selected.empty()) return false;
  s.points = std::move(selected);
  std::lock_guard<std::mutex> lock(mutex_);
  if (last_lidar_ns_ >= 0 && s.start_ns < last_lidar_ns_) ResetLocked();
  if (s.start_ns == last_lidar_ns_) return false;
  last_lidar_ns_ = s.start_ns;
  lidar_.push_back(std::move(s));
  if (lidar_.size() > static_cast<size_t>(config_.max_lidar_scans)) {
    lidar_.pop_front();
    ++stats_.lidar_dropped;
  }
  ready_.notify_one();
  return true;
}
bool Preprocess::SyncLocked(MeasurementGroup* g) {
  if (imu_.size() < 2 || lidar_.empty()) return false;
  // 扫描起始处的 IMU 已丢失时，不能安全补偿整帧，直接丢弃该扫描。
  while (!lidar_.empty() &&
         imu_.front().timestamp_ns > lidar_.front().start_ns) {
    lidar_.pop_front();
    ++stats_.lidar_dropped;
  }
  if (lidar_.empty() || imu_.back().timestamp_ns < lidar_.front().end_ns)
    return false;
  const auto end = lidar_.front().end_ns;
  size_t right = 0;
  while (right < imu_.size() && imu_[right].timestamp_ns < end) ++right;
  g->scan = std::move(lidar_.front());
  lidar_.pop_front();
  g->imu.assign(imu_.begin(), imu_.begin() + right + 1);
  // 保留帧末左侧样本及右侧样本，下一帧才能连续积分并在边界插值。
  const size_t retain =
      (imu_[right].timestamp_ns == end || right == 0) ? right : right - 1;
  for (size_t i = 0; i < retain; ++i) imu_.pop_front();
  g->reset = reset_pending_;
  reset_pending_ = false;
  return true;
}
bool Preprocess::SyncMeasurements(MeasurementGroup* g) {
  std::lock_guard<std::mutex> lock(mutex_);
  return SyncLocked(g);
}
bool Preprocess::WaitMeasurements(MeasurementGroup* g) {
  std::unique_lock<std::mutex> lock(mutex_);
  ready_.wait(lock, [&] { return stopped_ || SyncLocked(g); });
  return !stopped_;
}
void Preprocess::Stop() {
  std::lock_guard<std::mutex> lock(mutex_);
  stopped_ = true;
  ready_.notify_all();
}
BufferStats Preprocess::Stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}
PointCloud Preprocess::Downsample(const PointCloud& cloud) const {
  // 选择最接近体素中心的真实点，不对来自不同时刻的点时间求均值。
  using Key = std::tuple<int64_t, int64_t, int64_t>;
  std::map<Key, std::pair<float, Point>> cells;
  const float s = config_.scan_voxel_size;
  for (const auto& p : cloud) {
    if (!p.position.allFinite()) continue;
    const Vec3 key = (p.position / s).array().floor();
    if (key.cwiseAbs().maxCoeff() > 1e12f) continue;
    const Key k{static_cast<int64_t>(key.x()), static_cast<int64_t>(key.y()),
                static_cast<int64_t>(key.z())};
    const float d =
        (p.position - (key.array() + 0.5f).matrix() * s).squaredNorm();
    auto it = cells.find(k);
    if (it == cells.end() || d < it->second.first) cells[k] = {d, p};
  }
  PointCloud result;
  result.reserve(cells.size());
  for (const auto& cell : cells) result.push_back(cell.second.second);
  return result;
}
}  // namespace lio
