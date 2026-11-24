#include <chrono>
#include <exception>
#include <iostream>
#include <thread>

#include "lio/RosWrapper.h"
namespace lio {
namespace {
using Clock = std::chrono::steady_clock;
float Milliseconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<float, std::milli>(end - begin).count();
}
}  // namespace
// 所有跨模块状态均由唯一估计线程修改；ROS 回调只向 Preprocess 入队。
class LioSystem {
 public:
  explicit LioSystem(std::shared_ptr<RosWrapper> ros)
      : ros_(std::move(ros)),
        preprocess_(ros_->Config().preprocess),
        imu_(ros_->Config().imu),
        filter_(ros_->Config().filter),
        map_(ros_->Config().map) {
    ros_->BindInput(&preprocess_);
    estimator_ = std::thread(&LioSystem::Run, this);
  }
  ~LioSystem() {
    preprocess_.Stop();
    if (estimator_.joinable()) estimator_.join();
  }

 private:
  void Reset() {
    initialized_ = false;
    bootstrap_frames_ = 0;
    consecutive_failures_ = 0;
    imu_.Reset();
    filter_.Reset();
    map_.Clear();
    ros_->ResetOutput();
  }
  void Run() {
    MeasurementGroup measurements;
    while (preprocess_.WaitMeasurements(&measurements)) {
      if (!rclcpp::ok()) break;
      try {
        Process(measurements);
      } catch (const std::exception& e) {
        ++failures_;
        RCLCPP_ERROR(ros_->get_logger(), "LIO processing reset: %s", e.what());
        Reset();
      }
    }
  }
  void Process(MeasurementGroup& m) {
    const auto begin = Clock::now();
    if (m.reset) Reset();
    ResultSnapshot out;
    out.timestamp_ns = m.scan.end_ns;
    out.failures = failures_;
    if (!initialized_) {
      initialized_ = imu_.Initialize(m, &filter_);
      out.status = initialized_ ? "IMU initialized; waiting for next scan"
                                : "waiting for stationary IMU initialization";
      out.total_ms = Milliseconds(begin, Clock::now());
      ros_->Publish(out, preprocess_.Stats());
      return;
    }
    // Airy 相邻包可能有微秒级重叠，只移除上一滤波时刻之前的点并重置相对时间。
    out.trimmed_overlap_points = TrimScanBefore(filter_.Timestamp(), &m.scan);
    if (!imu_.Propagate(m, &filter_)) {
      ++failures_;
      Reset();
      out.status = "IMU coverage/gap failure; reinitializing";
      out.failures = failures_;
      ros_->Publish(out, preprocess_.Stats());
      return;
    }
    const auto propagated = Clock::now();
    PointCloud cloud =
        preprocess_.Downsample(imu_.Deskew(m.scan, filter_.GetState()));
    const auto deskewed = Clock::now();
    if (cloud.empty())
      throw std::runtime_error("empty cloud after deskew/downsample");
    const State& predicted = filter_.GetState();
    map_.Crop(predicted.position +
              predicted.rotation * predicted.extrinsic_translation);
    const bool bootstrap = map_.Planes(1).empty();
    if (!bootstrap) {
      out.update = filter_.Update(cloud, map_);
      if (out.update.accepted)
        consecutive_failures_ = 0;
      else {
        ++failures_;
        ++consecutive_failures_;
        RCLCPP_WARN_THROTTLE(ros_->get_logger(), *ros_->get_clock(), 2000,
                             "LiDAR correction rejected: %s",
                             out.update.reason.c_str());
      }
    }
    const auto updated = Clock::now();
    // 先关联历史地图，成功后再插入当前帧，禁止观测与自身建出的平面匹配。
    if (bootstrap || out.update.accepted) {
      const State& s = filter_.GetState();
      map_.Crop(s.position + s.rotation * s.extrinsic_translation);
      map_.Insert(filter_.MapPoints(cloud));
    }
    if (bootstrap && ++bootstrap_frames_ > 10 && map_.Planes(1).empty())
      throw std::runtime_error("no usable planes after ten bootstrap scans");
    const auto mapped = Clock::now();
    out.initialized = true;
    out.state = filter_.GetState();
    out.covariance = filter_.GetCovariance();
    out.cloud = std::move(cloud);
    out.map_voxels = map_.Size();
    out.failures = failures_;
    out.status = bootstrap
                     ? (map_.Planes(1).empty() ? "accumulating initial map"
                                               : "map initialized")
                     : out.update.reason;
    out.propagation_ms = Milliseconds(begin, propagated);
    out.deskew_ms = Milliseconds(propagated, deskewed);
    out.update_ms = Milliseconds(deskewed, updated);
    out.mapping_ms = Milliseconds(updated, mapped);
    if (ros_->MapDue(out.timestamp_ns)) out.planes = map_.Planes();
    out.total_ms = Milliseconds(begin, Clock::now());
    ros_->Publish(out, preprocess_.Stats());
    // 连续失配后回到受控初始化，避免无限纯惯导漂移污染地图。
    if (consecutive_failures_ >= 20) {
      RCLCPP_ERROR(ros_->get_logger(),
                   "Twenty consecutive rejected scans; resetting LIO");
      Reset();
    }
  }
  std::shared_ptr<RosWrapper> ros_;
  Preprocess preprocess_;
  ImuProcessing imu_;
  Ieskf filter_;
  VoxelMap map_;
  bool initialized_ = false;
  int bootstrap_frames_ = 0, consecutive_failures_ = 0;
  uint64_t failures_ = 0;
  std::thread estimator_;
};
}  // namespace lio
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  int code = 0;
  try {
    auto node = std::make_shared<lio::RosWrapper>();
    lio::LioSystem system(node);
    // 大点云解析不应阻塞 200 Hz IMU 回调；两个输入组并行，估计仍为单线程。
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(),
                                                      2);
    executor.add_node(node);
    executor.spin();
  } catch (const std::exception& e) {
    std::cerr << "LIO startup failed: " << e.what() << '\n';
    code = 1;
  }
  rclcpp::shutdown();
  return code;
}
