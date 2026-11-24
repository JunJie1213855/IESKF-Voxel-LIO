#ifndef LIO_ROS_WRAPPER_H_
#define LIO_ROS_WRAPPER_H_

#include <cstdint>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pcl/point_types.h>
#include <pcl/register_point_struct.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker_array.hpp>

#include "lio/ImuProcessing.h"

namespace lio {
// 驱动消息字段的类型声明仅用于 ROS 输入转换，核心算法不依赖 PCL。
struct EIGEN_ALIGN16 RobosensePointType {
  PCL_ADD_POINT4D;
  float intensity;
  double timestamp;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
struct EIGEN_ALIGN16 OusterPointType {
  PCL_ADD_POINT4D;
  float intensity;
  std::uint32_t t;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
struct EIGEN_ALIGN16 VelodynePointType {
  PCL_ADD_POINT4D;
  float intensity;
  float time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
struct EIGEN_ALIGN16 HesaiPointType {
  PCL_ADD_POINT4D;
  float intensity;
  double timestamp;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
struct EIGEN_ALIGN16 LivoxPointType {
  PCL_ADD_POINT4D;
  float intensity;
  double timestamp;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace lio

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(lio::RobosensePointType,
    (float, x, x)(float, y, y)(float, z, z)
    (float, intensity, intensity)(double, timestamp, timestamp))
POINT_CLOUD_REGISTER_POINT_STRUCT(lio::OusterPointType,
    (float, x, x)(float, y, y)(float, z, z)
    (float, intensity, intensity)(std::uint32_t, t, t))
POINT_CLOUD_REGISTER_POINT_STRUCT(lio::VelodynePointType,
    (float, x, x)(float, y, y)(float, z, z)
    (float, intensity, intensity)(float, time, time))
POINT_CLOUD_REGISTER_POINT_STRUCT(lio::HesaiPointType,
    (float, x, x)(float, y, y)(float, z, z)
    (float, intensity, intensity)(double, timestamp, timestamp))
POINT_CLOUD_REGISTER_POINT_STRUCT(lio::LivoxPointType,
    (float, x, x)(float, y, y)(float, z, z)
    (float, intensity, intensity)(double, timestamp, timestamp))
// clang-format on

namespace lio {
enum class LidarType : int {
  kRobosense = 1,
  kOuster = 2,
  kVelodyne = 3,
  kHesai = 4,
  kLivox = 5,
};
struct InputConfig {
  LidarType lidar_type = LidarType::kRobosense;
  bool livox_custom_msg = false;
  std::string time_unit = "seconds";
  std::string time_reference = "absolute";
  std::string header_stamp_reference = "scan_start";
  int64_t lidar_time_offset_ns = 0;
  float accel_scale = 1;
  void Validate() const;
};
struct PublishConfig {
  bool odometry = true, tf = true, path = true, deskewed_cloud = true,
       world_cloud = true, voxel_planes = false, diagnostics = true;
  int max_path_poses = 10000;
  float map_period_sec = 2;
  float plane_color_max_normal_cov_trace = 0.25f;
  float plane_color_gamma = 0.2f;
};
struct RosConfig {
  bool reliable_input = false;
  std::string lidar_topic = "/rslidar_points";
  std::string imu_topic = "/rslidar_imu_data";
  std::string world_frame = "odom";
  std::string body_frame = "imu_link";
  std::string lidar_frame = "lidar_link";
  PreprocessConfig preprocess;
  ImuConfig imu;
  FilterConfig filter;
  MapConfig map;
  InputConfig input;
  PublishConfig publish;
};
// 用法向量协方差 3×3 子块的迹映射蓝到红；非有限值显示灰色。
std_msgs::msg::ColorRGBA PlaneCovarianceColor(const PlaneMatch& plane,
                                              float max_trace, float gamma);
// 仅边界层读取 ROS 消息。时间先转整数纳秒，再相减得到扫描内时间。
LidarScan DecodeCloud(const sensor_msgs::msg::PointCloud2& message,
                      const InputConfig& config);
LidarScan DecodeLivoxCustom(const livox_ros_driver2::msg::CustomMsg& message,
                            const InputConfig& config);
struct ResultSnapshot {
  int64_t timestamp_ns = 0;
  State state;
  Covariance covariance = Covariance::Zero();
  PointCloud cloud;
  UpdateResult update;
  bool initialized = false;
  size_t map_voxels = 0;
  size_t trimmed_overlap_points = 0;
  float propagation_ms = 0;
  float deskew_ms = 0;
  float update_ms = 0;
  float mapping_ms = 0;
  float total_ms = 0;
  uint64_t failures = 0;
  std::string status;
  std::vector<PlaneMatch> planes;
};
class RosWrapper : public rclcpp::Node {
 public:
  explicit RosWrapper(
      const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  const RosConfig& Config() const { return config_; }
  void BindInput(Preprocess* input);
  void Publish(const ResultSnapshot& result, const BufferStats& stats);
  bool MapDue(int64_t timestamp_ns) const;
  void ResetOutput();

 private:
  RosConfig config_;
  Preprocess* input_ = nullptr;
  rclcpp::CallbackGroup::SharedPtr lidar_group_;
  rclcpp::CallbackGroup::SharedPtr imu_group_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
  rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr livox_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr deskew_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr world_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
      diagnostics_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      planes_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_;
  nav_msgs::msg::Path path_;
  int64_t last_map_ns_ = -1;
};
}  // namespace lio
#endif
