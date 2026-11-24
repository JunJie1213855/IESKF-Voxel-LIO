#include "lio/RosWrapper.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <Eigen/Geometry>
#include <sensor_msgs/point_cloud2_iterator.hpp>
namespace lio {
namespace {
class CloudFields {
 public:
  explicit CloudFields(const sensor_msgs::msg::PointCloud2& message)
      : message_(message) {
    const uint64_t row_bytes = uint64_t(message.width) * message.point_step;
    const uint64_t bytes =
        message.height
            ? uint64_t(message.height - 1) * message.row_step + row_bytes
            : 0;
    if (!message.width || !message.height || !message.point_step ||
        message.row_step < row_bytes || bytes > message.data.size())
      throw std::invalid_argument(
          "invalid PointCloud2 dimensions or data length");
    const uint16_t one = 1;
    swap_ =
        message.is_bigendian != (*reinterpret_cast<const uint8_t*>(&one) == 0);
  }
  const sensor_msgs::msg::PointField* Find(const std::string& name,
                                           bool required = true) const {
    for (const auto& f : message_.fields)
      if (f.name == name) {
        const size_t size = Size(f.datatype);
        if (!size || f.count != 1 ||
            uint64_t(f.offset) + size > message_.point_step)
          throw std::invalid_argument("invalid field: " + name);
        return &f;
      }
    if (required) throw std::invalid_argument("missing field: " + name);
    return nullptr;
  }
  double Read(size_t index, const sensor_msgs::msg::PointField& field) const {
    const size_t offset = (index / message_.width) * message_.row_step +
                          (index % message_.width) * message_.point_step +
                          field.offset;
    const auto* p = message_.data.data() + offset;
    using F = sensor_msgs::msg::PointField;
    switch (field.datatype) {
      case F::INT8:
        return Value<int8_t>(p);
      case F::UINT8:
        return Value<uint8_t>(p);
      case F::INT16:
        return Value<int16_t>(p);
      case F::UINT16:
        return Value<uint16_t>(p);
      case F::INT32:
        return Value<int32_t>(p);
      case F::UINT32:
        return Value<uint32_t>(p);
      case F::FLOAT32:
        return Value<float>(p);
      case F::FLOAT64:
        return Value<double>(p);
      default:
        throw std::invalid_argument("unsupported field type");
    }
  }

 private:
  static size_t Size(uint8_t type) {
    switch (type) {
      case 1:
      case 2:
        return 1;
      case 3:
      case 4:
        return 2;
      case 5:
      case 6:
      case 7:
        return 4;
      case 8:
        return 8;
      default:
        return 0;
    }
  }
  template <class T>
  T Value(const uint8_t* p) const {
    uint8_t data[sizeof(T)];
    std::memcpy(data, p, sizeof(T));
    if (swap_) std::reverse(data, data + sizeof(T));
    T v;
    std::memcpy(&v, data, sizeof(T));
    return v;
  }
  const sensor_msgs::msg::PointCloud2& message_;
  bool swap_ = false;
};
int64_t Nanoseconds(long double value) {
  if (!std::isfinite(value) || value < 0 ||
      value > static_cast<long double>(std::numeric_limits<int64_t>::max()) - 1)
    throw std::invalid_argument("timestamp outside int64 range");
  return static_cast<int64_t>(std::round(value));
}
int64_t StampNs(const builtin_interfaces::msg::Time& t) {
  if (t.sec < 0 || t.nanosec >= 1000000000u)
    throw std::invalid_argument("invalid ROS stamp");
  return int64_t(t.sec) * 1000000000LL + t.nanosec;
}
geometry_msgs::msg::Quaternion Quaternion(const Mat3& r) {
  const Eigen::Quaternionf q = Eigen::Quaternionf(r).normalized();
  geometry_msgs::msg::Quaternion out;
  out.x = q.x();
  out.y = q.y();
  out.z = q.z();
  out.w = q.w();
  return out;
}
geometry_msgs::msg::TransformStamped Transform(const State& s, int64_t t,
                                               const RosConfig& c,
                                               bool extrinsic) {
  geometry_msgs::msg::TransformStamped out;
  out.header.stamp = rclcpp::Time(t);
  out.header.frame_id = extrinsic ? c.body_frame : c.world_frame;
  out.child_frame_id = extrinsic ? c.lidar_frame : c.body_frame;
  const Vec3 p = extrinsic ? s.extrinsic_translation : s.position;
  out.transform.translation.x = p.x();
  out.transform.translation.y = p.y();
  out.transform.translation.z = p.z();
  out.transform.rotation =
      Quaternion(extrinsic ? s.extrinsic_rotation : s.rotation);
  return out;
}
sensor_msgs::msg::PointCloud2 CloudMessage(const PointCloud& cloud,
                                           const State& state,
                                           const std_msgs::msg::Header& header,
                                           bool world) {
  sensor_msgs::msg::PointCloud2 out;
  out.header = header;
  sensor_msgs::PointCloud2Modifier mod(out);
  mod.setPointCloud2Fields(4, "x", 1, sensor_msgs::msg::PointField::FLOAT32,
                           "y", 1, sensor_msgs::msg::PointField::FLOAT32, "z",
                           1, sensor_msgs::msg::PointField::FLOAT32,
                           "intensity", 1,
                           sensor_msgs::msg::PointField::FLOAT32);
  mod.resize(cloud.size());
  sensor_msgs::PointCloud2Iterator<float> x(out, "x"), y(out, "y"), z(out, "z"),
      intensity(out, "intensity");
  for (const auto& p : cloud) {
    const Vec3 v = world ? WorldPoint(state, p.position) : p.position;
    *x = v.x();
    *y = v.y();
    *z = v.z();
    *intensity = p.intensity;
    ++x;
    ++y;
    ++z;
    ++intensity;
  }
  out.is_dense = true;
  return out;
}
}  // namespace
std_msgs::msg::ColorRGBA PlaneCovarianceColor(const PlaneMatch& plane,
                                              float max_trace, float gamma) {
  std_msgs::msg::ColorRGBA color;
  color.a = 0.6f;
  // 只使用法向量协方差的迹，不与单位为 m² 的平面中心协方差相加。
  const float trace = plane.covariance.topLeftCorner<3, 3>().trace();
  if (!std::isfinite(trace) || !std::isfinite(max_trace) || max_trace <= 0 ||
      !std::isfinite(gamma) || gamma <= 0) {
    color.r = color.g = color.b = 0.5f;
    return color;
  }
  // 固定量程确保不同帧颜色可直接比较；幂次增强较小方差之间的区分。
  const float t = std::pow(std::clamp(trace / max_trace, 0.0f, 1.0f), gamma);
  // 蓝 -> 青 -> 绿 -> 黄 -> 红，端点精确表示最小值和量程上限。
  color.r = std::clamp(4.0f * t - 2.0f, 0.0f, 1.0f);
  color.g = std::clamp(2.0f - std::abs(4.0f * t - 2.0f), 0.0f, 1.0f);
  color.b = std::clamp(2.0f - 4.0f * t, 0.0f, 1.0f);
  return color;
}
void InputConfig::Validate() const {
  if ((lidar_type != LidarType::kRobosense &&
       lidar_type != LidarType::kOuster && lidar_type != LidarType::kVelodyne &&
       lidar_type != LidarType::kHesai && lidar_type != LidarType::kLivox) ||
      (time_unit != "seconds" && time_unit != "milliseconds" &&
       time_unit != "microseconds" && time_unit != "nanoseconds") ||
      (time_reference != "absolute" && time_reference != "relative") ||
      (header_stamp_reference != "scan_start" &&
       header_stamp_reference != "scan_end") ||
      !std::isfinite(accel_scale) || accel_scale <= 0)
    throw std::invalid_argument("invalid input parameters");
}
namespace {
template <typename PointT>
struct PointFields;
template <>
struct PointFields<RobosensePointType> {
  static constexpr const char* kTime = "timestamp";
  static double Time(const RobosensePointType& p) { return p.timestamp; }
  static void SetTime(RobosensePointType* p, double value) {
    p->timestamp = value;
  }
};
template <>
struct PointFields<OusterPointType> {
  static constexpr const char* kTime = "t";
  static double Time(const OusterPointType& p) { return p.t; }
  static void SetTime(OusterPointType* p, double value) {
    p->t = static_cast<std::uint32_t>(value);
  }
};
template <>
struct PointFields<VelodynePointType> {
  static constexpr const char* kTime = "time";
  static double Time(const VelodynePointType& p) { return p.time; }
  static void SetTime(VelodynePointType* p, double value) {
    p->time = static_cast<float>(value);
  }
};
template <>
struct PointFields<HesaiPointType> {
  static constexpr const char* kTime = "timestamp";
  static double Time(const HesaiPointType& p) { return p.timestamp; }
  static void SetTime(HesaiPointType* p, double value) { p->timestamp = value; }
};
template <>
struct PointFields<LivoxPointType> {
  static constexpr const char* kTime = "timestamp";
  static double Time(const LivoxPointType& p) { return p.timestamp; }
  static void SetTime(LivoxPointType* p, double value) { p->timestamp = value; }
};

template <typename PointT>
LidarScan DecodeTypedCloud(const sensor_msgs::msg::PointCloud2& msg,
                           const InputConfig& c) {
  CloudFields fields(msg);
  const auto* x = fields.Find("x");
  const auto* y = fields.Find("y");
  const auto* z = fields.Find("z");
  const auto* intensity = fields.Find("intensity", false);
  const auto* time = fields.Find(PointFields<PointT>::kTime);
  if constexpr (std::is_same_v<PointT, OusterPointType>) {
    if (time->datatype != sensor_msgs::msg::PointField::UINT32)
      throw std::invalid_argument("Ouster t must be UINT32");
  }
  if constexpr (std::is_same_v<PointT, VelodynePointType>) {
    if (time->datatype != sensor_msgs::msg::PointField::FLOAT32)
      throw std::invalid_argument("Velodyne time must be FLOAT32");
  }
  long double scale = 1;
  if (c.time_unit == "seconds")
    scale = 1e9L;
  else if (c.time_unit == "milliseconds")
    scale = 1e6L;
  else if (c.time_unit == "microseconds")
    scale = 1e3L;
  const size_t count = size_t(msg.width) * msg.height;
  std::vector<int64_t> times(count, -1);
  int64_t minimum = std::numeric_limits<int64_t>::max(), maximum = -1;
  const int64_t header = StampNs(msg.header.stamp);
  // 在过滤距离、无效坐标和抽点之前确定时间范围，避免扫描时长随抽点改变。
  for (size_t i = 0; i < count; ++i) {
    const double field_value = fields.Read(i, *time);
    if (!std::isfinite(field_value) || field_value < 0) continue;
    PointT driver_point{};
    PointFields<PointT>::SetTime(&driver_point, field_value);
    const double raw = PointFields<PointT>::Time(driver_point);
    const int64_t t = Nanoseconds(static_cast<long double>(raw) * scale);
    if (c.time_reference == "absolute") {
      if (std::abs(static_cast<long double>(t) - header) > 2e9L) continue;
    } else if (t > 500000000LL)
      continue;
    times[i] = t;
    minimum = std::min(minimum, t);
    maximum = std::max(maximum, t);
  }
  if (maximum < 0 || maximum <= minimum || maximum - minimum > 500000000LL)
    throw std::invalid_argument(
        "missing/constant timestamps or scan duration exceeds 0.5 s");
  int64_t start = minimum, end = maximum, origin = minimum;
  if (c.time_reference == "relative") {
    const int64_t base =
        c.header_stamp_reference == "scan_start" ? header : header - maximum;
    if (base < 0)
      throw std::invalid_argument("relative timestamp before epoch");
    start = base;
    end = base + maximum;
    origin = 0;
  }
  const auto shift = [&](int64_t t) {
    return Nanoseconds(static_cast<long double>(t) + c.lidar_time_offset_ns);
  };
  LidarScan scan;
  scan.start_ns = shift(start);
  scan.end_ns = shift(end);
  scan.points.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    if (times[i] < 0) continue;
    // 驱动点类型按字段偏移填充，避免假定 ROS 消息布局与 PCL 的对齐布局一致。
    PointT driver_point{};
    driver_point.x = static_cast<float>(fields.Read(i, *x));
    driver_point.y = static_cast<float>(fields.Read(i, *y));
    driver_point.z = static_cast<float>(fields.Read(i, *z));
    driver_point.intensity =
        intensity ? static_cast<float>(fields.Read(i, *intensity)) : 0;
    Point p;
    p.position = Vec3(driver_point.x, driver_point.y, driver_point.z);
    p.intensity = driver_point.intensity;
    p.relative_time_ns = times[i] - origin;
    if (p.position.allFinite() && std::isfinite(p.intensity))
      scan.points.push_back(p);
  }
  if (scan.points.empty())
    throw std::invalid_argument("no finite timed points");
  return scan;
}
}  // namespace
LidarScan DecodeCloud(const sensor_msgs::msg::PointCloud2& msg,
                      const InputConfig& c) {
  c.Validate();
  switch (c.lidar_type) {
    case LidarType::kRobosense:
      return DecodeTypedCloud<RobosensePointType>(msg, c);
    case LidarType::kOuster:
      return DecodeTypedCloud<OusterPointType>(msg, c);
    case LidarType::kVelodyne:
      return DecodeTypedCloud<VelodynePointType>(msg, c);
    case LidarType::kHesai:
      return DecodeTypedCloud<HesaiPointType>(msg, c);
    case LidarType::kLivox:
      return DecodeTypedCloud<LivoxPointType>(msg, c);
  }
  throw std::invalid_argument("unsupported LiDAR type");
}

LidarScan DecodeLivoxCustom(const livox_ros_driver2::msg::CustomMsg& msg,
                            const InputConfig& c) {
  c.Validate();
  if (c.lidar_type != LidarType::kLivox || msg.point_num != msg.points.size() ||
      msg.points.empty())
    throw std::invalid_argument("invalid Livox CustomMsg or lidar type");
  const int64_t base = Nanoseconds(static_cast<long double>(msg.timebase));
  if (std::abs(static_cast<long double>(base) - StampNs(msg.header.stamp)) >
      2e9L)
    throw std::invalid_argument("Livox timebase and header clock differ");
  std::uint32_t max_offset = 0;
  for (const auto& point : msg.points)
    max_offset = std::max(max_offset, point.offset_time);
  if (max_offset == 0 || max_offset > 500000000U)
    throw std::invalid_argument("invalid Livox scan duration");
  LidarScan scan;
  // Livox 基准时间和点内偏移均使用整数纳秒，避免大时间戳转换为浮点后丢失精度。
  scan.start_ns =
      Nanoseconds(static_cast<long double>(base) + c.lidar_time_offset_ns);
  scan.end_ns =
      Nanoseconds(static_cast<long double>(scan.start_ns) + max_offset);
  scan.points.reserve(msg.points.size());
  for (const auto& raw : msg.points) {
    Point point;
    point.position = Vec3(raw.x, raw.y, raw.z);
    point.intensity = static_cast<float>(raw.reflectivity);
    point.relative_time_ns = raw.offset_time;
    if (point.position.allFinite()) scan.points.push_back(point);
  }
  if (scan.points.empty())
    throw std::invalid_argument("no finite Livox points");
  return scan;
}
RosWrapper::RosWrapper(const rclcpp::NodeOptions& options)
    : Node("ieskf_voxel_lio", options) {
  const auto f = [&](const std::string& name, float value) {
    const double v = declare_parameter<double>(name, value);
    if (!std::isfinite(v) || std::abs(v) > std::numeric_limits<float>::max())
      throw std::invalid_argument("nonfinite/out-of-range parameter: " + name);
    return static_cast<float>(v);
  };
  const auto i = [&](const std::string& name, int value) {
    return declare_parameter<int>(name, value);
  };
  const auto b = [&](const std::string& name, bool value) {
    return declare_parameter<bool>(name, value);
  };
  const auto str = [&](const std::string& name, const std::string& value) {
    return declare_parameter<std::string>(name, value);
  };
  auto& c = config_;
  c.reliable_input = b("common.reliable_input", false);
  c.lidar_topic = str("common.lidar_topic", c.lidar_topic);
  c.imu_topic = str("common.imu_topic", c.imu_topic);
  c.world_frame = str("common.world_frame", c.world_frame);
  c.body_frame = str("common.body_frame", c.body_frame);
  c.lidar_frame = str("common.lidar_frame", c.lidar_frame);
  const double offset =
      declare_parameter<double>("common.lidar_time_offset_sec", 0);
  if (!std::isfinite(offset) || std::abs(offset) > 10)
    throw std::invalid_argument("LiDAR time offset exceeds 10 seconds");
  c.input.lidar_time_offset_ns =
      static_cast<int64_t>(std::llround(offset * 1e9));
  c.preprocess.min_range = f("preprocess.min_range", .5f);
  c.preprocess.max_range = f("preprocess.max_range", 150);
  c.preprocess.point_stride = i("preprocess.point_stride", 3);
  c.preprocess.scan_voxel_size = f("preprocess.scan_voxel_size", .5f);
  // 参数边界使用整数，运行时以枚举分发，避免在点云处理路径进行字符串类型判断。
  c.input.lidar_type = static_cast<LidarType>(i("preprocess.lidar_type", 1));
  c.input.livox_custom_msg = b("preprocess.livox_custom_msg", false);
  if (c.input.livox_custom_msg && c.input.lidar_type != LidarType::kLivox)
    throw std::invalid_argument("livox_custom_msg requires lidar_type 5");
  c.input.time_unit = str("preprocess.time_unit", "seconds");
  c.input.time_reference = str("preprocess.time_reference", "absolute");
  c.input.header_stamp_reference =
      str("Preprocess.header_stamp_reference", "scan_start");
  c.imu.gravity_magnitude = f("imu.gravity_magnitude", 9.81f);
  const auto vector = [&](const std::string& name,
                          const std::vector<double>& initial, size_t count) {
    auto values = declare_parameter<std::vector<double>>(name, initial);
    if (values.size() != count)
      throw std::invalid_argument("invalid vector length: " + name);
    for (double v : values)
      if (!std::isfinite(v) || std::abs(v) > std::numeric_limits<float>::max())
        throw std::invalid_argument("invalid vector: " + name);
    return values;
  };
  const auto gravity = vector("imu.gravity_direction", {0, 0, -1}, 3);
  c.imu.gravity_direction = Vec3(gravity[0], gravity[1], gravity[2]);
  c.imu.init_duration_sec = f("imu.init_duration_sec", 1);
  c.imu.init_min_samples = i("imu.init_min_samples", 100);
  c.imu.init_max_gyro_std = f("imu.init_max_gyro_std", .02f);
  c.imu.init_max_accel_std = f("imu.init_max_accel_std", .2f);
  c.imu.init_max_gyro_mean = f("imu.init_max_gyro_mean", .1f);
  c.imu.max_gap_sec = f("imu.max_gap_sec", .05f);
  c.input.accel_scale = f("imu.accel_scale", 1);
  c.filter.gyro_noise_density = f("imu.gyro_noise_density", .01f);
  c.filter.accel_noise_density = f("imu.accel_noise_density", .1f);
  c.filter.gyro_bias_random_walk = f("imu.gyro_bias_random_walk", .0001f);
  c.filter.accel_bias_random_walk = f("imu.accel_bias_random_walk", .001f);
  c.filter.estimate_extrinsic = b("extrinsic.estimate", false);
  const auto translation = vector("extrinsic.translation", {0, 0, 0}, 3);
  c.imu.extrinsic_translation =
      Vec3(translation[0], translation[1], translation[2]);
  const auto rotation =
      vector("extrinsic.rotation", {1, 0, 0, 0, 1, 0, 0, 0, 1}, 9);
  for (int r = 0; r < 3; ++r)
    for (int col = 0; col < 3; ++col)
      c.imu.extrinsic_rotation(r, col) = rotation[r * 3 + col];
  c.filter.max_iterations = i("ieskf.max_iterations", 4);
  c.filter.min_effective_points = i("ieskf.min_effective_points", 30);
  c.filter.rotation_convergence_rad =
      f("ieskf.rotation_convergence_rad", .001f);
  c.filter.translation_convergence_m =
      f("ieskf.translation_convergence_m", .001f);
  c.filter.rematch_each_iteration = b("ieskf.rematch_each_iteration", true);
  c.filter.measurement_model = str("ieskf.measurement_model", "constant");
  c.filter.plane_residual_std = f("ieskf.plane_residual_std", .1f);
  c.filter.lidar_range_std = f("ieskf.lidar_range_std", .04f);
  c.filter.lidar_angle_std_deg = f("ieskf.lidar_angle_std_deg", .1f);
  c.filter.residual_sigma_gate = f("ieskf.residual_sigma_gate", 3);
  c.filter.min_residual_std = f("ieskf.min_residual_std", .01f);
  c.filter.max_plane_distance = f("ieskf.max_plane_distance", .3f);
  c.map.root_voxel_size = f("voxel_map.root_voxel_size", 3);
  c.map.max_layer = i("voxel_map.max_layer", 3);
  const auto layer = declare_parameter<std::vector<int64_t>>(
      "voxel_map.min_points_per_layer", {10, 10, 8, 5});
  c.map.min_points_per_layer.clear();
  for (int64_t n : layer) {
    if (n < 3 || n > 1000000)
      throw std::invalid_argument("invalid layer point count");
    c.map.min_points_per_layer.push_back(static_cast<int>(n));
  }
  c.map.plane_max_min_eigenvalue =
      f("voxel_map.plane_max_min_eigenvalue", .01f);
  c.map.plane_max_eigenvalue_ratio =
      f("voxel_map.plane_max_eigenvalue_ratio", .1f);
  c.map.max_points_per_node = i("voxel_map.max_points_per_node", 100);
  c.map.max_cov_points_per_node = i("voxel_map.max_cov_points_per_node", 100);
  c.map.neighbor_voxel_radius = i("voxel_map.neighbor_voxel_radius", 1);
  c.map.local_map_radius = f("voxel_map.local_map_radius", 100);
  c.map.max_root_voxels = i("voxel_map.max_root_voxels", 100000);
  c.preprocess.max_imu_samples = i("buffers.max_imu_samples", 4000);
  c.preprocess.max_lidar_scans = i("buffers.max_lidar_scans", 20);
  auto& p = c.publish;
  p.odometry = b("publish.odometry", true);
  p.tf = b("publish.tf", true);
  p.path = b("publish.path", true);
  p.deskewed_cloud = b("publish.deskewed_cloud", true);
  p.world_cloud = b("publish.world_cloud", true);
  p.voxel_planes = b("publish.voxel_planes", false);
  p.diagnostics = b("publish.diagnostics", true);
  p.max_path_poses = i("publish.max_path_poses", 10000);
  p.map_period_sec = f("publish.map_period_sec", 2);
  p.plane_color_max_normal_cov_trace =
      f("publish.plane_color_max_normal_cov_trace", 0.25f);
  p.plane_color_gamma = f("publish.plane_color_gamma", 0.2f);
  c.preprocess.Validate();
  c.imu.Validate();
  c.filter.Validate();
  c.map.Validate();
  c.input.Validate();
  if (c.world_frame.empty() || c.body_frame.empty() || c.lidar_frame.empty() ||
      c.world_frame == c.body_frame || c.body_frame == c.lidar_frame ||
      c.world_frame == c.lidar_frame || c.lidar_topic.empty() ||
      c.imu_topic.empty() || p.max_path_poses < 1 || p.map_period_sec <= 0 ||
      p.plane_color_max_normal_cov_trace <= 0 || p.plane_color_gamma <= 0)
    throw std::invalid_argument("invalid ROS frames/output parameters");
  odometry_pub_ = create_publisher<nav_msgs::msg::Odometry>("odometry", 10);
  path_pub_ = create_publisher<nav_msgs::msg::Path>("path", 1);
  deskew_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "cloud_deskewed", rclcpp::SensorDataQoS());
  world_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "cloud_world", rclcpp::SensorDataQoS());
  diagnostics_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
      "diagnostics", 10);
  planes_pub_ =
      create_publisher<visualization_msgs::msg::MarkerArray>("voxel_planes", 1);
  if (p.tf) {
    tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    if (!c.filter.estimate_extrinsic)
      static_tf_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);
  }
}
void RosWrapper::BindInput(Preprocess* input) {
  if (!input || input_)
    throw std::invalid_argument("input already bound or null");
  input_ = input;
  lidar_group_ =
      create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  imu_group_ =
      create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  rclcpp::SubscriptionOptions lidar_options, imu_options;
  lidar_options.callback_group = lidar_group_;
  imu_options.callback_group = imu_group_;
  auto lidar_qos = rclcpp::SensorDataQoS().keep_last(10);
  auto imu_qos = rclcpp::SensorDataQoS().keep_last(200);
  if (config_.reliable_input) {
    lidar_qos.reliable();
    imu_qos.reliable();
  }
  if (config_.input.livox_custom_msg) {
    livox_sub_ = create_subscription<livox_ros_driver2::msg::CustomMsg>(
        config_.lidar_topic, lidar_qos,
        [this](livox_ros_driver2::msg::CustomMsg::ConstSharedPtr msg) {
          try {
            if (!input_->AddLidar(DecodeLivoxCustom(*msg, config_.input)))
              RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                                   "Livox scan rejected by preprocessing");
          } catch (const std::exception& e) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                                 "Livox input rejected: %s", e.what());
          }
        },
        lidar_options);
  } else {
    lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        config_.lidar_topic, lidar_qos,
        [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
          try {
            if (!input_->AddLidar(DecodeCloud(*msg, config_.input)))
              RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                                   "LiDAR scan rejected by preprocessing");
          } catch (const std::exception& e) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                                 "LiDAR input rejected: %s", e.what());
          }
        },
        lidar_options);
  }
  imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      config_.imu_topic, imu_qos,
      [this](sensor_msgs::msg::Imu::ConstSharedPtr msg) {
        try {
          ImuSample u;
          u.timestamp_ns = StampNs(msg->header.stamp);
          u.accel = Vec3(msg->linear_acceleration.x, msg->linear_acceleration.y,
                         msg->linear_acceleration.z) *
                    config_.input.accel_scale;
          u.gyro = Vec3(msg->angular_velocity.x, msg->angular_velocity.y,
                        msg->angular_velocity.z);
          if (!input_->AddImu(u))
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                                 "IMU input rejected");
        } catch (const std::exception& e) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                               "IMU input rejected: %s", e.what());
        }
      },
      imu_options);
  if (static_tf_) {
    State s;
    s.extrinsic_rotation = config_.imu.extrinsic_rotation;
    s.extrinsic_translation = config_.imu.extrinsic_translation;
    static_tf_->sendTransform(Transform(s, 0, config_, true));
  }
}
bool RosWrapper::MapDue(int64_t stamp) const {
  return config_.publish.voxel_planes &&
         (last_map_ns_ < 0 || stamp < last_map_ns_ ||
          SecondsBetween(stamp, last_map_ns_) >=
              config_.publish.map_period_sec);
}
void RosWrapper::ResetOutput() {
  path_.poses.clear();
  last_map_ns_ = -1;
}
void RosWrapper::Publish(const ResultSnapshot& r, const BufferStats& stats) {
  const auto& p = config_.publish;
  std_msgs::msg::Header header;
  header.stamp = rclcpp::Time(r.timestamp_ns);
  header.frame_id = config_.world_frame;
  if (r.initialized) {
    nav_msgs::msg::Odometry odom;
    odom.header = header;
    odom.child_frame_id = config_.body_frame;
    odom.pose.pose.position.x = r.state.position.x();
    odom.pose.pose.position.y = r.state.position.y();
    odom.pose.pose.position.z = r.state.position.z();
    odom.pose.pose.orientation = Quaternion(r.state.rotation);
    const Vec3 velocity = r.state.rotation.transpose() * r.state.velocity;
    odom.twist.twist.linear.x = velocity.x();
    odom.twist.twist.linear.y = velocity.y();
    odom.twist.twist.linear.z = velocity.z();
    // ROS 姿态协方差使用父坐标系小转角；内部姿态误差在机体系右切空间。
    Eigen::Matrix<float, 6, kStateDim> jp =
        Eigen::Matrix<float, 6, kStateDim>::Zero();
    jp.block<3, 3>(0, 0) = Mat3::Identity();
    jp.block<3, 3>(3, 3) = r.state.rotation;
    const Eigen::Matrix<float, 6, 6> pose_cov =
        jp * r.covariance * jp.transpose();
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 6; ++j)
        odom.pose.covariance[i * 6 + j] = pose_cov(i, j);
    Eigen::Matrix<float, 3, kStateDim> jv =
        Eigen::Matrix<float, 3, kStateDim>::Zero();
    jv.block<3, 3>(0, 3) = Hat(velocity);
    jv.block<3, 3>(0, 12) = r.state.rotation.transpose();
    const Mat3 velocity_cov = jv * r.covariance * jv.transpose();
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        odom.twist.covariance[i * 6 + j] = velocity_cov(i, j);
    // 本版不发布角速度估计，给未提供分量明确的大不确定性。
    for (int i = 3; i < 6; ++i) odom.twist.covariance[i * 6 + i] = 1e6;
    if (p.odometry) odometry_pub_->publish(odom);
    if (p.tf) {
      tf_->sendTransform(Transform(r.state, r.timestamp_ns, config_, false));
      if (config_.filter.estimate_extrinsic)
        tf_->sendTransform(Transform(r.state, r.timestamp_ns, config_, true));
    }
    if (p.path) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header = header;
      pose.pose = odom.pose.pose;
      path_.header = header;
      path_.poses.push_back(pose);
      if (path_.poses.size() > static_cast<size_t>(p.max_path_poses))
        path_.poses.erase(path_.poses.begin());
      path_pub_->publish(path_);
    }
    if (p.deskewed_cloud) {
      auto h = header;
      h.frame_id = config_.lidar_frame;
      deskew_pub_->publish(CloudMessage(r.cloud, r.state, h, false));
    }
    if (p.world_cloud)
      world_pub_->publish(CloudMessage(r.cloud, r.state, header, true));
    if (MapDue(r.timestamp_ns)) {
      visualization_msgs::msg::MarkerArray markers;
      visualization_msgs::msg::Marker clear;
      clear.header = header;
      clear.action = visualization_msgs::msg::Marker::DELETEALL;
      markers.markers.push_back(clear);
      int id = 0;
      for (const auto& plane : r.planes) {
        visualization_msgs::msg::Marker m;
        m.header = header;
        m.ns = "planes";
        m.id = id++;
        m.type = visualization_msgs::msg::Marker::CYLINDER;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.pose.position.x = plane.center.x();
        m.pose.position.y = plane.center.y();
        m.pose.position.z = plane.center.z();
        m.pose.orientation = Quaternion(
            Eigen::Quaternionf::FromTwoVectors(Vec3::UnitZ(), plane.normal)
                .toRotationMatrix());
        m.scale.x = m.scale.y = std::max(.05f, 2 * plane.radius);
        m.scale.z = .02;
        m.color = PlaneCovarianceColor(
            plane, p.plane_color_max_normal_cov_trace, p.plane_color_gamma);
        markers.markers.push_back(m);
      }
      planes_pub_->publish(markers);
      last_map_ns_ = r.timestamp_ns;
    }
  }
  if (p.diagnostics) {
    diagnostic_msgs::msg::DiagnosticArray out;
    out.header = header;
    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "ieskf_voxel_lio";
    status.hardware_id = "lidar_imu";
    status.message = r.status;
    status.level = (!r.initialized ||
                    (!r.update.accepted && r.status != "map initialized"))
                       ? diagnostic_msgs::msg::DiagnosticStatus::WARN
                       : diagnostic_msgs::msg::DiagnosticStatus::OK;
    const auto add = [&](const std::string& k, const auto& value) {
      diagnostic_msgs::msg::KeyValue kv;
      kv.key = k;
      kv.value = std::to_string(value);
      status.values.push_back(kv);
    };
    add("initialized", r.initialized);
    add("effective_points", r.update.effective_points);
    add("iterations", r.update.iterations);
    add("mean_abs_residual", r.update.mean_abs_residual);
    add("pose_information_ratio", r.update.pose_information_ratio);
    add("map_voxels", r.map_voxels);
    add("trimmed_overlap_points", r.trimmed_overlap_points);
    add("imu_dropped", stats.imu_dropped);
    add("lidar_dropped", stats.lidar_dropped);
    add("buffer_resets", stats.resets);
    add("failures", r.failures);
    add("propagation_ms", r.propagation_ms);
    add("deskew_ms", r.deskew_ms);
    add("update_ms", r.update_ms);
    add("mapping_ms", r.mapping_ms);
    add("total_ms", r.total_ms);
    out.status.push_back(status);
    diagnostics_pub_->publish(out);
  }
}
}  // namespace lio
