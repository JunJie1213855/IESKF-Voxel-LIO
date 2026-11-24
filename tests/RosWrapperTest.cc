#include <gtest/gtest.h>

#include "lio/RosWrapper.h"

TEST(LivoxCustomInput, UsesIntegerTimebaseAndOffsets) {
  lio::InputConfig config;
  config.lidar_type = lio::LidarType::kLivox;
  config.lidar_time_offset_ns = -100;
  livox_ros_driver2::msg::CustomMsg msg;
  msg.header.stamp.sec = 1712063705;
  msg.header.stamp.nanosec = 51177979;
  msg.timebase = 1712063705051177887ULL;
  msg.point_num = 2;
  msg.points.resize(2);
  msg.points[0].x = 1;
  msg.points[1].x = 2;
  msg.points[1].reflectivity = 42;
  msg.points[1].offset_time = 100505484;
  const auto scan = lio::DecodeLivoxCustom(msg, config);
  EXPECT_EQ(scan.start_ns, 1712063705051177787LL);
  EXPECT_EQ(scan.end_ns, 1712063705151683271LL);
  ASSERT_EQ(scan.points.size(), 2U);
  EXPECT_EQ(scan.points[1].relative_time_ns, 100505484);
  EXPECT_FLOAT_EQ(scan.points[1].intensity, 42);
  msg.point_num = 3;
  EXPECT_THROW(lio::DecodeLivoxCustom(msg, config), std::invalid_argument);
}

#include <cstring>
#include <limits>

namespace lio {
namespace {
template <class T>
void Write(std::vector<uint8_t>* data, size_t offset, T value, bool big) {
  uint8_t bytes[sizeof(T)];
  std::memcpy(bytes, &value, sizeof(T));
  const uint16_t one = 1;
  if (big != (*reinterpret_cast<const uint8_t*>(&one) == 0))
    std::reverse(bytes, bytes + sizeof(T));
  std::memcpy(data->data() + offset, bytes, sizeof(T));
}
sensor_msgs::msg::PointCloud2 Cloud(bool big = false, bool relative = false) {
  sensor_msgs::msg::PointCloud2 msg;
  msg.header.stamp.sec = 1795500000;
  msg.header.stamp.nanosec = 100000000;
  msg.width = 2;
  msg.height = 2;
  msg.point_step = 25;
  msg.row_step = 55;
  msg.data.resize(110);
  msg.is_bigendian = big;
  const auto field = [&](const std::string& name, uint32_t offset,
                         uint8_t type) {
    sensor_msgs::msg::PointField f;
    f.name = name;
    f.offset = offset;
    f.datatype = type;
    f.count = 1;
    msg.fields.push_back(f);
  };
  using F = sensor_msgs::msg::PointField;
  field("x", 0, F::FLOAT32);
  field("y", 4, F::FLOAT32);
  field("z", 8, F::FLOAT32);
  field("intensity", 12, F::UINT8);
  field("timestamp", 17, F::FLOAT64);
  for (size_t k = 0; k < 4; ++k) {
    const size_t offset = (k / 2) * 55 + (k % 2) * 25;
    Write(&msg.data, offset, 2.0f, big);
    Write(&msg.data, offset + 4, 1.0f, big);
    Write(&msg.data, offset + 8, -.5f, big);
    Write(&msg.data, offset + 12, uint8_t(200), big);
    Write(&msg.data, offset + 17, relative ? k * .03 : 1795500000.0 + k * .03,
          big);
  }
  return msg;
}
}  // namespace
TEST(RosInput, AbsoluteEpochBigEndianOrganizedCloud) {
  InputConfig c;
  auto msg = Cloud(true);
  const auto scan = DecodeCloud(msg, c);
  EXPECT_EQ(scan.start_ns, 1795500000000000000LL);
  EXPECT_NEAR(SecondsBetween(scan.end_ns, scan.start_ns), .09f, 1e-6f);
  ASSERT_EQ(scan.points.size(), 4u);
  EXPECT_EQ(scan.points.back().intensity, 200);
  EXPECT_EQ(scan.points.back().position.z(), -.5f);
}
TEST(RosInput, RelativeEndStampedHeaderAndOffset) {
  InputConfig c;
  c.time_reference = "relative";
  c.header_stamp_reference = "scan_end";
  c.lidar_time_offset_ns = -10000000;
  const auto scan = DecodeCloud(Cloud(false, true), c);
  EXPECT_EQ(scan.end_ns, 1795500000090000000LL);
  EXPECT_EQ(scan.end_ns - scan.start_ns, 90000000);
  EXPECT_EQ(scan.points.back().relative_time_ns, 90000000);
}
TEST(RosInput, MalformedMissingAndConstantTimeRejected) {
  InputConfig c;
  auto msg = Cloud();
  msg.data.resize(1);
  EXPECT_THROW(DecodeCloud(msg, c), std::invalid_argument);
  msg = Cloud();
  msg.fields.pop_back();
  EXPECT_THROW(DecodeCloud(msg, c), std::invalid_argument);
  msg = Cloud();
  for (size_t k = 0; k < 4; ++k)
    Write(&msg.data, (k / 2) * 55 + (k % 2) * 25 + 17, 1795500000.0, false);
  EXPECT_THROW(DecodeCloud(msg, c), std::invalid_argument);
}
TEST(RosInput, TimeRangeIndependentOfInvalidCoordinates) {
  InputConfig c;
  auto msg = Cloud();
  Write(&msg.data, 0, std::numeric_limits<float>::quiet_NaN(), false);
  const auto scan = DecodeCloud(msg, c);
  EXPECT_EQ(scan.points.size(), 3u);
  EXPECT_EQ(scan.start_ns, 1795500000000000000LL);
}
TEST(RosInput, WrongClockAndInvalidUnitsRejected) {
  InputConfig c;
  auto msg = Cloud();
  msg.header.stamp.sec = 20;
  EXPECT_THROW(DecodeCloud(msg, c), std::invalid_argument);
  c.time_unit = "unknown";
  EXPECT_THROW(DecodeCloud(Cloud(), c), std::invalid_argument);
}
}  // namespace lio

namespace lio {
TEST(PlaneColor, NormalCovarianceMapsBlueToRedAndClamps) {
  PlaneMatch plane;
  auto color = PlaneCovarianceColor(plane, 0.25f, 1.0f);
  EXPECT_FLOAT_EQ(color.r, 0.0f);
  EXPECT_FLOAT_EQ(color.g, 0.0f);
  EXPECT_FLOAT_EQ(color.b, 1.0f);
  EXPECT_FLOAT_EQ(color.a, 0.6f);

  plane.covariance(0, 0) = 0.125f;
  color = PlaneCovarianceColor(plane, 0.25f, 1.0f);
  EXPECT_FLOAT_EQ(color.r, 0.0f);
  EXPECT_FLOAT_EQ(color.g, 1.0f);
  EXPECT_FLOAT_EQ(color.b, 0.0f);

  plane.covariance(1, 1) = 0.125f;
  color = PlaneCovarianceColor(plane, 0.25f, 1.0f);
  EXPECT_FLOAT_EQ(color.r, 1.0f);
  EXPECT_FLOAT_EQ(color.g, 0.0f);
  EXPECT_FLOAT_EQ(color.b, 0.0f);
  plane.covariance(2, 2) = 1.0f;
  EXPECT_FLOAT_EQ(PlaneCovarianceColor(plane, 0.25f, 1.0f).r, 1.0f);
}
TEST(PlaneColor, UsesNormalTraceAndRejectsInvalidData) {
  PlaneMatch plane;
  plane.covariance(0, 0) = 0.0625f;
  plane.covariance(3, 3) = 100.0f;
  const auto linear = PlaneCovarianceColor(plane, 0.25f, 1.0f);
  EXPECT_FLOAT_EQ(linear.r, 0.0f);
  EXPECT_FLOAT_EQ(linear.g, 1.0f);
  EXPECT_FLOAT_EQ(linear.b, 1.0f);
  const auto enhanced = PlaneCovarianceColor(plane, 0.25f, 0.5f);
  EXPECT_FLOAT_EQ(enhanced.g, 1.0f);
  EXPECT_FLOAT_EQ(enhanced.b, 0.0f);
  plane.covariance(0, 0) = std::numeric_limits<float>::quiet_NaN();
  const auto invalid = PlaneCovarianceColor(plane, 0.25f, 0.2f);
  EXPECT_FLOAT_EQ(invalid.r, 0.5f);
  EXPECT_FLOAT_EQ(invalid.g, 0.5f);
  EXPECT_FLOAT_EQ(invalid.b, 0.5f);
}
}  // namespace lio

namespace lio {
TEST(RosInput, OusterUint32NanosecondOffsetsAndBigEndian) {
  auto msg = Cloud(true);
  auto& field = msg.fields.back();
  field.name = "t";
  field.datatype = sensor_msgs::msg::PointField::UINT32;
  for (size_t k = 0; k < 4; ++k) {
    Write(&msg.data, (k / 2) * 55 + (k % 2) * 25 + 17,
          static_cast<uint32_t>(k * 30000000), true);
  }
  InputConfig config;
  config.lidar_type = LidarType::kOuster;
  config.time_unit = "nanoseconds";
  config.time_reference = "relative";
  const auto scan = DecodeCloud(msg, config);
  ASSERT_EQ(scan.points.size(), 4u);
  EXPECT_EQ(scan.start_ns, 1795500000100000000LL);
  EXPECT_EQ(scan.points.back().relative_time_ns, 90000000);
  field.datatype = sensor_msgs::msg::PointField::FLOAT32;
  EXPECT_THROW(DecodeCloud(msg, config), std::invalid_argument);
}
TEST(RosInput, VelodyneFloatSecondsRelativeAndEndStamped) {
  auto msg = Cloud();
  auto& field = msg.fields.back();
  field.name = "time";
  field.datatype = sensor_msgs::msg::PointField::FLOAT32;
  for (size_t k = 0; k < 4; ++k) {
    Write(&msg.data, (k / 2) * 55 + (k % 2) * 25 + 17,
          static_cast<float>(k * .03), false);
  }
  InputConfig config;
  config.lidar_type = LidarType::kVelodyne;
  config.time_reference = "relative";
  config.header_stamp_reference = "scan_end";
  const auto scan = DecodeCloud(msg, config);
  EXPECT_EQ(scan.end_ns, 1795500000100000000LL);
  EXPECT_NEAR(scan.points.back().relative_time_ns, 90000000, 10);
}
TEST(RosInput, HesaiAbsoluteSecondsAndLivoxAbsoluteNanoseconds) {
  auto msg = Cloud();
  InputConfig config;
  config.lidar_type = LidarType::kHesai;
  const auto hesai = DecodeCloud(msg, config);
  ASSERT_EQ(hesai.points.size(), 4u);
  EXPECT_NEAR(SecondsBetween(hesai.end_ns, hesai.start_ns), .09f, 1e-6f);
  config.lidar_type = LidarType::kLivox;
  config.time_unit = "nanoseconds";
  for (size_t k = 0; k < 4; ++k) {
    Write(&msg.data, (k / 2) * 55 + (k % 2) * 25 + 17,
          1795500000000000000.0 + k * 30000000.0, false);
  }
  const auto livox = DecodeCloud(msg, config);
  EXPECT_NEAR(SecondsBetween(livox.end_ns, livox.start_ns), .09f, 1e-6f);
  EXPECT_NEAR(livox.start_ns, 1795500000000000000LL, 256);
}
TEST(RosInput, UnknownLidarTypeIsRejected) {
  InputConfig config;
  config.lidar_type = static_cast<LidarType>(99);
  EXPECT_THROW(DecodeCloud(Cloud(), config), std::invalid_argument);
}
}  // namespace lio
