#include "lio/Preprocess.h"

#include <future>

#include <gtest/gtest.h>
namespace lio {
namespace {
LidarScan Scan(int64_t begin, int64_t end) {
  LidarScan s;
  s.start_ns = begin;
  s.end_ns = end;
  Point p;
  p.position = Vec3(2, 0, 0);
  p.relative_time_ns = end - begin;
  s.points.push_back(p);
  return s;
}
ImuSample Sample(int64_t t) {
  ImuSample s;
  s.timestamp_ns = t;
  return s;
}
}  // namespace
TEST(Sync, WaitsForRightBoundaryAndRetainsIt) {
  Preprocess p;
  p.AddImu(Sample(0));
  p.AddImu(Sample(60000000));
  p.AddLidar(Scan(0, 100000000));
  MeasurementGroup g;
  EXPECT_FALSE(p.SyncMeasurements(&g));
  p.AddImu(Sample(120000000));
  ASSERT_TRUE(p.SyncMeasurements(&g));
  EXPECT_EQ(g.imu.back().timestamp_ns, 120000000);
  p.AddLidar(Scan(100000000, 200000000));
  p.AddImu(Sample(210000000));
  ASSERT_TRUE(p.SyncMeasurements(&g));
  EXPECT_EQ(g.imu.front().timestamp_ns, 60000000);
  EXPECT_EQ(g.imu.back().timestamp_ns, 210000000);
}
TEST(Sync, TimeReversalNotifiesCoordinatedReset) {
  Preprocess p;
  p.AddImu(Sample(100000000));
  p.AddImu(Sample(0));
  p.AddImu(Sample(20000000));
  p.AddLidar(Scan(0, 10000000));
  MeasurementGroup g;
  ASSERT_TRUE(p.SyncMeasurements(&g));
  EXPECT_TRUE(g.reset);
  EXPECT_EQ(p.Stats().resets, 1u);
}
TEST(Sync, StopUnblocksEstimator) {
  Preprocess p;
  auto future = std::async(std::launch::async, [&] {
    MeasurementGroup g;
    return p.WaitMeasurements(&g);
  });
  p.Stop();
  EXPECT_FALSE(future.get());
}
TEST(Sync, CapacityAndDownsampleAreBounded) {
  PreprocessConfig c;
  c.max_imu_samples = 2;
  c.scan_voxel_size = 1;
  Preprocess p(c);
  p.AddImu(Sample(0));
  p.AddImu(Sample(1));
  p.AddImu(Sample(2));
  EXPECT_EQ(p.Stats().imu_dropped, 1u);
  Point a, b, d;
  a.position = Vec3(-.1f, 0, 0);
  b.position = Vec3(.1f, 0, 0);
  d.position = Vec3(.4f, 0, 0);
  auto out = p.Downsample({a, b, d});
  EXPECT_EQ(out.size(), 2u);
}
}  // namespace lio
namespace lio {
TEST(Sync, SmallScanOverlapTrimsOldPointsAndRebasesTime) {
  LidarScan scan;
  scan.start_ns = 99900000;
  scan.end_ns = 200000000;
  for (int64_t t : {0LL, 100000LL, 100100000LL}) {
    Point point;
    point.relative_time_ns = t;
    scan.points.push_back(point);
  }
  EXPECT_EQ(TrimScanBefore(100000000, &scan), 1u);
  ASSERT_EQ(scan.points.size(), 2u);
  EXPECT_EQ(scan.start_ns, 100000000);
  EXPECT_EQ(scan.points.front().relative_time_ns, 0);
  EXPECT_EQ(scan.points.back().relative_time_ns, 100000000);
  EXPECT_THROW(TrimScanBefore(102000000, &scan), std::invalid_argument);
}
}  // namespace lio
