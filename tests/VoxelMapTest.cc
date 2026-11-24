#include "lio/VoxelMap.h"

#include <gtest/gtest.h>
namespace lio {
TEST(Map, NegativeCoordinatesAdjacentQueriesAndPlaneCovariance) {
  MapConfig c;
  c.max_layer = 0;
  c.min_points_per_layer = {5};
  c.max_points_per_node = c.max_cov_points_per_node = 200;
  VoxelMap map(c);
  std::vector<MapPoint> samples;
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j) {
      MapPoint p;
      p.position = Vec3(-2.8f + .25f * i, -2.8f + .25f * j, -.01f);
      samples.push_back(p);
    }
  map.Build(samples);
  PlaneMatch plane;
  ASSERT_TRUE(map.QueryPlane(Vec3(-1, -1, .02f), &plane));
  EXPECT_NEAR(std::abs(plane.normal.z()), 1, 1e-5f);
  EXPECT_NEAR(plane.center.z(), -.01f, 1e-5f);
  EXPECT_TRUE(plane.covariance.allFinite());
  EXPECT_GE(plane.covariance.diagonal().minCoeff(), 0);
  map.Crop(Vec3(1000, 0, 0));
  EXPECT_EQ(map.Size(), 0u);
}
TEST(Map, CollinearAndRepeatedPointsAreNotPlanes) {
  MapConfig c;
  c.max_layer = 0;
  c.min_points_per_layer = {5};
  VoxelMap map(c);
  std::vector<MapPoint> samples;
  for (int i = 0; i < 20; ++i) {
    MapPoint p;
    p.position = Vec3(1 + .01f * i, 1, 1);
    samples.push_back(p);
  }
  map.Build(samples);
  PlaneMatch plane;
  EXPECT_FALSE(map.QueryPlane(Vec3(1, 1, 1), &plane));
  for (auto& p : samples) p.position = Vec3(1, 1, 1);
  map.Build(samples);
  EXPECT_FALSE(map.QueryPlane(Vec3(1, 1, 1), &plane));
}
TEST(Map, RootCapacityAndAxisAlignedNoiseRemainFinite) {
  MapConfig c;
  c.max_root_voxels = 2;
  VoxelMap map(c);
  std::vector<MapPoint> samples;
  for (int i = 0; i < 10; ++i) {
    MapPoint p;
    p.position = Vec3(i * 3.0f, 0, 0);
    samples.push_back(p);
  }
  map.Insert(samples);
  EXPECT_LE(map.Size(), 2u);
  for (const Vec3& v :
       {Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1), Vec3::Zero().eval()})
    EXPECT_TRUE(LidarCovariance(v, .04f, .1f).allFinite());
}
}  // namespace lio
namespace lio {
TEST(Map, BoundedSearchAgreesWithExhaustivePlaneDiskSearch) {
  MapConfig config;
  config.max_layer = 2;
  config.min_points_per_layer = {5, 5, 5};
  VoxelMap map(config);
  std::vector<MapPoint> points;
  for (int i = 0; i < 30; ++i)
    for (int j = 0; j < 30; ++j) {
      MapPoint a, b;
      a.position = Vec3(.4f, .05f + i * .095f, .05f + j * .095f);
      b.position = Vec3(.05f + i * .095f, .05f + j * .095f, 1.7f);
      points.push_back(a);
      points.push_back(b);
    }
  map.Build(points);
  const auto planes = map.Planes();
  ASSERT_GT(planes.size(), 5u);
  for (int k = 0; k < 300; ++k) {
    const Vec3 point(-.1f + 3.2f * ((k * 17) % 307) / 307.0f,
                     -.1f + 3.2f * ((k * 43) % 311) / 311.0f,
                     -.1f + 3.2f * ((k * 79) % 313) / 313.0f);
    constexpr float kLimit = .25f;
    float expected = kLimit;
    bool found = false;
    for (const auto& plane : planes) {
      const Vec3 d = point - plane.center;
      const float r = plane.normal.dot(d);
      if ((d - plane.normal * r).norm() <= plane.radius &&
          std::abs(r) < expected) {
        expected = std::abs(r);
        found = true;
      }
    }
    PlaneMatch match;
    const bool actual = map.QueryPlane(point, &match, kLimit);
    ASSERT_EQ(actual, found) << k;
    if (actual)
      EXPECT_NEAR(std::abs(match.normal.dot(point - match.center)), expected,
                  1e-5f)
          << k;
  }
}
}  // namespace lio
