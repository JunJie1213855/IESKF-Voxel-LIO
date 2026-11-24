#include "lio/Ieskf.h"

#include <Eigen/Cholesky>
#include <gtest/gtest.h>
namespace lio {
TEST(Filter, PlaneJacobianMatchesCentralDifferences) {
  State s;
  s.rotation = ExpSO3(Vec3(.2f, -.1f, .3f));
  s.extrinsic_rotation = ExpSO3(Vec3(-.1f, .15f, .2f));
  s.extrinsic_translation = Vec3(.3f, .2f, -.1f);
  const Vec3 p(2, -1, 3), n = Vec3(1, 2, 3).normalized();
  const auto h = PlaneJacobian(s, p, n);
  for (int k = 0; k < kStateDim; ++k) {
    ErrorVector d = ErrorVector::Zero();
    d[k] = 1e-3f;
    const float numerical =
        n.dot(WorldPoint(BoxPlus(s, d), p) - WorldPoint(BoxPlus(s, -d), p)) /
        (2e-3f);
    EXPECT_NEAR(h[k], numerical, 4e-4f) << k;
  }
}
TEST(Filter, IterationDoesNotMultiplyInformationAndExtrinsicsStayFrozen) {
  MapConfig mc;
  mc.max_layer = 0;
  mc.min_points_per_layer = {5};
  mc.max_points_per_node = mc.max_cov_points_per_node = 500;
  VoxelMap map(mc);
  std::vector<MapPoint> samples;
  PointCloud cloud;
  for (int i = -10; i <= 10; ++i)
    for (int j = -10; j <= 10; ++j) {
      Point p;
      p.position = Vec3(i * .15f + .01f, j * .15f + .01f, 0);
      cloud.push_back(p);
      MapPoint mp;
      mp.position = p.position;
      samples.push_back(mp);
    }
  map.Build(samples);
  FilterConfig ca;
  ca.max_iterations = 1;
  FilterConfig cb = ca;
  cb.max_iterations = 8;
  Ieskf a(ca), b(cb);
  State s;
  s.position.z() = .1f;
  a.Reset(s);
  b.Reset(s);
  const auto ra = a.Update(cloud, map), rb = b.Update(cloud, map);
  ASSERT_TRUE(ra.accepted) << ra.reason;
  ASSERT_TRUE(rb.accepted) << rb.reason;
  EXPECT_LT(std::abs(b.GetState().position.z()), .002f);
  EXPECT_NEAR(a.GetCovariance()(2, 2), b.GetCovariance()(2, 2), 1e-6f);
  EXPECT_EQ((b.GetState().extrinsic_rotation - s.extrinsic_rotation).norm(), 0);
  EXPECT_EQ(
      (b.GetState().extrinsic_translation - s.extrinsic_translation).norm(), 0);
  EXPECT_EQ((b.GetCovariance().block<6, 6>(6, 6).norm()), 0);
  EXPECT_GE(b.GetCovariance().diagonal().minCoeff(), 0);
  EXPECT_LT(rb.pose_information_ratio, 1e-5f);
}
TEST(Filter, RejectedObservationLeavesPredictionUntouched) {
  Ieskf f;
  VoxelMap map;
  const State before = f.GetState();
  const auto cov = f.GetCovariance();
  const auto r = f.Update({}, map);
  EXPECT_FALSE(r.accepted);
  EXPECT_EQ(BoxMinus(f.GetState(), before).norm(), 0);
  EXPECT_EQ((f.GetCovariance() - cov).norm(), 0);
}
}  // namespace lio
namespace lio {
TEST(Filter, LinearPlaneCovarianceMatchesSingleKalmanUpdate) {
  MapConfig mc;
  mc.max_layer = 0;
  mc.min_points_per_layer = {5};
  mc.max_points_per_node = mc.max_cov_points_per_node = 500;
  VoxelMap map(mc);
  std::vector<MapPoint> samples;
  PointCloud cloud;
  for (int i = -10; i <= 10; ++i)
    for (int j = -10; j <= 10; ++j) {
      Point p;
      p.position = Vec3(i * .15f, j * .15f, 0);
      cloud.push_back(p);
      MapPoint mp;
      mp.position = p.position;
      samples.push_back(mp);
    }
  map.Build(samples);
  Ieskf filter;
  const auto result = filter.Update(cloud, map);
  ASSERT_TRUE(result.accepted) << result.reason;
  const float expected = 1.0f / (1.0f + result.effective_points / (.1f * .1f));
  EXPECT_NEAR(filter.GetCovariance()(2, 2), expected, 5e-7f);
}
TEST(Filter, ProbabilisticObservationAndOnlineExtrinsics) {
  MapConfig mc;
  mc.max_layer = 0;
  mc.min_points_per_layer = {5};
  mc.max_points_per_node = mc.max_cov_points_per_node = 500;
  VoxelMap map(mc);
  std::vector<MapPoint> samples;
  PointCloud cloud;
  for (int i = -10; i <= 10; ++i)
    for (int j = -10; j <= 10; ++j) {
      Point p;
      p.position = Vec3(i * .15f + .01f, j * .15f + .01f, 0);
      cloud.push_back(p);
      MapPoint mp;
      mp.position = p.position;
      samples.push_back(mp);
    }
  map.Build(samples);
  FilterConfig c;
  c.measurement_model = "probabilistic";
  c.estimate_extrinsic = true;
  Ieskf filter(c);
  State s;
  s.position.z() = .05f;
  filter.Reset(s);
  const auto result = filter.Update(cloud, map);
  ASSERT_TRUE(result.accepted) << result.reason;
  EXPECT_LT(std::abs(filter.GetState().position.z()), .005f);
  Eigen::LLT<Covariance> solver(filter.GetCovariance());
  EXPECT_EQ(solver.info(), Eigen::Success);
}
}  // namespace lio
