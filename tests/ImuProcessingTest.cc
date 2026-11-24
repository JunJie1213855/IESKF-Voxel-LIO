#include "lio/ImuProcessing.h"

#include <gtest/gtest.h>
namespace lio {
TEST(Imu, StationaryPropagationAndContinuousNoiseScaling) {
  FilterConfig c;
  Ieskf a(c), b(c);
  ImuSample u;
  u.accel = Vec3(0, 0, 9.81f);
  for (int k = 0; k < 100; ++k) ASSERT_TRUE(a.Predict(u, .01f));
  for (int k = 0; k < 200; ++k) ASSERT_TRUE(b.Predict(u, .005f));
  EXPECT_LT(a.GetState().position.norm(), 1e-6f);
  EXPECT_LT(a.GetState().velocity.norm(), 1e-6f);
  EXPECT_LT(
      (a.GetCovariance() - b.GetCovariance()).norm() / a.GetCovariance().norm(),
      2e-4f);
  EXPECT_TRUE(a.GetCovariance().isApprox(a.GetCovariance().transpose(), 1e-6f));
  const auto before = a.GetCovariance();
  EXPECT_FALSE(a.Predict(u, -.01f));
  EXPECT_EQ((before - a.GetCovariance()).norm(), 0);
}
TEST(Imu, ReverseDeskewRotationTranslationAndLeverArm) {
  Ieskf filter;
  State s;
  s.velocity = Vec3(1, .2f, 0);
  s.extrinsic_translation = Vec3(.3f, -.1f, 0);
  s.extrinsic_rotation = ExpSO3(Vec3(.2f, 0, .1f));
  filter.Reset(s, 0);
  ImuProcessing imu;
  MeasurementGroup m;
  m.scan.start_ns = 0;
  m.scan.end_ns = 100000000;
  for (int k = 0; k <= 10; ++k) {
    ImuSample u;
    u.timestamp_ns = k * 10000000;
    u.accel = Vec3(0, 0, 9.81f);
    u.gyro = Vec3(0, 0, 1);
    m.imu.push_back(u);
  }
  const Vec3 world(4, 2, 1);
  for (int k = 0; k <= 10; ++k) {
    const float t = k * .01f;
    Point p;
    p.relative_time_ns = k * 10000000;
    p.position = s.extrinsic_rotation.transpose() *
                 (ExpSO3(Vec3(0, 0, t)).transpose() * (world - s.velocity * t) -
                  s.extrinsic_translation);
    m.scan.points.push_back(p);
  }
  ASSERT_TRUE(imu.Propagate(m, &filter));
  const auto cloud = imu.Deskew(m.scan, filter.GetState());
  const auto& end = filter.GetState();
  const Vec3 target = end.extrinsic_rotation.transpose() *
                      (end.rotation.transpose() * (world - end.position) -
                       end.extrinsic_translation);
  for (const auto& p : cloud) EXPECT_LT((p.position - target).norm(), 5e-5f);
  EXPECT_EQ(filter.Timestamp(), m.scan.end_ns);
}
TEST(Imu, GapsFailBeforePropagation) {
  Ieskf f;
  ImuProcessing imu;
  MeasurementGroup m;
  m.scan.start_ns = 0;
  m.scan.end_ns = 100000000;
  ImuSample a, b;
  a.timestamp_ns = 0;
  b.timestamp_ns = m.scan.end_ns;
  m.imu = {a, b};
  EXPECT_FALSE(imu.Propagate(m, &f));
  EXPECT_EQ(f.Timestamp(), 0);
}
TEST(Imu, StaticInitializationRejectsWrongUnits) {
  ImuConfig c;
  c.init_duration_sec = .1f;
  c.init_min_samples = 10;
  ImuProcessing imu(c);
  Ieskf f;
  MeasurementGroup m;
  m.scan.end_ns = 200000000;
  for (int k = 0; k <= 20; ++k) {
    ImuSample u;
    u.timestamp_ns = k * 10000000;
    u.accel = Vec3(0, 0, 1);
    m.imu.push_back(u);
  }
  EXPECT_FALSE(imu.Initialize(m, &f));
  imu.Reset();
  for (auto& u : m.imu) u.accel = Vec3(0, 0, 9.81f);
  ASSERT_TRUE(imu.Initialize(m, &f));
  EXPECT_LT(
      (f.GetState().rotation * Vec3(0, 0, 9.81f) + f.GetState().gravity).norm(),
      1e-5f);
}
}  // namespace lio
namespace lio {
TEST(Imu, DiscreteTransitionMatchesManifoldFiniteDifferences) {
  State s;
  s.rotation = ExpSO3(Vec3(.2f, -.3f, .1f));
  s.velocity = Vec3(1, 2, 3);
  s.gyro_bias = Vec3(.01f, -.02f, .005f);
  s.gravity = Vec3(.1f, .2f, -9.8f).normalized() * 9.81f;
  ImuSample u;
  u.gyro = Vec3(.4f, .2f, -.1f);
  u.accel = Vec3(1, 2, 9);
  const float dt = .02f;
  const State predicted = IntegrateImu(s, u, dt);
  const Covariance phi = TransitionJacobian(s, u, dt);
  for (int k = 0; k < kStateDim; ++k) {
    ErrorVector d = ErrorVector::Zero();
    d[k] = .001f;
    const ErrorVector numeric =
        (BoxMinus(IntegrateImu(BoxPlus(s, d), u, dt), predicted) -
         BoxMinus(IntegrateImu(BoxPlus(s, -d), u, dt), predicted)) /
        .002f;
    EXPECT_LT((numeric - phi.col(k)).norm(), 6e-4f) << k;
  }
}
}  // namespace lio
namespace lio {
TEST(Pipeline, MultiFrameAccelerationRotationAndVoxelUpdates) {
  PreprocessConfig pc;
  pc.point_stride = 1;
  pc.scan_voxel_size = .2f;
  Preprocess preprocess(pc);
  ImuConfig ic;
  ic.init_duration_sec = .2f;
  ic.init_min_samples = 30;
  ImuProcessing imu(ic);
  Ieskf filter;
  MapConfig mc;
  mc.min_points_per_layer = {5, 5, 5, 5};
  VoxelMap map(mc);
  const auto position = [](float t) -> Vec3 {
    const float d = std::max(0.0f, t - 1);
    return Vec3(.2f * d * d, 0, 1);
  };
  const auto rotation = [](float t) -> Mat3 {
    return ExpSO3(Vec3(0, 0, .1f * std::max(0.0f, t - 1)));
  };
  bool initialized = false;
  int accepted = 0;
  for (int frame = 0; frame < 30; ++frame) {
    MeasurementGroup group;
    group.scan.start_ns = int64_t(frame) * 100000000;
    group.scan.end_ns = group.scan.start_ns + 100000000;
    for (int k = 0; k <= 20; ++k) {
      ImuSample u;
      u.timestamp_ns = group.scan.start_ns + k * 5000000;
      const float t = static_cast<float>(u.timestamp_ns) * 1e-9f;
      u.gyro = Vec3(0, 0, t > 1 ? .1f : 0);
      u.accel = rotation(t).transpose() * Vec3(t > 1 ? .4f : 0, 0, 9.81f);
      group.imu.push_back(u);
    }
    int index = 0;
    for (int plane = 0; plane < 3; ++plane)
      for (int i = -6; i <= 6; ++i)
        for (int j = -6; j <= 6; ++j) {
          Point p;
          p.relative_time_ns = (index++ % 101) * 1000000;
          const float t =
              static_cast<float>(group.scan.start_ns + p.relative_time_ns) *
              1e-9f;
          Vec3 world;
          if (plane == 0)
            world = Vec3(i * .4f, j * .4f, 0);
          else if (plane == 1)
            world = Vec3(5, i * .4f, 2 + j * .25f);
          else
            world = Vec3(i * .4f, 5, 2 + j * .25f);
          p.position = rotation(t).transpose() * (world - position(t));
          group.scan.points.push_back(p);
        }
    if (!initialized) {
      initialized = imu.Initialize(group, &filter);
      continue;
    }
    ASSERT_TRUE(imu.Propagate(group, &filter));
    const PointCloud cloud =
        preprocess.Downsample(imu.Deskew(group.scan, filter.GetState()));
    map.Crop(filter.GetState().position);
    if (map.Size() == 0)
      map.Build(filter.MapPoints(cloud));
    else {
      const auto result = filter.Update(cloud, map);
      ASSERT_TRUE(result.accepted) << "frame=" << frame << " " << result.reason;
      ++accepted;
      map.Insert(filter.MapPoints(cloud));
    }
  }
  EXPECT_GT(accepted, 20);
  EXPECT_LT((filter.GetState().position - (position(3) - Vec3(0, 0, 1))).norm(),
            .05f);
  EXPECT_LT(LogSO3(rotation(3).transpose() * filter.GetState().rotation).norm(),
            .02f);
}
}  // namespace lio
namespace lio {
TEST(Imu, InvertedSensorInitializationHasDeterministicZeroHeading) {
  ImuConfig c;
  c.init_duration_sec = .1f;
  c.init_min_samples = 10;
  ImuProcessing imu(c);
  Ieskf filter;
  MeasurementGroup m;
  m.scan.end_ns = 200000000;
  const Vec3 accel = Vec3(.02f, -.03f, -.98f).normalized() * 9.81f;
  for (int k = 0; k <= 20; ++k) {
    ImuSample u;
    u.timestamp_ns = k * 10000000;
    u.accel = accel;
    m.imu.push_back(u);
  }
  ASSERT_TRUE(imu.Initialize(m, &filter));
  const auto& state = filter.GetState();
  EXPECT_LT((state.rotation * accel + state.gravity).norm(), 2e-5f);
  EXPECT_NEAR(std::atan2(state.rotation(1, 0), state.rotation(0, 0)), 0, 1e-5f);
}
}  // namespace lio
