#include <gtest/gtest.h>

#include "lio/Ieskf.h"
namespace lio {
TEST(Lie, RotationAndNearPiRoundTrip) {
  for (const Vec3& v : {Vec3(1e-6f, -2e-6f, 0), Vec3(0.2f, -0.4f, 0.3f),
                        Vec3(3.14158f, 0, 0)}) {
    const Mat3 r = ExpSO3(v);
    EXPECT_TRUE(ExpSO3(LogSO3(r)).isApprox(r, 2e-5f));
    EXPECT_TRUE((r.transpose() * r).isApprox(Mat3::Identity(), 2e-6f));
    EXPECT_TRUE((RightJacobianSO3(v) * RightJacobianInverseSO3(v))
                    .isApprox(Mat3::Identity(), 2e-5f));
  }
}
TEST(Lie, GravityAndStateRoundTrip) {
  State s;
  s.rotation = ExpSO3(Vec3(.2f, -.1f, .4f));
  s.gravity = Vec3(.3f, .4f, -9.7f).normalized() * 9.81f;
  ErrorVector d = ErrorVector::LinSpaced(-.01f, .02f);
  const State t = BoxPlus(s, d);
  EXPECT_NEAR(t.gravity.norm(), s.gravity.norm(), 2e-6f);
  EXPECT_TRUE(BoxMinus(t, s).isApprox(d, 2e-4f));
  EXPECT_NEAR((GravityBasis(s.gravity).transpose() * s.gravity).norm(), 0,
              2e-6f);
}
TEST(Lie, PriorDerivativeIncludesChartChanges) {
  State prior, current;
  prior.gravity = Vec3(0, 0, -9.81f);
  ErrorVector d = ErrorVector::Zero();
  d.segment<3>(3) = Vec3(.1f, .2f, -.15f);
  d.tail<2>() << .1f, -.08f;
  current = BoxPlus(prior, d);
  const Covariance j = PriorJacobian(current, prior);
  for (int k = 0; k < kStateDim; ++k) {
    ErrorVector eps = ErrorVector::Zero();
    eps[k] = 1e-3f;
    const ErrorVector numeric = (BoxMinus(BoxPlus(current, eps), prior) -
                                 BoxMinus(BoxPlus(current, -eps), prior)) /
                                (2e-3f);
    EXPECT_LT((numeric - j.col(k)).norm(), 5e-4f) << k;
  }
}
TEST(Lie, EpochTimeSubtractedBeforeFloatConversion) {
  const int64_t time = 1795500000000000000LL;
  EXPECT_NEAR(SecondsBetween(time + 1234567, time), .001234567f, 1e-9f);
}
}  // namespace lio
