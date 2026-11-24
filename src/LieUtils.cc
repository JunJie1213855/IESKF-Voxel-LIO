#include "lio/LieUtils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <Eigen/Geometry>
namespace lio {
Mat3 Hat(const Vec3& v) {
  Mat3 m;
  m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
  return m;
}
Mat3 ExpSO3(const Vec3& v) {
  const float t2 = v.squaredNorm();
  const Mat3 w = Hat(v);
  if (t2 < 1e-8f)
    return Mat3::Identity() + (1 - t2 / 6) * w + (0.5f - t2 / 24) * w * w;
  const float t = std::sqrt(t2);
  return Mat3::Identity() + std::sin(t) / t * w +
         (1 - std::cos(t)) / t2 * w * w;
}
Vec3 LogSO3(const Mat3& r) {
  Eigen::Quaternionf q(r);
  q.normalize();
  if (q.w() < 0) q.coeffs() *= -1;
  const float n = q.vec().norm();
  if (n < 1e-6f) return 2 * q.vec();
  return (2 * std::atan2(n, q.w()) / n) * q.vec();
}
Mat3 RightJacobianSO3(const Vec3& v) {
  const float t2 = v.squaredNorm();
  const Mat3 w = Hat(v);
  if (t2 < 1e-6f)
    return Mat3::Identity() - (0.5f - t2 / 24) * w +
           (1.0f / 6 - t2 / 120) * w * w;
  const float t = std::sqrt(t2);
  return Mat3::Identity() - (1 - std::cos(t)) / t2 * w +
         (t - std::sin(t)) / (t2 * t) * w * w;
}
Mat3 RightJacobianInverseSO3(const Vec3& v) {
  const float t2 = v.squaredNorm();
  const Mat3 w = Hat(v);
  if (t2 < 1e-6f)
    return Mat3::Identity() + 0.5f * w + (1.0f / 12 + t2 / 720) * w * w;
  const float t = std::sqrt(t2);
  return Mat3::Identity() + 0.5f * w +
         (1 - 0.5f * t / std::tan(0.5f * t)) / t2 * w * w;
}
Basis32 GravityBasis(const Vec3& g) {
  if (!g.allFinite() || g.norm() < 1e-6f)
    throw std::invalid_argument("invalid gravity");
  const Vec3 u = g.normalized();
  // 选择与重力最不平行的轴建立正交切基，避免接近坐标轴时除零。
  Eigen::Index axis;
  u.cwiseAbs().minCoeff(&axis);
  const Vec3 a = Vec3::Unit(axis);
  Basis32 b;
  b.col(0) = (a - u * u.dot(a)).normalized();
  b.col(1) = u.cross(b.col(0));
  return b;
}
Vec3 GravityPlus(const Vec3& g, const Eigen::Vector2f& delta) {
  const float t = delta.norm();
  const Vec3 tangent = GravityBasis(g) * delta;
  const float sinc = t < 1e-4f ? 1 - t * t / 6 : std::sin(t) / t;
  return (std::cos(t) * g + g.norm() * sinc * tangent).normalized() * g.norm();
}
Eigen::Vector2f GravityMinus(const Vec3& g, const Vec3& reference) {
  const Vec3 u = reference.normalized(), v = g.normalized();
  const float c = std::clamp(u.dot(v), -1.0f, 1.0f);
  const Vec3 t = v - c * u;
  const float s = t.norm();
  if (s < 1e-6f) {
    if (c < 0) throw std::runtime_error("antipodal gravity prior");
    return GravityBasis(reference).transpose() * t;
  }
  return GravityBasis(reference).transpose() * (std::atan2(s, c) / s * t);
}
float SecondsBetween(int64_t end, int64_t begin) {
  // 绝对纳秒时间戳先相减，严禁先转 float，否则会丢失毫秒级运动时间。
  return static_cast<float>(end - begin) * 1e-9f;
}
}  // namespace lio
