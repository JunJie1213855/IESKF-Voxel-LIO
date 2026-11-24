#ifndef LIO_LIE_UTILS_H_
#define LIO_LIE_UTILS_H_

#include <cstdint>

#include <Eigen/Core>

namespace lio {
using Vec3 = Eigen::Vector3f;
using Mat3 = Eigen::Matrix3f;
using Basis32 = Eigen::Matrix<float, 3, 2>;
Mat3 Hat(const Vec3& v);
Mat3 ExpSO3(const Vec3& v);
Vec3 LogSO3(const Mat3& r);
Mat3 RightJacobianSO3(const Vec3& v);
Mat3 RightJacobianInverseSO3(const Vec3& v);
Basis32 GravityBasis(const Vec3& g);
Vec3 GravityPlus(const Vec3& g, const Eigen::Vector2f& delta);
Eigen::Vector2f GravityMinus(const Vec3& g, const Vec3& reference);
float SecondsBetween(int64_t end, int64_t begin);
}  // namespace lio
#endif
