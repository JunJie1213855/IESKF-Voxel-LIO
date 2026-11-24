#include "lio/Ieskf.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/LU>
namespace lio {
void FilterConfig::Validate() const {
  if (max_iterations < 1 || max_iterations > 30 || min_effective_points < 3 ||
      (measurement_model != "constant" && measurement_model != "probabilistic"))
    throw std::invalid_argument("invalid filter mode/count");
  for (float v :
       {rotation_convergence_rad, translation_convergence_m, plane_residual_std,
        lidar_range_std, lidar_angle_std_deg, residual_sigma_gate,
        min_residual_std, max_plane_distance, gyro_noise_density,
        accel_noise_density, gyro_bias_random_walk, accel_bias_random_walk})
    if (!std::isfinite(v) || v <= 0)
      throw std::invalid_argument("filter scales must be finite and positive");
}
State BoxPlus(const State& s, const ErrorVector& d) {
  State r = s;
  r.position += d.segment<3>(0);
  r.rotation = s.rotation * ExpSO3(d.segment<3>(3));
  r.extrinsic_rotation = s.extrinsic_rotation * ExpSO3(d.segment<3>(6));
  r.extrinsic_translation += d.segment<3>(9);
  r.velocity += d.segment<3>(12);
  r.gyro_bias += d.segment<3>(15);
  r.accel_bias += d.segment<3>(18);
  r.gravity = GravityPlus(s.gravity, d.tail<2>());
  return r;
}
ErrorVector BoxMinus(const State& s, const State& r) {
  ErrorVector d;
  d.segment<3>(0) = s.position - r.position;
  d.segment<3>(3) = LogSO3(r.rotation.transpose() * s.rotation);
  d.segment<3>(6) =
      LogSO3(r.extrinsic_rotation.transpose() * s.extrinsic_rotation);
  d.segment<3>(9) = s.extrinsic_translation - r.extrinsic_translation;
  d.segment<3>(12) = s.velocity - r.velocity;
  d.segment<3>(15) = s.gyro_bias - r.gyro_bias;
  d.segment<3>(18) = s.accel_bias - r.accel_bias;
  d.tail<2>() = GravityMinus(s.gravity, r.gravity);
  return d;
}
Covariance PriorJacobian(const State& s, const State& r) {
  Covariance j = Covariance::Identity();
  j.block<3, 3>(3, 3) =
      RightJacobianInverseSO3(LogSO3(r.rotation.transpose() * s.rotation));
  j.block<3, 3>(6, 6) = RightJacobianInverseSO3(
      LogSO3(r.extrinsic_rotation.transpose() * s.extrinsic_rotation));
  // 只对显式二维重力图的坐标变换做差分，不引入通用流形模板。
  constexpr float kEpsilon = 1e-3f;
  for (int k = 0; k < 2; ++k) {
    Eigen::Vector2f d = Eigen::Vector2f::Zero();
    d[k] = kEpsilon;
    j.block<2, 1>(21, 21 + k) =
        (GravityMinus(GravityPlus(s.gravity, d), r.gravity) -
         GravityMinus(GravityPlus(s.gravity, -d), r.gravity)) /
        (2 * kEpsilon);
  }
  return j;
}
Vec3 WorldPoint(const State& s, const Vec3& p) {
  return s.rotation * (s.extrinsic_rotation * p + s.extrinsic_translation) +
         s.position;
}
Eigen::Matrix<float, 1, kStateDim> PlaneJacobian(const State& s, const Vec3& p,
                                                 const Vec3& n) {
  Eigen::Matrix<float, 1, kStateDim> h =
      Eigen::Matrix<float, 1, kStateDim>::Zero();
  h.segment<3>(0) = n.transpose();
  h.segment<3>(3) = -n.transpose() * s.rotation *
                    Hat(s.extrinsic_rotation * p + s.extrinsic_translation);
  h.segment<3>(6) = -n.transpose() * s.rotation * s.extrinsic_rotation * Hat(p);
  h.segment<3>(9) = n.transpose() * s.rotation;
  return h;
}
State IntegrateImu(const State& s, const ImuSample& u, float dt) {
  const Vec3 omega = u.gyro - s.gyro_bias;
  const Mat3 midpoint_rotation = s.rotation * ExpSO3(omega * (0.5f * dt));
  const Vec3 acceleration =
      midpoint_rotation * (u.accel - s.accel_bias) + s.gravity;
  State next = s;
  next.position += s.velocity * dt + 0.5f * acceleration * dt * dt;
  next.velocity += acceleration * dt;
  next.rotation = Eigen::Quaternionf(s.rotation * ExpSO3(omega * dt))
                      .normalized()
                      .toRotationMatrix();
  return next;
}
Covariance TransitionJacobian(const State& s, const ImuSample& u, float dt) {
  const Vec3 omega = u.gyro - s.gyro_bias, accel = u.accel - s.accel_bias;
  const Mat3 half_rotation = ExpSO3(omega * (0.5f * dt));
  const Mat3 midpoint_rotation = s.rotation * half_rotation;
  const Mat3 acceleration_rotation =
      -midpoint_rotation * Hat(accel) * half_rotation.transpose();
  const Mat3 acceleration_bias = midpoint_rotation * Hat(accel) *
                                 RightJacobianSO3(omega * (0.5f * dt)) *
                                 (0.5f * dt);
  Covariance phi = Covariance::Identity();
  // 离散积分的解析导数：姿态误差随右切空间旋转，不能简单使用 I + F dt。
  phi.block<3, 3>(0, 3) = 0.5f * dt * dt * acceleration_rotation;
  phi.block<3, 3>(0, 12) = Mat3::Identity() * dt;
  phi.block<3, 3>(0, 15) = 0.5f * dt * dt * acceleration_bias;
  phi.block<3, 3>(0, 18) = -0.5f * dt * dt * midpoint_rotation;
  phi.block<3, 2>(0, 21) =
      0.5f * dt * dt * s.gravity.norm() * GravityBasis(s.gravity);
  phi.block<3, 3>(3, 3) = ExpSO3(-omega * dt);
  phi.block<3, 3>(3, 15) = -RightJacobianSO3(omega * dt) * dt;
  phi.block<3, 3>(12, 3) = dt * acceleration_rotation;
  phi.block<3, 3>(12, 15) = dt * acceleration_bias;
  phi.block<3, 3>(12, 18) = -dt * midpoint_rotation;
  phi.block<3, 2>(12, 21) = dt * s.gravity.norm() * GravityBasis(s.gravity);
  return phi;
}
Ieskf::Ieskf(const FilterConfig& c) : config_(c) {
  c.Validate();
  Reset();
}
void Ieskf::Reset(const State& s, int64_t timestamp) {
  if (!s.position.allFinite() || !s.gravity.allFinite() ||
      s.gravity.norm() < 1e-6f)
    throw std::invalid_argument("invalid reset state");
  state_ = s;
  timestamp_ns_ = timestamp;
  covariance_.setZero();
  covariance_.diagonal().segment<3>(0).setConstant(1.0f);
  covariance_.diagonal().segment<3>(3).setConstant(0.01f);
  if (config_.estimate_extrinsic)
    covariance_.diagonal().segment<6>(6).setConstant(1e-4f);
  covariance_.diagonal().segment<3>(12).setConstant(1.0f);
  covariance_.diagonal().segment<3>(15).setConstant(1e-4f);
  covariance_.diagonal().segment<3>(18).setConstant(0.01f);
  covariance_.diagonal().tail<2>().setConstant(1e-4f);
}
bool Ieskf::Predict(const ImuSample& u, float dt) {
  if (!std::isfinite(dt) || dt <= 0 || dt > 0.1f || !u.accel.allFinite() ||
      !u.gyro.allFinite())
    return false;
  const Vec3 w = u.gyro - state_.gyro_bias, a = u.accel - state_.accel_bias;
  const Mat3 rm = state_.rotation * ExpSO3(w * (0.5f * dt));
  const State next = IntegrateImu(state_, u, dt);
  Covariance f = Covariance::Zero();
  f.block<3, 3>(0, 12) = Mat3::Identity();
  f.block<3, 3>(3, 3) = -Hat(w);
  f.block<3, 3>(3, 15) = -Mat3::Identity();
  f.block<3, 3>(12, 3) = -rm * Hat(a);
  f.block<3, 3>(12, 18) = -rm;
  f.block<3, 2>(12, 21) = state_.gravity.norm() * GravityBasis(state_.gravity);
  Eigen::Matrix<float, kStateDim, 12> g =
      Eigen::Matrix<float, kStateDim, 12>::Zero();
  g.block<3, 3>(3, 0) = -Mat3::Identity();
  g.block<3, 3>(12, 3) = -rm;
  g.block<3, 3>(15, 6) = Mat3::Identity();
  g.block<3, 3>(18, 9) = Mat3::Identity();
  Eigen::Matrix<float, 12, 1> q;
  q.segment<3>(0).setConstant(config_.gyro_noise_density *
                              config_.gyro_noise_density);
  q.segment<3>(3).setConstant(config_.accel_noise_density *
                              config_.accel_noise_density);
  q.segment<3>(6).setConstant(config_.gyro_bias_random_walk *
                              config_.gyro_bias_random_walk);
  q.segment<3>(9).setConstant(config_.accel_bias_random_walk *
                              config_.accel_bias_random_walk);
  const Covariance phi = TransitionJacobian(state_, u, dt);
  // 连续白噪声用中点近似积分：Qd = L Qc L^T dt；噪声密度只平方一次。
  const Eigen::Matrix<float, kStateDim, 12> l =
      (Covariance::Identity() + 0.5f * f * dt) * g;
  Covariance p = phi * covariance_ * phi.transpose() +
                 dt * l * q.asDiagonal() * l.transpose();
  p = (0.5f * (p + p.transpose())).eval();
  if (!next.position.allFinite() || !next.velocity.allFinite() ||
      !next.rotation.allFinite() || !p.allFinite())
    return false;
  state_ = next;
  covariance_ = p;
  return true;
}
UpdateResult Ieskf::Update(const PointCloud& cloud, const VoxelMap& map) {
  UpdateResult result;
  const State prior = state_;
  const Covariance prior_cov = covariance_;
  // 固定外参时仅求解 17 维活动子块，避免冻结块导致全维先验奇异。
  std::vector<int> active;
  for (int i = 0; i < kStateDim; ++i)
    if (config_.estimate_extrinsic || i < 6 || i >= 12) active.push_back(i);
  const int dim = static_cast<int>(active.size());
  const auto submatrix = [&](const Covariance& m) {
    Eigen::MatrixXf out(dim, dim);
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j) out(i, j) = m(active[i], active[j]);
    return out;
  };
  const Eigen::MatrixXf p0 = submatrix(prior_cov);
  Eigen::LDLT<Eigen::MatrixXf> prior_solver(p0);
  if (prior_solver.info() != Eigen::Success || !prior_solver.isPositive()) {
    result.reason = "invalid prior covariance";
    return result;
  }
  std::vector<PlaneMatch> cached(cloud.size());
  std::vector<bool> valid(cloud.size(), false);
  State current = prior;
  Covariance measurement_information = Covariance::Zero();
  const auto linearize = [&](const State& s, bool rematch, Eigen::MatrixXf* a,
                             Eigen::VectorXf* b) {
    const Covariance full_j = PriorJacobian(s, prior);
    const Eigen::MatrixXf j = submatrix(full_j);
    const ErrorVector full_e = BoxMinus(s, prior);
    Eigen::VectorXf e(dim);
    for (int k = 0; k < dim; ++k) e[k] = full_e[active[k]];
    // 每轮都从同一个传播先验构建代价，不能使用上一轮的后验继续融合本帧。
    *a = j.transpose() * prior_solver.solve(j);
    *b = -j.transpose() * prior_solver.solve(e);
    // J 将当前切空间误差映射到先验切空间；门限所用的 P 也必须做相同坐标变换。
    Eigen::PartialPivLU<Eigen::MatrixXf> chart_solver(j);
    const Eigen::MatrixXf transport =
        chart_solver.solve(Eigen::MatrixXf::Identity(dim, dim));
    const Eigen::MatrixXf tangent_cov = transport * p0 * transport.transpose();
    measurement_information.setZero();
    result.effective_points = 0;
    result.mean_abs_residual = 0;
    for (size_t i = 0; i < cloud.size(); ++i) {
      const Vec3& p = cloud[i].position;
      const Vec3 world = WorldPoint(s, p);
      if (rematch)
        valid[i] =
            map.QueryPlane(world, &cached[i], config_.max_plane_distance);
      if (!valid[i]) continue;
      const auto& plane = cached[i];
      const float residual = plane.normal.dot(world - plane.center);
      if (!std::isfinite(residual) ||
          std::abs(residual) > config_.max_plane_distance)
        continue;
      const auto h = PlaneJacobian(s, p, plane.normal);
      Eigen::VectorXf ha(dim);
      for (int k = 0; k < dim; ++k) ha[k] = h[active[k]];
      float variance = config_.plane_residual_std * config_.plane_residual_std;
      if (config_.measurement_model == "probabilistic") {
        const Mat3 r = s.rotation * s.extrinsic_rotation;
        const Mat3 cov = r *
                         LidarCovariance(p, config_.lidar_range_std,
                                         config_.lidar_angle_std_deg) *
                         r.transpose();
        Eigen::Matrix<float, 1, 6> jp;
        // 显式写入固定 6 维行向量，避免 Eigen 逗号初始化触发越界向量化警告。
        for (int k = 0; k < 3; ++k) {
          jp[k] = world[k] - plane.center[k];
          jp[k + 3] = -plane.normal[k];
        }
        variance = (plane.normal.transpose() * cov * plane.normal)(0, 0) +
                   (jp * plane.covariance * jp.transpose())(0, 0);
        variance = std::max(
            variance, config_.min_residual_std * config_.min_residual_std);
        // 创新门限可包含预测不确定性；构建观测权重时 R 仍只含测量和平面噪声。
        const float predicted = ha.dot(tangent_cov * ha);
        if (std::abs(residual) >
            config_.residual_sigma_gate *
                std::sqrt(variance + std::max(0.0f, predicted)))
          continue;
      }
      variance = std::max(variance,
                          config_.min_residual_std * config_.min_residual_std);
      a->noalias() += ha * ha.transpose() / variance;
      b->noalias() -= ha * (residual / variance);
      measurement_information.noalias() += h.transpose() * h / variance;
      ++result.effective_points;
      result.mean_abs_residual += std::abs(residual);
    }
    if (result.effective_points)
      result.mean_abs_residual /= result.effective_points;
  };
  try {
    Eigen::MatrixXf a;
    Eigen::VectorXf b;
    for (int iteration = 0; iteration < config_.max_iterations; ++iteration) {
      linearize(current, iteration == 0 || config_.rematch_each_iteration, &a,
                &b);
      result.iterations = iteration + 1;
      if (result.effective_points < config_.min_effective_points) {
        result.reason = "insufficient plane matches";
        return result;
      }
      Eigen::LDLT<Eigen::MatrixXf> solver(a);
      if (solver.info() != Eigen::Success || !solver.isPositive()) {
        result.reason = "observation factorization failed";
        return result;
      }
      const Eigen::VectorXf d = solver.solve(b);
      if (!d.allFinite()) {
        result.reason = "nonfinite correction";
        return result;
      }
      ErrorVector delta = ErrorVector::Zero();
      for (int k = 0; k < dim; ++k) delta[active[k]] = d[k];
      if (delta.segment<3>(0).norm() > 5 || delta.segment<3>(3).norm() > 0.5f ||
          delta.tail<2>().norm() > 0.5f) {
        result.reason = "correction exceeds trust region";
        return result;
      }
      current = BoxPlus(current, delta);
      if (delta.segment<3>(0).norm() < config_.translation_convergence_m &&
          delta.segment<3>(3).norm() < config_.rotation_convergence_rad &&
          delta.segment<3>(6).norm() < config_.rotation_convergence_rad &&
          delta.segment<3>(9).norm() < config_.translation_convergence_m &&
          delta.segment<3>(12).norm() < 1e-3f &&
          delta.segment<6>(15).norm() < 1e-3f &&
          delta.tail<2>().norm() < 1e-4f) {
        result.converged = true;
        break;
      }
    }
    // 最终状态处重新线性化：后验已处于新切空间，等价地包含误差注入后的坐标变换。
    // 因此不能再重复执行旧切空间协方差的 reset，也不能每轮收缩一次协方差。
    linearize(current, config_.rematch_each_iteration, &a, &b);
    if (result.effective_points < config_.min_effective_points) {
      result.reason = "insufficient final matches";
      return result;
    }
    Eigen::LDLT<Eigen::MatrixXf> solver(a);
    if (solver.info() != Eigen::Success || !solver.isPositive()) {
      result.reason = "posterior factorization failed";
      return result;
    }
    const Eigen::MatrixXf post =
        solver.solve(Eigen::MatrixXf::Identity(dim, dim));
    if (!post.allFinite()) {
      result.reason = "nonfinite posterior";
      return result;
    }
    Covariance p = Covariance::Zero();
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j) p(active[i], active[j]) = post(i, j);
    p = (0.5f * (p + p.transpose())).eval();
    Eigen::Matrix<float, 6, 6> info =
        measurement_information.topLeftCorner<6, 6>();
    Eigen::Matrix<float, 6, 1> scales;
    for (int i = 0; i < 6; ++i)
      scales[i] = 1 / std::sqrt(std::max(info(i, i), 1e-12f));
    info = scales.asDiagonal() * info * scales.asDiagonal();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6>> eig(info);
    if (eig.info() == Eigen::Success)
      result.pose_information_ratio = std::max(0.0f, eig.eigenvalues()[0]) /
                                      std::max(eig.eigenvalues()[5], 1e-12f);
    state_ = current;
    covariance_ = p;
    result.accepted = true;
    result.reason = result.converged ? "converged" : "iteration limit";
  } catch (const std::exception& e) {
    result.reason = e.what();
  }
  return result;
}
std::vector<MapPoint> Ieskf::MapPoints(const PointCloud& cloud) const {
  std::vector<MapPoint> out;
  out.reserve(cloud.size());
  const Mat3 r = state_.rotation * state_.extrinsic_rotation;
  for (const auto& point : cloud) {
    MapPoint p;
    p.position = WorldPoint(state_, point.position);
    Eigen::Matrix<float, 3, kStateDim> j =
        Eigen::Matrix<float, 3, kStateDim>::Zero();
    j.block<3, 3>(0, 0) = Mat3::Identity();
    j.block<3, 3>(0, 3) =
        -state_.rotation * Hat(state_.extrinsic_rotation * point.position +
                               state_.extrinsic_translation);
    j.block<3, 3>(0, 6) = -r * Hat(point.position);
    j.block<3, 3>(0, 9) = state_.rotation;
    // 地图点额外携带插入时状态不确定性，使用完整交叉协方差传播。
    p.covariance = r *
                       LidarCovariance(point.position, config_.lidar_range_std,
                                       config_.lidar_angle_std_deg) *
                       r.transpose() +
                   j * covariance_ * j.transpose();
    if (p.position.allFinite() && p.covariance.allFinite()) out.push_back(p);
  }
  return out;
}
}  // namespace lio
