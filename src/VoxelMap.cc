#include "lio/VoxelMap.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/Eigenvalues>
namespace lio {
void MapConfig::Validate() const {
  if (!std::isfinite(root_voxel_size) || root_voxel_size <= 0 ||
      max_layer < 0 || max_layer > 8 ||
      min_points_per_layer.size() != static_cast<size_t>(max_layer + 1) ||
      !std::isfinite(plane_max_min_eigenvalue) ||
      plane_max_min_eigenvalue <= 0 ||
      !std::isfinite(plane_max_eigenvalue_ratio) ||
      plane_max_eigenvalue_ratio <= 0 || plane_max_eigenvalue_ratio >= 1 ||
      max_points_per_node < 3 || max_cov_points_per_node < 3 ||
      max_cov_points_per_node > max_points_per_node ||
      neighbor_voxel_radius < 0 || neighbor_voxel_radius > 3 ||
      !std::isfinite(local_map_radius) || local_map_radius <= 0 ||
      max_root_voxels < 1)
    throw std::invalid_argument("invalid voxel map parameters");
  for (int n : min_points_per_layer)
    if (n < 3 || n > max_points_per_node || n > max_cov_points_per_node)
      throw std::invalid_argument("invalid plane sample count");
}
Mat3 LidarCovariance(const Vec3& p, float sr, float sa_deg) {
  if (p.norm() < 1e-6f) return Mat3::Identity() * sr * sr;
  const Vec3 u = p.normalized();
  const float sa = sa_deg * 0.017453292519943295f;
  const float tangential_std = p.norm() * std::sin(sa);
  return sr * sr * u * u.transpose() +
         tangential_std * tangential_std *
             (Mat3::Identity() - u * u.transpose());
}
struct VoxelMap::Node {
  Vec3 center;
  float size;
  int layer;
  bool split = false, valid = false;
  bool has_bounds = false;
  Vec3 bound_min = Vec3::Zero();
  Vec3 bound_max = Vec3::Zero();
  uint64_t seen = 0;
  std::vector<MapPoint> samples;
  PlaneMatch plane;
  std::array<std::unique_ptr<Node>, 8> children;
  Node(Vec3 c, float s, int l) : center(std::move(c)), size(s), layer(l) {}
};
VoxelMap::VoxelMap(const MapConfig& c) : config_(c) { c.Validate(); }
VoxelMap::~VoxelMap() = default;
size_t VoxelMap::Hash::operator()(const Key& k) const {
  // 使用无符号哈希混合，避免负体素索引参与有符号乘法溢出。
  size_t h = std::hash<int64_t>{}(k.x);
  h ^= std::hash<int64_t>{}(k.y) + 0x9e3779b9u + (h << 6) + (h >> 2);
  h ^= std::hash<int64_t>{}(k.z) + 0x9e3779b9u + (h << 6) + (h >> 2);
  return h;
}
VoxelMap::Key VoxelMap::Index(const Vec3& p) const {
  const Vec3 k = (p / config_.root_voxel_size).array().floor();
  if (!k.allFinite() || k.cwiseAbs().maxCoeff() > 1e12f)
    throw std::invalid_argument("map coordinate out of range");
  return {static_cast<int64_t>(k.x()), static_cast<int64_t>(k.y()),
          static_cast<int64_t>(k.z())};
}
void VoxelMap::Clear() { roots_.clear(); }
void VoxelMap::Build(const std::vector<MapPoint>& points) {
  Clear();
  Insert(points);
}
void VoxelMap::FitPlane(Node* n) {
  n->valid = false;
  n->has_bounds = false;
  if (n->samples.size() <
      static_cast<size_t>(config_.min_points_per_layer[n->layer]))
    return;
  // 围绕体素局部中心累计，避免 float32 在大世界坐标下二阶矩相减损失精度。
  Vec3 mean = Vec3::Zero();
  for (const auto& p : n->samples) mean += p.position - n->center;
  mean /= static_cast<float>(n->samples.size());
  Mat3 cov = Mat3::Zero();
  for (const auto& p : n->samples) {
    const Vec3 d = (p.position - n->center) - mean;
    cov.noalias() += d * d.transpose();
  }
  cov /= static_cast<float>(n->samples.size());
  Eigen::SelfAdjointEigenSolver<Mat3> eig(cov);
  if (eig.info() != Eigen::Success || !eig.eigenvalues().allFinite()) return;
  const auto values = eig.eigenvalues();
  // 中间特征值过小表示线状/重复点，不能只凭最小特征值小就判为平面。
  if (values[1] < 1e-6f || values[0] > config_.plane_max_min_eigenvalue ||
      std::max(0.0f, values[0]) / values[1] >
          config_.plane_max_eigenvalue_ratio)
    return;
  const Mat3 vectors = eig.eigenvectors();
  Vec3 normal = vectors.col(0);
  Eigen::Index dominant;
  normal.cwiseAbs().maxCoeff(&dominant);
  const float sign = normal[dominant] < 0 ? -1.0f : 1.0f;
  normal *= sign;
  PlaneMatch plane;
  plane.normal = normal;
  plane.center = n->center + mean;
  plane.layer = n->layer;
  const float count = static_cast<float>(n->samples.size());
  float radius2 = 0;
  for (const auto& p : n->samples) {
    const Vec3 d = (p.position - n->center) - mean;
    radius2 = std::max(radius2, (d - normal * normal.dot(d)).squaredNorm());
    Eigen::Matrix<float, 6, 3> j = Eigen::Matrix<float, 6, 3>::Zero();
    // 对最小特征向量求导，叠加各保留点的协方差，得到法向量/中心联合协方差。
    for (int k = 1; k < 3; ++k) {
      const Vec3 vk = vectors.col(k);
      const Eigen::RowVector3f a = (d.dot(vectors.col(0)) * vk.transpose() +
                                    d.dot(vk) * vectors.col(0).transpose()) /
                                   (count * (values[0] - values[k]));
      j.topRows<3>().noalias() += sign * vk * a;
    }
    j.bottomRows<3>() = Mat3::Identity() / count;
    plane.covariance.noalias() += j * p.covariance * j.transpose();
  }
  plane.radius = std::sqrt(radius2) + 0.1f * n->size;
  if (!plane.covariance.allFinite()) return;
  n->plane = plane;
  n->valid = true;
  UpdateBounds(n);
}
void VoxelMap::UpdateBounds(Node* n) {
  n->has_bounds = false;
  if (n->valid) {
    // 圆形平面支撑域在各坐标轴上的精确投影边界；不把无限平面当作体素内容。
    const Vec3 extent =
        (Vec3::Ones() - n->plane.normal.cwiseProduct(n->plane.normal))
            .cwiseMax(0)
            .cwiseSqrt() *
        n->plane.radius;
    n->bound_min = n->plane.center - extent;
    n->bound_max = n->plane.center + extent;
    n->has_bounds = true;
  }
  for (const auto& child : n->children) {
    if (!child || !child->has_bounds) continue;
    if (!n->has_bounds) {
      n->bound_min = child->bound_min;
      n->bound_max = child->bound_max;
      n->has_bounds = true;
    } else {
      n->bound_min = n->bound_min.cwiseMin(child->bound_min);
      n->bound_max = n->bound_max.cwiseMax(child->bound_max);
    }
  }
}
void VoxelMap::InsertNode(Node* n, const MapPoint& p) {
  if (n->split) {
    const int i = (p.position.x() >= n->center.x() ? 1 : 0) |
                  (p.position.y() >= n->center.y() ? 2 : 0) |
                  (p.position.z() >= n->center.z() ? 4 : 0);
    if (!n->children[i]) {
      const Vec3 offset((i & 1) ? 1 : -1, (i & 2) ? 1 : -1, (i & 4) ? 1 : -1);
      n->children[i] = std::make_unique<Node>(
          n->center + offset * (n->size * 0.25f), n->size * 0.5f, n->layer + 1);
    }
    InsertNode(n->children[i].get(), p);
    UpdateBounds(n);
    return;
  }
  ++n->seen;
  const size_t cap = static_cast<size_t>(
      std::min(config_.max_points_per_node, config_.max_cov_points_per_node));
  if (n->samples.size() < cap)
    n->samples.push_back(p);
  else
    n->samples[(n->seen - 1) % cap] = p;  // 有界滚动样本，不无限保存历史点。
  const size_t threshold =
      static_cast<size_t>(config_.min_points_per_layer[n->layer]);
  if (n->samples.size() < threshold) return;
  if (n->seen != threshold && n->seen % 5 != 0) return;
  FitPlane(n);
  if (!n->valid && n->layer < config_.max_layer) {
    n->split = true;
    auto samples = std::move(n->samples);
    for (const auto& sample : samples) InsertNode(n, sample);
  }
}
void VoxelMap::Insert(const std::vector<MapPoint>& points) {
  for (const auto& p : points) {
    if (!p.position.allFinite() || !p.covariance.allFinite()) continue;
    if ((p.position - crop_center_).norm() > config_.local_map_radius) continue;
    const auto key = Index(p.position);
    auto it = roots_.find(key);
    if (it == roots_.end()) {
      if (roots_.size() >= static_cast<size_t>(config_.max_root_voxels))
        continue;
      const float s = config_.root_voxel_size;
      const Vec3 center((static_cast<float>(key.x) + 0.5f) * s,
                        (static_cast<float>(key.y) + 0.5f) * s,
                        (static_cast<float>(key.z) + 0.5f) * s);
      it = roots_.emplace(key, std::make_unique<Node>(center, s, 0)).first;
    }
    InsertNode(it->second.get(), p);
  }
}
void VoxelMap::QueryNode(const Node& n, const Vec3& p, float* best,
                         PlaneMatch* match, bool* found) const {
  if (!n.has_bounds) return;
  // 到子树平面支撑域包围盒的距离是点到候选平面的下界，可安全剪枝。
  // 边界来自实际平面圆盘的并集，不使用可能截断圆盘的原始体素盒。
  const Vec3 outside = (n.bound_min - p).cwiseMax(p - n.bound_max).cwiseMax(0);
  if (outside.squaredNorm() > (*best) * (*best)) return;
  if (n.valid) {
    const Vec3 d = p - n.plane.center;
    const float r = n.plane.normal.dot(d);
    if ((d - n.plane.normal * r).norm() <= n.plane.radius &&
        std::abs(r) < *best) {
      *best = std::abs(r);
      *match = n.plane;
      *found = true;
    }
  }
  if (n.split)
    for (const auto& child : n.children)
      if (child) QueryNode(*child, p, best, match, found);
}
bool VoxelMap::QueryPlane(const Vec3& p, PlaneMatch* match,
                          float max_distance) const {
  if (!p.allFinite() || std::isnan(max_distance) || max_distance <= 0)
    return false;
  const auto k = Index(p);
  const int r = config_.neighbor_voxel_radius;
  float best = max_distance;
  bool found = false;
  for (int x = -r; x <= r; ++x)
    for (int y = -r; y <= r; ++y)
      for (int z = -r; z <= r; ++z) {
        const auto it = roots_.find({k.x + x, k.y + y, k.z + z});
        if (it != roots_.end()) QueryNode(*it->second, p, &best, match, &found);
      }
  return found;
}
void VoxelMap::Crop(const Vec3& p) {
  if (!p.allFinite()) throw std::invalid_argument("invalid crop center");
  crop_center_ = p;
  const float half = config_.root_voxel_size * 0.5f;
  for (auto it = roots_.begin(); it != roots_.end();) {
    const Vec3 outside =
        ((it->second->center - p).cwiseAbs().array() - half).max(0).matrix();
    if (outside.norm() > config_.local_map_radius)
      it = roots_.erase(it);
    else
      ++it;
  }
}
void VoxelMap::Collect(const Node& n, size_t limit,
                       std::vector<PlaneMatch>* out) const {
  if (out->size() >= limit) return;
  if (n.valid) out->push_back(n.plane);
  for (const auto& child : n.children)
    if (child) Collect(*child, limit, out);
}
std::vector<PlaneMatch> VoxelMap::Planes(size_t limit) const {
  std::vector<PlaneMatch> out;
  for (const auto& root : roots_) {
    if (out.size() >= limit) break;
    Collect(*root.second, limit, &out);
  }
  return out;
}
}  // namespace lio
