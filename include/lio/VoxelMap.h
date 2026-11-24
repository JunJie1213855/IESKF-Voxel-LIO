#ifndef LIO_VOXEL_MAP_H_
#define LIO_VOXEL_MAP_H_

#include <array>
#include <limits>
#include <memory>
#include <unordered_map>

#include "lio/Preprocess.h"

namespace lio {
struct MapConfig {
  float root_voxel_size = 3;
  int max_layer = 3;
  std::vector<int> min_points_per_layer{10, 10, 8, 5};
  float plane_max_min_eigenvalue = 0.01f;
  float plane_max_eigenvalue_ratio = 0.1f;
  int max_points_per_node = 100;
  int max_cov_points_per_node = 100;
  int neighbor_voxel_radius = 1;
  float local_map_radius = 100;
  int max_root_voxels = 100000;
  void Validate() const;
};
struct MapPoint {
  Vec3 position = Vec3::Zero();
  Mat3 covariance = Mat3::Identity() * 1e-4f;
};
struct PlaneMatch {
  Vec3 center = Vec3::Zero();
  Vec3 normal = Vec3::UnitZ();
  Eigen::Matrix<float, 6, 6> covariance = Eigen::Matrix<float, 6, 6>::Zero();
  float radius = 0;
  int layer = 0;
};
Mat3 LidarCovariance(const Vec3& point, float range_std, float angle_std_deg);
class VoxelMap {
 public:
  explicit VoxelMap(const MapConfig& config = {});
  ~VoxelMap();
  VoxelMap(const VoxelMap&) = delete;
  VoxelMap& operator=(const VoxelMap&) = delete;
  void Build(const std::vector<MapPoint>& points);
  void Insert(const std::vector<MapPoint>& points);
  // 查询返回值快照，不将内部节点指针交给滤波器。
  bool QueryPlane(
      const Vec3& world_point, PlaneMatch* match,
      float max_distance = std::numeric_limits<float>::infinity()) const;
  void Crop(const Vec3& lidar_position);
  void Clear();
  size_t Size() const { return roots_.size(); }
  std::vector<PlaneMatch> Planes(size_t limit = 5000) const;

 private:
  struct Key {
    int64_t x, y, z;
    bool operator==(const Key& b) const {
      return x == b.x && y == b.y && z == b.z;
    }
  };
  struct Hash {
    size_t operator()(const Key& k) const;
  };
  struct Node;
  Key Index(const Vec3& p) const;
  void InsertNode(Node* node, const MapPoint& point);
  void FitPlane(Node* node);
  void UpdateBounds(Node* node);
  void QueryNode(const Node& node, const Vec3& point, float* distance,
                 PlaneMatch* match, bool* found) const;
  void Collect(const Node& node, size_t limit,
               std::vector<PlaneMatch>* out) const;
  MapConfig config_;
  std::unordered_map<Key, std::unique_ptr<Node>, Hash> roots_;
  Vec3 crop_center_ = Vec3::Zero();
};
}  // namespace lio
#endif
