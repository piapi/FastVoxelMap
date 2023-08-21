
#pragma once

#include <vector>
#include <vector>
#include <unordered_map>

#include <Eigen/Core>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
namespace VoxelMap {

// a point to plane matching structure
struct ptpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d point;
  Eigen::Vector3d point_world;
  // // 平面法向量
  // Eigen::Vector3d normal;
  // // 平面中心
  // Eigen::Vector3d center;
  // Eigen::Matrix<double, 6, 6> plane_cov;
  // 平面参数D
  // double d;
  // 点到平面的距离
  float dist;
  // 平面实例的地址
  uint64_t plane_ptr;
};

// 3D point with covariance
struct PointWithCov {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d point;
  Eigen::Vector3d point_world;
  Eigen::Matrix3d cov;
};

struct Plane {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // ~Plane() { printf("~Plane [%d] - [%d]\n", id, pts_num); }

  bool is_init = false;
  int id{-1};
  bool is_plane = false;
  int pts_num = 0;
  //平面方程 Ax + By + Cz + D = 0
  float d = 0;
  Eigen::Vector3d normal;
  // 平面方向系数norm，不需要了，特征向量norm恒为1
  // double coff = 0;
  // 平面的重心
  Eigen::Vector3d center;
  // 评估的平面半径 todo 不准确
  float radius = 0;
  // 构成平面的点到平面距离的平均误差
  float dist_err{-1.};
  // Eigen::Vector3d y_normal;
  // Eigen::Vector3d x_normal;
  // 组成平面的点的协方差
  Eigen::Matrix3d covariance;
  // 平面协方差
  Eigen::Matrix<double, 6, 6> plane_cov;

  // float width = 0;
  // float height = 0;

  // 用于发送可视化数据
  bool is_update = false;
  // int last_update_points_size = 0;
  // 用于调试的点
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> debug_points;
  // 显示的半径
  float show_radius{0.};
};

class VoxelLoc {
 public:
  VoxelLoc(const Eigen::Vector3d &pos, const double voxel_size) {
    Eigen::Vector3i index = (pos / voxel_size).array().floor().template cast<int>();
    x = index.x();
    y = index.y();
    z = index.z();
  }

  bool operator==(const VoxelLoc &other) const { return (x == other.x && y == other.y && z == other.z); }

  int x, y, z;
};

class OctoTree {
 public:
  OctoTree(int max_layer, int layer, int plane_fixed_pts_num, int cov_fixed_pts_num, float plane_min_eigen_val_thd);

  // check is plane, calc plane parameters including plane covariance
  void InitPlane(const std::vector<PointWithCov> &points, Plane *plane) const;

  // only update plane normal, center and radius with new points
  // todo 感觉这种更新没有什么意义
  void UpdatePlane(const std::vector<PointWithCov> &points, Plane *plane) const;

  void InitOctoTree();

  void CutOctoTree();

  void UpdateOctoTree(const PointWithCov &pv);

  static void SetLayersParam(const std::vector<int> &layers_param) { each_layer_min_pts_num = layers_param; }

 public:
  // 是否初始化过
  bool init_octo;
  // 最大层数
  int max_layer;
  // 当前层数
  int layer;
  // 状态 0 is end of tree, 1 is not
  int state;
  // 中心点位置坐标
  double voxel_center[3]{};
  // 四分之一边长
  float quater_length{};
  // 平面最小特征特征值门限
  float plane_min_eigen_val_thd;
  // 评估平面所需的最少点数
  int estimate_plane_min_num;
  // 重新估计平面参数的新增点数, 目前该值为固定的5
  int plane_update_pts_num;
  // 当前组成平面的所有点数
  // int all_pts_num;
  // 平面确定后新增的点数
  int new_pts_num;
  // 达到该点数就不再更新平面参数了
  int plane_fixed_pts_num;
  // 达到该点数就不再更新协方差了
  int cov_fixed_pts_num;
  // 需要更新平面协方差，相当与重新计算一遍平面参数
  bool update_cov_enable;
  // 更新平面除平面协方差外的参数
  bool update_enable;
  // 对应平面 todo 当前体素中有子体素这个平面应该删掉
  Plane *plane_ptr{nullptr};
  // 子体素（八叉树）
  std::array<OctoTree *, 8> leaves{};
  // 当前体素内的点
  std::vector<PointWithCov> temp_points;
  // 新加入的点
  std::vector<PointWithCov> new_points;

  static int plane_id;
  // 不同的层评估平面所需的最少点数不同，这个就是列表
  static std::vector<int> each_layer_min_pts_num;
};

void FreeVoxelMap(std::unordered_map<VoxelLoc, OctoTree *> &feat_map);

void BuildSingleResidual(const PointWithCov &pv, const OctoTree *current_octo, int current_layer, int max_layer,
                         double sigma_num, bool &is_sucess, double &prob, ptpl &single_ptpl);

void GetUpdatePlane(const OctoTree *current_octo, int pub_max_voxel_layer, std::vector<Plane *> &plane_list);

void BuildResidualListOMP(const std::unordered_map<VoxelLoc, OctoTree *> &voxel_map, double voxel_size,
                          double sigma_num, int max_layer, const std::vector<PointWithCov> &pv_list,
                          std::vector<ptpl> &ptpl_list, std::vector<Eigen::Vector3d> &non_match);

void BuildResidualListNormal(const std::unordered_map<VoxelLoc, OctoTree *> &voxel_map, double voxel_size,
                             double sigma_num, int max_layer, const std::vector<PointWithCov> &pv_list,
                             std::vector<ptpl> &ptpl_list, std::vector<Eigen::Vector3d> &non_match);

void BuildVoxelMap(const std::vector<PointWithCov> &input_points, float voxel_size, int max_layer,
                   int plane_fixed_pts_num, int cov_fixed_pts_num, float planer_threshold,
                   std::unordered_map<VoxelLoc, OctoTree *> &feat_map);

void UpdateVoxelMap(const std::vector<PointWithCov> &input_points, float voxel_size, int max_layer,
                    int plane_fixed_pts_num, int cov_fixed_pts_num, float planer_threshold,
                    std::unordered_map<VoxelLoc, OctoTree *> &feat_map);

void CalcBodyCov(const Eigen::Vector3d &pb, double range_inc, double degree_inc, Eigen::Matrix3d &cov);

#ifdef USE_PATC_API
void MapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b);
void PubVoxelMap(const std::string &name, const std::unordered_map<VoxelLoc, OctoTree *> &voxel_map);
void PubMatchResult(const std::string &name, const std::vector<VoxelMap::ptpl> &ptpl_list);
#endif


}  // namespace voxel_map

// #define HASH_P 116101
// #define MAX_N 10000000000

namespace std {
template <>
struct hash<VoxelMap::VoxelLoc> {
  size_t operator()(const VoxelMap::VoxelLoc &s) const {
    // using std::hash;
    // using std::size_t;
    // return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
    return size_t(((s.x) * 73856093) ^ ((s.y) * 471943) ^ ((s.z) * 83492791)) % 10000000;
  }
};
}  // namespace std