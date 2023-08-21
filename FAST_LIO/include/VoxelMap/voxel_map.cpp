
#include "voxel_map.h"

#include <mutex>

#include <Eigen/Dense>
#include <pcl/pcl_macros.h>
#include <pcl/point_types.h>
#include <iostream>
#include <string>
#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/common.h>
#include <vtkAutoInit.h>
#include <Eigen/Core>
#include <pcl/common/transforms.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#ifdef MP_EN
#include <omp.h>
#endif

#ifdef USE_PATC_API
// 性能分析工具客户端
#include "patc/patc_api.h"
#include "mvt_msg_plot.pb.h"
// #else
// #include "not_use_patc.h"
#endif

// 点到平面的有效距离值
#define PT_TO_PLANE_THD (0.03)

namespace VoxelMap {

int OctoTree::plane_id{0};

std::vector<int> OctoTree::each_layer_min_pts_num;

OctoTree::OctoTree(int max_layer, int layer, int plane_fixed_pts_num, int cov_fixed_pts_num,
                   float plane_min_eigen_val_thd)
    : max_layer(max_layer),
      layer(layer),
      plane_fixed_pts_num(plane_fixed_pts_num),
      cov_fixed_pts_num(cov_fixed_pts_num),
      plane_min_eigen_val_thd(plane_min_eigen_val_thd) {
  temp_points.clear();
  state = 0;
  new_pts_num = 0;
  // when new points num > 5, do a update
  plane_update_pts_num = 5;
  init_octo = false;
  update_enable = true;
  update_cov_enable = true;
  estimate_plane_min_num = each_layer_min_pts_num[layer];
  for (auto &leave : leaves) {
    leave = nullptr;
  }
  plane_ptr = new Plane;
}

bool EstimatePlane(const std::vector<PointWithCov> &points, Plane *plane, float &err) {
  err = 0.;
  for (auto pv : points) {
    // clang-format off
    // 计算点到面的距离
    auto dist = (float)std::fabs(pv.point_world.x() * plane->normal.x() + pv.point_world.y() * plane->normal.y() +
                            pv.point_world.z() * plane->normal.z() + plane->d);
    err += dist;
    // clang-format on
    if (dist > PT_TO_PLANE_THD) {
      return false;
    }
  }
  err /= (float)points.size();
  return true;
}

void OctoTree::InitPlane(const std::vector<PointWithCov> &points, Plane *plane) const {
  // 这里只是防止逻辑bug导致异常进入
  if (points.size() <= plane_update_pts_num) {
    printf("InitPlane points num wrong! %zu\n", points.size());
    exit(-1);
  }
  plane->plane_cov = Eigen::Matrix<double, 6, 6>::Zero();
  plane->covariance = Eigen::Matrix3d::Zero();
  plane->center = Eigen::Vector3d::Zero();
  plane->normal = Eigen::Vector3d::Zero();
  plane->pts_num = (int)points.size();
  plane->radius = 0;
  for (auto pv : points) {
    plane->covariance += pv.point_world * pv.point_world.transpose();
    plane->center += pv.point_world;
  }

  plane->center = plane->center / plane->pts_num;
  plane->covariance = plane->covariance / plane->pts_num - plane->center * plane->center.transpose();

  // 计算特征值与特征向量
  // Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(plane->covariance);

  auto evecs = es.eigenvectors();
  auto evals = es.eigenvalues();
  // Eigen::Matrix3d cov = evecs * evals.asDiagonal() * evecs.transpose();

  int evals_min, evals_max;
  evals.rowwise().sum().minCoeff(&evals_min);
  evals.rowwise().sum().maxCoeff(&evals_max);
  // int evals_mid = 3 - evals_min - evals_max;
  // Eigen::Vector3d evecMin = evecs.real().col(evals_min);
  // Eigen::Vector3d evecMid = evecs.real().col(evals_mid);
  // Eigen::Vector3d evecMax = evecs.real().col(evals_max);

  plane->normal << evecs(0, evals_min), evecs(1, evals_min), evecs(2, evals_min);
  // plane->y_normal << evecs.real()(0, evals_mid), evecs.real()(1, evals_mid), evecs.real()(2, evals_mid);
  // plane->x_normal << evecs.real()(0, evals_max), evecs.real()(1, evals_max), evecs.real()(2, evals_max);
  // plane->min_eigen_value = evalsReal(evals_min);
  // plane->mid_eigen_value = evalsReal(evals_mid);
  // plane->max_eigen_value = evalsReal(evals_max);
  plane->radius = (float)std::sqrt(evals(evals_max));
  plane->d = (float)(-(plane->normal(0) * plane->center(0) + plane->normal(1) * plane->center(1) +
                       plane->normal(2) * plane->center(2)));
  // plane->width = plane->radius * 2;
  // plane->height = (float)std::sqrt(evalsReal(evals_mid));
  float dist_err{0.};
  if (evals(evals_min) < plane_min_eigen_val_thd && EstimatePlane(points, plane, dist_err)) {
    plane->dist_err = dist_err;
    // plane covariance calculation
    Eigen::Matrix3d J_Q;
    J_Q << 1.0 / plane->pts_num, 0, 0, 0, 1.0 / plane->pts_num, 0, 0, 0, 1.0 / plane->pts_num;
    std::vector<int> index(points.size());
    std::vector<Eigen::Matrix<double, 6, 6>> temp_matrix(points.size());
    for (const auto &point : points) {
      Eigen::Matrix<double, 6, 3> J;
      Eigen::Matrix3d F;
      for (int m = 0; m < 3; m++) {
        if (m != (int)evals_min) {
          Eigen::Matrix<double, 1, 3> F_m = (point.point_world - plane->center).transpose() /
                                            ((plane->pts_num) * (evals[evals_min] - evals[m])) *
                                            (evecs.real().col(m) * evecs.real().col(evals_min).transpose() +
                                             evecs.real().col(evals_min) * evecs.real().col(m).transpose());
          F.row(m) = F_m;
        } else {
          Eigen::Matrix<double, 1, 3> F_m;
          F_m << 0, 0, 0;
          F.row(m) = F_m;
        }
      }
      J.block<3, 3>(0, 0) = evecs.real() * F;
      J.block<3, 3>(3, 0) = J_Q;
      plane->plane_cov += J * point.cov * J.transpose();
    }

    // if (plane->last_update_points_size == 0) {
    //   plane->last_update_points_size = plane->pts_num;
    //   plane->is_update = true;
    // } else if (plane->pts_num - plane->last_update_points_size > 20) {
    //   plane->last_update_points_size = plane->pts_num;
    //   plane->is_update = true;
    // }

    if (!plane->is_init) {
      plane->id = plane_id;
      plane_id++;
      plane->is_init = true;
    }
    plane->is_plane = true;
    plane->is_update = true;
  } else {
    // if (plane->last_update_points_size == 0) {
    //   plane->last_update_points_size = plane->pts_num;
    //   plane->is_update = true;
    // } else if (plane->pts_num - plane->last_update_points_size > 100) {
    //   plane->last_update_points_size = plane->pts_num;
    //   plane->is_update = true;
    // }
    // 设置一个很大的值
    plane->dist_err = 1.;

    if (!plane->is_init) {
      plane->id = plane_id;
      plane_id++;
      plane->is_init = true;
    }
    plane->is_plane = false;
    if (plane->pts_num) {
      plane->is_update = true;
    }
  }
}

void OctoTree::UpdatePlane(const std::vector<PointWithCov> &points, Plane *plane) const {
  std::vector<PointWithCov> valid_pts;
  float dist_err = plane->dist_err * (float)plane->pts_num;
  // 到平面距离大的点直接丢掉
  for (const auto &pv : points) {
    auto dist = (float)std::fabs(pv.point_world.x() * plane->normal.x() + pv.point_world.y() * plane->normal.y() +
                                 pv.point_world.z() * plane->normal.z() + plane->d);
    if (dist < PT_TO_PLANE_THD) {
      valid_pts.push_back(pv);
      dist_err += dist;
    }
  }

  Eigen::Matrix3d sum_ppt = (plane->covariance + plane->center * plane->center.transpose()) * plane->pts_num;
  Eigen::Vector3d sum_p = plane->center * plane->pts_num;
  for (const auto &point : valid_pts) {
    Eigen::Vector3d pv = point.point_world;
    sum_ppt += pv * pv.transpose();
    sum_p += pv;
  }
  plane->pts_num += (int)valid_pts.size();
  plane->center = sum_p / plane->pts_num;
  plane->covariance = sum_ppt / plane->pts_num - plane->center * plane->center.transpose();
  // Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance);
  // 计算特征值与特征向量
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(plane->covariance);
  auto evecs = es.eigenvectors();
  auto evals = es.eigenvalues();

  int evals_min, evals_max;
  evals.rowwise().sum().minCoeff(&evals_min);
  evals.rowwise().sum().maxCoeff(&evals_max);
  // int evalsMid = 3 - evalsMin - evalsMax;
  // Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
  // Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
  // Eigen::Vector3d evecMax = evecs.real().col(evalsMax);

  plane->normal << evecs.real()(0, evals_min), evecs.real()(1, evals_min), evecs.real()(2, evals_min);
  // plane->y_normal << evecs.real()(0, evals_mid), evecs.real()(1, evals_mid), evecs.real()(2, evals_mid);
  // plane->x_normal << evecs.real()(0, evals_max), evecs.real()(1, evals_max), evecs.real()(2, evals_max);
  // plane->min_eigen_value = evalsReal(evals_min);
  // plane->mid_eigen_value = evalsReal(evals_mid);
  // plane->max_eigen_value = evalsReal(evals_max);
  plane->radius = (float)std::sqrt(evals(evals_max));
  plane->d = (float)(-(plane->normal(0) * plane->center(0) + plane->normal(1) * plane->center(1) +
                       plane->normal(2) * plane->center(2)));

  if (evals(evals_min) < plane_min_eigen_val_thd) {
    plane->dist_err = dist_err / (float)plane->pts_num;
    plane->is_plane = true;
    plane->is_update = true;
  } else {
    plane->dist_err = 1.;
    plane->is_plane = false;
    plane->is_update = true;
  }
}

void OctoTree::InitOctoTree() {
  if (temp_points.size() > estimate_plane_min_num) {
    InitPlane(temp_points, plane_ptr);
    if (plane_ptr->is_plane) {
      state = 0;
      if (temp_points.size() > cov_fixed_pts_num) {
        update_cov_enable = false;
      }
      if (temp_points.size() > plane_fixed_pts_num) {
        update_enable = false;
      }
    } else {
      state = 1;
      CutOctoTree();
    }
    init_octo = true;
  }
}

void OctoTree::CutOctoTree() {
  if (layer >= max_layer) {
    state = 0;
    return;
  }
  for (auto &temp_point : temp_points) {
    int xyz[3] = {0, 0, 0};
    if (temp_point.point_world[0] > voxel_center[0]) {
      xyz[0] = 1;
    }
    if (temp_point.point_world[1] > voxel_center[1]) {
      xyz[1] = 1;
    }
    if (temp_point.point_world[2] > voxel_center[2]) {
      xyz[2] = 1;
    }
    int leaf_num = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    if (leaves[leaf_num] == nullptr) {
      leaves[leaf_num] =
          new OctoTree(max_layer, layer + 1, plane_fixed_pts_num, cov_fixed_pts_num, plane_min_eigen_val_thd);
      leaves[leaf_num]->voxel_center[0] = voxel_center[0] + float(2 * xyz[0] - 1) * quater_length;
      leaves[leaf_num]->voxel_center[1] = voxel_center[1] + float(2 * xyz[1] - 1) * quater_length;
      leaves[leaf_num]->voxel_center[2] = voxel_center[2] + float(2 * xyz[2] - 1) * quater_length;
      leaves[leaf_num]->quater_length = quater_length / 2;
    }
    leaves[leaf_num]->temp_points.push_back(temp_point);
  }
  std::vector<PointWithCov>().swap(temp_points);
  for (auto &leave : leaves) {
    if (leave != nullptr) {
      if (leave->temp_points.size() > leave->estimate_plane_min_num) {
        InitPlane(leave->temp_points, leave->plane_ptr);
        if (leave->plane_ptr->is_plane) {
          leave->state = 0;
          if (leave->temp_points.size() > cov_fixed_pts_num) {
            leave->update_cov_enable = false;
          }
          if (leave->temp_points.size() > plane_fixed_pts_num) {
            leave->update_enable = false;
          }
        } else {
          leave->state = 1;
          leave->CutOctoTree();
        }
        leave->init_octo = true;
      }
    }
  }
}

void OctoTree::UpdateOctoTree(const PointWithCov &pv) {
  if (!init_octo) {
    // new_pts_num++;
    // all_pts_num++;
    temp_points.push_back(pv);
    if (temp_points.size() > estimate_plane_min_num) {
      InitOctoTree();
    }
    return;
  } else {
    // 是平面
    if (plane_ptr->is_plane) {
      if (update_enable) {
        new_pts_num++;
        // all_pts_num++;
        if (update_cov_enable) {
          temp_points.push_back(pv);
        } else {
          new_points.push_back(pv);
        }
        if (new_pts_num > plane_update_pts_num) {
          if (update_cov_enable) {
            InitPlane(temp_points, plane_ptr);
            if (plane_ptr->is_plane) {
              state = 0;
              if (plane_ptr->pts_num > cov_fixed_pts_num) {
                update_cov_enable = false;
              }
              if (plane_ptr->pts_num > plane_fixed_pts_num) {
                update_enable = false;
              }
            } else {
              // 增加点后，不是所有的点都在平面上，需要进一步细分
              state = 1;
              CutOctoTree();
            }
            new_pts_num = 0;
            if (!new_points.empty()) {
              std::vector<PointWithCov>().swap(new_points);
            }
            return;
          }
          if (update_enable) {
            UpdatePlane(new_points, plane_ptr);
            // 需要把点加入
            for (const auto &p : new_points) {
              temp_points.push_back(p);
            }
            std::vector<PointWithCov>().swap(new_points);
            new_pts_num = 0;
            if (plane_ptr->is_plane) {
              state = 0;
              if (plane_ptr->pts_num > cov_fixed_pts_num) {
                update_cov_enable = false;
              }
              if (plane_ptr->pts_num > plane_fixed_pts_num) {
                update_enable = false;
              }
            } else {
              // 增加点后，不是所有的点都在平面上，需要进一步细分
              state = 1;
              CutOctoTree();
            }
            return;
          }
        }
      } else {
        return;
      }
    } else {
      // 不是平面
      // 且不是最后一层
      if (layer < max_layer) {
        // if (!temp_points.empty()) {
        //   std::vector<PointWithCov>().swap(temp_points);
        // }
        // if (!new_points.empty()) {
        //   std::vector<PointWithCov>().swap(new_points);
        // }
        int xyz[3] = {0, 0, 0};
        if (pv.point_world[0] > voxel_center[0]) {
          xyz[0] = 1;
        }
        if (pv.point_world[1] > voxel_center[1]) {
          xyz[1] = 1;
        }
        if (pv.point_world[2] > voxel_center[2]) {
          xyz[2] = 1;
        }
        int leaf_num = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
        if (leaves[leaf_num] != nullptr) {
          leaves[leaf_num]->UpdateOctoTree(pv);
        } else {
          leaves[leaf_num] =
              new OctoTree(max_layer, layer + 1, plane_fixed_pts_num, cov_fixed_pts_num, plane_min_eigen_val_thd);
          leaves[leaf_num]->voxel_center[0] = voxel_center[0] + float(2 * xyz[0] - 1) * quater_length;
          leaves[leaf_num]->voxel_center[1] = voxel_center[1] + float(2 * xyz[1] - 1) * quater_length;
          leaves[leaf_num]->voxel_center[2] = voxel_center[2] + float(2 * xyz[2] - 1) * quater_length;
          leaves[leaf_num]->quater_length = quater_length / 2;
          leaves[leaf_num]->UpdateOctoTree(pv);
        }
      } else {
        // 不是平面 且是最后一层
        // 已经不能形成平面了，无需计算
        // 用于可视化
        temp_points.push_back(pv);
        plane_ptr->is_update = true;
        return;
      }
    }
  }
}

void BuildSingleResidual(const PointWithCov &pv, const OctoTree *current_octo, const int current_layer,
                         const int max_layer, const double sigma_num, bool &is_sucess, double &prob,
                         ptpl &single_ptpl) {
  const double radius_k = 1;
  auto &p_w = pv.point_world;
  if (current_octo->plane_ptr->is_plane) {
    Plane &plane = *current_octo->plane_ptr;
    // Eigen::Vector3d p_world_to_center = p_w - plane.center;
    // double proj_x = p_world_to_center.dot(plane.x_normal);
    // double proj_y = p_world_to_center.dot(plane.y_normal);
    auto dis_to_plane =
        (float)(plane.normal(0) * p_w(0) + plane.normal(1) * p_w(1) + plane.normal(2) * p_w(2) + plane.d);
    auto dis_to_center = (float)((plane.center(0) - p_w(0)) * (plane.center(0) - p_w(0)) +
                                 (plane.center(1) - p_w(1)) * (plane.center(1) - p_w(1)) +
                                 (plane.center(2) - p_w(2)) * (plane.center(2) - p_w(2)));
    // 计算平面投影到平面中心的距离
    auto range_dis = (float)std::sqrt(dis_to_center - dis_to_plane * dis_to_plane);

    if (range_dis <= radius_k * plane.radius) {
      Eigen::Matrix<double, 1, 6> J_nq;
      J_nq.block<1, 3>(0, 0) = p_w - plane.center;
      J_nq.block<1, 3>(0, 3) = -plane.normal;
      double sigma_l = J_nq * plane.plane_cov * J_nq.transpose();
      sigma_l += plane.normal.transpose() * pv.cov * plane.normal;
      // if (fabs(dis_to_plane) < sigma_num * sqrt(sigma_l)) {
      if (fabs(dis_to_plane) < 0.05) {
        is_sucess = true;
        double this_prob = 1.0 / (sqrt(sigma_l)) * exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);
        if (this_prob > prob) {
          prob = this_prob;
          single_ptpl.point = pv.point;
          single_ptpl.point_world = pv.point_world;
          // single_ptpl.plane_cov = plane.plane_cov;
          // single_ptpl.normal = plane.normal;
          // single_ptpl.center = plane.center;
          // single_ptpl.d = plane.d;
          single_ptpl.dist = dis_to_plane;
          single_ptpl.plane_ptr = (uint64_t)current_octo->plane_ptr;
          current_octo->plane_ptr->show_radius = current_octo->quater_length * 2;
        }
        return;
      } else {
        return;
      }
    } else {
      return;
    }
  } else {
    if (current_layer < max_layer) {
      for (auto leave : current_octo->leaves) {
        if (leave != nullptr) {
          BuildSingleResidual(pv, leave, current_layer + 1, max_layer, sigma_num, is_sucess, prob, single_ptpl);
        }
      }
      return;
    } else {
      // is_sucess = false;
      return;
    }
  }
}

void GetUpdatePlane(const OctoTree *current_octo, const int pub_max_voxel_layer, std::vector<Plane *> &plane_list) {
  if (current_octo->layer > pub_max_voxel_layer) {
    return;
  }
  // 更新后不是平面的也需要发送
  if (current_octo->plane_ptr->is_update) {
    plane_list.push_back(current_octo->plane_ptr);
    // 填充显示的点
    for (const auto &p : current_octo->temp_points) {
      current_octo->plane_ptr->debug_points.emplace_back(p.point_world.cast<float>());
    }
    current_octo->plane_ptr->show_radius = current_octo->quater_length * 2;
  }
  if (current_octo->layer < current_octo->max_layer) {
    if (!current_octo->plane_ptr->is_plane) {
      for (auto leave : current_octo->leaves) {
        if (leave != nullptr) {
          GetUpdatePlane(leave, pub_max_voxel_layer, plane_list);
        }
      }
    }
  }
}

void BuildResidualListOMP(const std::unordered_map<VoxelLoc, OctoTree *> &voxel_map, const double voxel_size,
                          const double sigma_num, const int max_layer, const std::vector<PointWithCov> &pv_list,
                          std::vector<ptpl> &ptpl_list, std::vector<Eigen::Vector3d> &non_match) {
  std::mutex mylock;
  ptpl_list.clear();
  std::vector<ptpl> all_ptpl_list(pv_list.size());
  std::vector<bool> useful_ptpl(pv_list.size());
  std::vector<size_t> index(pv_list.size());
  for (size_t i = 0; i < index.size(); ++i) {
    index[i] = i;
    useful_ptpl[i] = false;
  }

#ifdef MP_EN
  auto max_thread_num = omp_get_max_threads();
  omp_set_num_threads(max_thread_num);
#ifdef _MSC_VER
#pragma omp parallel for
#else
#pragma omp parallel for default(none) \
    shared(index, pv_list, voxel_size, voxel_map, max_layer, sigma_num, mylock, useful_ptpl, all_ptpl_list)
#endif
#endif
  for (int i = 0; i < index.size(); i++) {
    PointWithCov pv = pv_list[i];
    // if (fabs(pv.point_world.x() + 4.318) < 0.02 && fabs(pv.point_world.y() + 2.247) < 0.02 &&
    //     fabs(pv.point_world.z() - 5.196) < 0.02) {
    //   printf("\n");
    // }
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = (float)(pv.point_world[j] / voxel_size);
      // if (loc_xyz[j] < 0) {
      //   loc_xyz[j] -= 1.0;
      // }
    }
    // VoxelLoc position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    VoxelLoc position(pv.point_world, voxel_size);
    auto iter = voxel_map.find(position);
    if (iter != voxel_map.end()) {
      OctoTree *current_octo = iter->second;
      ptpl single_ptpl;
      bool is_sucess = false;
      double prob = 0;
      BuildSingleResidual(pv, current_octo, 0, max_layer, sigma_num, is_sucess, prob, single_ptpl);
      if (!is_sucess) {
        VoxelLoc near_position = position;
        if (loc_xyz[0] > (current_octo->voxel_center[0] + current_octo->quater_length)) {
          near_position.x = near_position.x + 1;
        } else if (loc_xyz[0] < (current_octo->voxel_center[0] - current_octo->quater_length)) {
          near_position.x = near_position.x - 1;
        }
        if (loc_xyz[1] > (current_octo->voxel_center[1] + current_octo->quater_length)) {
          near_position.y = near_position.y + 1;
        } else if (loc_xyz[1] < (current_octo->voxel_center[1] - current_octo->quater_length)) {
          near_position.y = near_position.y - 1;
        }
        if (loc_xyz[2] > (current_octo->voxel_center[2] + current_octo->quater_length)) {
          near_position.z = near_position.z + 1;
        } else if (loc_xyz[2] < (current_octo->voxel_center[2] - current_octo->quater_length)) {
          near_position.z = near_position.z - 1;
        }
        auto iter_near = voxel_map.find(near_position);
        if (iter_near != voxel_map.end()) {
          BuildSingleResidual(pv, iter_near->second, 0, max_layer, sigma_num, is_sucess, prob, single_ptpl);
        }
      }
      if (is_sucess) {
        mylock.lock();
        useful_ptpl[i] = true;
        all_ptpl_list[i] = single_ptpl;
        mylock.unlock();
      } else {
        mylock.lock();
        useful_ptpl[i] = false;
        mylock.unlock();
      }
    }
  }
  for (size_t i = 0; i < useful_ptpl.size(); i++) {
    if (useful_ptpl[i]) {
      ptpl_list.push_back(all_ptpl_list[i]);
    }
  }
}

void BuildResidualListNormal(const std::unordered_map<VoxelLoc, OctoTree *> &voxel_map, const double voxel_size,
                             const double sigma_num, const int max_layer, const std::vector<PointWithCov> &pv_list,
                             std::vector<ptpl> &ptpl_list, std::vector<Eigen::Vector3d> &non_match) {
  ptpl_list.clear();
  std::vector<size_t> index(pv_list.size());
  for (auto &pv : pv_list) {
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = (float)(pv.point_world[j] / voxel_size);
      // if (loc_xyz[j] < 0) {
      //   loc_xyz[j] -= 1.0;
      // }
    }
    // VoxelLoc position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    VoxelLoc position(pv.point_world, voxel_size);
    auto iter = voxel_map.find(position);
    if (iter != voxel_map.end()) {
      OctoTree *current_octo = iter->second;
      ptpl single_ptpl;
      bool is_sucess = false;
      double prob = 0;
      BuildSingleResidual(pv, current_octo, 0, max_layer, sigma_num, is_sucess, prob, single_ptpl);

      if (!is_sucess) {
        VoxelLoc near_position = position;
        if (loc_xyz[0] > (current_octo->voxel_center[0] + current_octo->quater_length)) {
          near_position.x = near_position.x + 1;
        } else if (loc_xyz[0] < (current_octo->voxel_center[0] - current_octo->quater_length)) {
          near_position.x = near_position.x - 1;
        }
        if (loc_xyz[1] > (current_octo->voxel_center[1] + current_octo->quater_length)) {
          near_position.y = near_position.y + 1;
        } else if (loc_xyz[1] < (current_octo->voxel_center[1] - current_octo->quater_length)) {
          near_position.y = near_position.y - 1;
        }
        if (loc_xyz[2] > (current_octo->voxel_center[2] + current_octo->quater_length)) {
          near_position.z = near_position.z + 1;
        } else if (loc_xyz[2] < (current_octo->voxel_center[2] - current_octo->quater_length)) {
          near_position.z = near_position.z - 1;
        }
        auto iter_near = voxel_map.find(near_position);
        if (iter_near != voxel_map.end()) {
          BuildSingleResidual(pv, iter_near->second, 0, max_layer, sigma_num, is_sucess, prob, single_ptpl);
        }
      }
      if (is_sucess) {
        ptpl_list.push_back(single_ptpl);
      } else {
        non_match.push_back(pv.point_world);
      }
    }
  }
}

void CalcBodyCov(const Eigen::Vector3d &pb, const double range_sigma, const double degree_sigma, Eigen::Matrix3d &cov) {
  double range = (float)sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
  double range_var = range_sigma * range_sigma;
  Eigen::Matrix2d direction_var;
  direction_var << pow(sin(DEG2RAD(degree_sigma)), 2), 0, 0, pow(sin(DEG2RAD(degree_sigma)), 2);
  Eigen::Vector3d direction(pb);
  direction.normalize();
  Eigen::Matrix3d direction_hat;
  direction_hat << 0, -direction(2), direction(1), direction(2), 0, -direction(0), -direction(1), direction(0), 0;
  Eigen::Vector3d base_vector1(1, 1, -(direction(0) + direction(1)) / direction(2));
  base_vector1.normalize();
  Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
  base_vector2.normalize();
  Eigen::Matrix<double, 3, 2> N;
  N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1), base_vector1(2), base_vector2(2);
  Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
  cov = direction * range_var * direction.transpose() + A * direction_var * A.transpose();
}

void BuildVoxelMap(const std::vector<PointWithCov> &input_points, const float voxel_size, const int max_layer,
                   const int plane_fixed_pts_num, const int cov_fixed_pts_num, const float planer_threshold,
                   std::unordered_map<VoxelLoc, OctoTree *> &feat_map) {
  for (const auto &pv : input_points) {
    // float loc_xyz[3];
    // for (int j = 0; j < 3; j++) {
    //   loc_xyz[j] = pv.point_world[j] / voxel_size;
    //   // if (loc_xyz[j] < 0) {
    //   //   loc_xyz[j] -= 1.0;
    //   // }
    // }
    // VoxelLoc position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    VoxelLoc position(pv.point_world, voxel_size);
    auto iter = feat_map.find(position);
    if (iter != feat_map.end()) {
      feat_map[position]->temp_points.push_back(pv);
      feat_map[position]->new_pts_num++;
    } else {
      auto octo_tree = new OctoTree(max_layer, 0, plane_fixed_pts_num, cov_fixed_pts_num, planer_threshold);
      feat_map[position] = octo_tree;
      octo_tree->quater_length = voxel_size / 4;
      octo_tree->voxel_center[0] = (0.5 + position.x) * voxel_size;
      octo_tree->voxel_center[1] = (0.5 + position.y) * voxel_size;
      octo_tree->voxel_center[2] = (0.5 + position.z) * voxel_size;
      octo_tree->temp_points.push_back(pv);
      octo_tree->new_pts_num++;
      // octo_tree->each_layer_min_pts_num = layer_point_size;
    }
  }
  for (auto &iter : feat_map) {
    iter.second->InitOctoTree();
  }
}

void UpdateVoxelMap(const std::vector<PointWithCov> &input_points, const float voxel_size, const int max_layer,
                    const int plane_fixed_pts_num, const int cov_fixed_pts_num, const float planer_threshold,
                    std::unordered_map<VoxelLoc, OctoTree *> &feat_map) {
  for (const auto &pv : input_points) {
    VoxelLoc position(pv.point_world, voxel_size);
    auto iter = feat_map.find(position);
    if (iter != feat_map.end()) {
      feat_map[position]->UpdateOctoTree(pv);
    } else {
      auto octo_tree = new OctoTree(max_layer, 0, plane_fixed_pts_num, cov_fixed_pts_num, planer_threshold);
      feat_map[position] = octo_tree;
      feat_map[position]->quater_length = voxel_size / 4;
      feat_map[position]->voxel_center[0] = (0.5 + position.x) * voxel_size;
      feat_map[position]->voxel_center[1] = (0.5 + position.y) * voxel_size;
      feat_map[position]->voxel_center[2] = (0.5 + position.z) * voxel_size;
      feat_map[position]->UpdateOctoTree(pv);
    }
  }
}

void FreeOctoTree(OctoTree *tree) {
  if (tree->state == 0) {
    delete tree->plane_ptr;
    delete tree;
  } else {
    delete tree->plane_ptr;
    for (auto &leaf : tree->leaves) {
      if (leaf) {
        FreeOctoTree(leaf);
      }
    }
    delete tree;
  }
}

void FreeVoxelMap(std::unordered_map<VoxelLoc, OctoTree *> &feat_map) {
  for (auto &iter : feat_map) {
    FreeOctoTree(iter.second);
  }
  OctoTree::plane_id = 0;
  feat_map.clear();
}

#ifdef USE_PATC_API

void MapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b) {
  r = 255;
  g = 255;
  b = 255;

  if (v < vmin) {
    v = vmin;
  }

  if (v > vmax) {
    v = vmax;
  }

  double dr, dg, db;

  if (v < 0.1242) {
    dg = 0.504 + ((1. - 0.504) / 0.1242) * v;
    db = dr = 0.;
  } else if (v < 0.3747) {
    dg = 1.;
    dr = 0.;
    db = (v - 0.1242) * (1. / (0.3747 - 0.1242));
  } else if (v < 0.6253) {
    dr = (0.6253 - v) * (1. / (0.6253 - 0.3747));
    dg = 1.;
    db = (v - 0.3747) * (1. / (0.6253 - 0.3747));
  } else if (v < 0.8758) {
    dg = 0.;
    dr = 1.;
    db = (0.8758 - v) * (1. / (0.8758 - 0.6253));
  } else {
    db = 0.;
    dg = 0.;
    dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758));
  }

  r = (uint8_t)(255 * dr);
  g = (uint8_t)(255 * dg);
  b = (uint8_t)(255 * db);
}

void AddPlotPlane(mvt::protocol::PlotPlanes &plot_planes, const Plane &plane, std::array<uint8_t, 3> &rgb,
                  float radius) {
  auto plot_plan = plot_planes.mutable_planes()->Add();
  plot_plan->set_id(plane.id);
  if (plane.debug_points.empty()) {
    plot_plan->set_valid(false);
    return;
  }

  plot_plan->set_valid(true);
  plot_plan->set_radius(radius);
  auto pos = plot_plan->mutable_pose()->mutable_position();
  pos->set_x(plane.center.x());
  pos->set_y(plane.center.y());
  pos->set_z(plane.center.z());
  auto q = plot_plan->mutable_pose()->mutable_orientation();

  // Eigen::Vector3d normal = plane.normal.normalized();
  // // 在XY平面的投影向量
  // Eigen::Vector3d p_xy{normal.x(), normal.y(), 0.};
  // // Z轴的旋转角度
  // double angle_z = std::acos(p_xy.dot(Eigen::Vector3d::UnitX()));
  // if (normal.y() < 0.) {
  //   angle_z *= -1.;
  // }
  //
  // // 在YZ平面的投影向量
  // Eigen::Vector3d p_yz{0., normal.y(), normal.z()};
  // // X轴的旋转角度
  // double angle_x = std::acos(p_yz.dot(Eigen::Vector3d::UnitY()));
  // if (normal.z() < 0.) {
  //   angle_x *= -1.;
  // }
  //
  // // 在YZ平面的投影向量
  // Eigen::Vector3d p_zx{normal.x(), 0, normal.z()};
  // // Y轴的旋转角度
  // double angle_y = std::acos(p_zx.dot(Eigen::Vector3d::UnitZ()));
  // if (normal.x() < 0.) {
  //   angle_y *= -1.;
  // }

  // Eigen::Quaterniond eq{Eigen::AngleAxisd(angle_z, Eigen::Vector3d::UnitZ()) *
  //                       Eigen::AngleAxisd(angle_y, Eigen::Vector3d::UnitY()) *
  //                       Eigen::AngleAxisd(angle_x, Eigen::Vector3d::UnitX())};

  Eigen::Quaterniond eq = Eigen::Quaterniond().setFromTwoVectors(Eigen::Vector3d::UnitX(), plane.normal);

  q->set_x(eq.x());
  q->set_y(eq.y());
  q->set_z(eq.z());
  q->set_w(eq.w());

  auto coeff = plot_plan->mutable_coefficient();
  coeff->Reserve(4);
  coeff->AddAlreadyReserved((float)plane.normal.x());
  coeff->AddAlreadyReserved((float)plane.normal.y());
  coeff->AddAlreadyReserved((float)plane.normal.z());
  coeff->AddAlreadyReserved((float)plane.d);
  auto cov = plot_plan->mutable_cov();
  cov->Reserve(9);
  for (int i = 0; i < 9; i++) {
    cov->AddAlreadyReserved((float)plane.covariance.data()[i]);
  }
  auto color = plot_plan->mutable_rgb();
  color->Reserve(3);
  color->AddAlreadyReserved(rgb[0]);
  color->AddAlreadyReserved(rgb[1]);
  color->AddAlreadyReserved(rgb[2]);

  // 设置平面平均距离误差
  plot_plan->set_dist_err(plane.dist_err);

  // 调试点
  if (!plane.debug_points.empty()) {
    auto pts = plot_plan->mutable_points();
    pts->Resize(3 * (int)plane.debug_points.size(), 0.);
    auto pt_ptr = pts->mutable_data();
    for (int i = 0; i < plane.debug_points.size(); i++) {
      pt_ptr[i * 3] = (float)plane.debug_points[i].x();
      pt_ptr[i * 3 + 1] = (float)plane.debug_points[i].y();
      pt_ptr[i * 3 + 2] = (float)plane.debug_points[i].z();
    }
  }
}

void PubVoxelMap(const std::string &name, const std::unordered_map<VoxelLoc, OctoTree *> &voxel_map) {
  PATC_ZoneScopedN("PubVoxelMap");
  // double max_trace = 0.25;
  // double pow_num = 0.2;
  const double bad_plane_val = 0.05;
  std::array<uint8_t, 3> rgb{};

  if (voxel_map.empty()) {
    return;
  }

  int max_layer = voxel_map.begin()->second->max_layer;

  std::vector<Plane *> pub_plane_list;
  for (const auto &iter : voxel_map) {
    GetUpdatePlane(iter.second, max_layer, pub_plane_list);
  }

  mvt::protocol::PlotPlanes plot_planes;
  plot_planes.mutable_header()->set_name(name);
  plot_planes.mutable_header()->set_time(PATC_GetTime);

  for (auto &plane : pub_plane_list) {
    MapJet(plane->dist_err / bad_plane_val, 0, 1, rgb[0], rgb[1], rgb[2]);
    AddPlotPlane(plot_planes, *plane, rgb, plane->show_radius / 2);
    plane->debug_points.clear();
    // plane->debug_points.shrink_to_fit();
    plane->is_update = false;
  }

  if (plot_planes.planes_size() > 0) {
    PATC_Pub(plot_planes);
  }
  // printf("plane_id: %d\n", OctoTree::plane_id);
  std::vector<double> plane_id(1);
  plane_id.at(0) = OctoTree::plane_id;
  PATC_Plot("odometry plane_id", plane_id);
}

void PubMatchResult(const std::string &name, const std::vector<VoxelMap::ptpl> &ptpl_list) {
  PATC_ZoneScopedN("PubMatchResult");
  const double bad_plane_val = 0.02;
  std::array<uint8_t, 3> rgb{};

  printf("match result size: %zu\n", ptpl_list.size());

  // plane的地址
  std::set<uint64_t> plane_ptr;
  int cnt = 0;
  // 先准备显示点
  try {
    for (auto &i : ptpl_list) {
      auto plane = (Plane *)i.plane_ptr;
      plane->debug_points.emplace_back(i.point_world.cast<float>());
      plane_ptr.emplace((uint64_t)plane);
      cnt++;
    }
  } catch (const std::exception &e) {
    printf("[%d] %s\n", cnt, e.what());
    exit(0);
  }

  // 发送
  mvt::protocol::PlotPlanes plot_planes;
  plot_planes.mutable_header()->set_name(name);
  plot_planes.mutable_header()->set_time(PATC_GetTime);

  for (auto &i : plane_ptr) {
    auto plane = (Plane *)i;
    MapJet(plane->dist_err / bad_plane_val, 0, 1, rgb[0], rgb[1], rgb[2]);
    AddPlotPlane(plot_planes, *plane, rgb, plane->show_radius / 2);
    plane->debug_points.clear();
    // plane->debug_points.shrink_to_fit();
  }
  PATC_Pub(plot_planes);
}

#endif

}  // namespace voxel_map