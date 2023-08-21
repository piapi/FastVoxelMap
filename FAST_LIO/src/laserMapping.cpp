// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>
#include <VoxelMap/voxel_map.h>
#include <unordered_map>

#ifdef USE_PATC_API
// 性能分析工具客户端
#include "patc/patc_api.h"
#include "mvt_msg_plot.pb.h"
#include "mvt_msg_keyframe.pb.h"
// #else
// #include "not_use_patc.h"
#endif
#define INIT_TIME (0.1)
#define LASER_POINT_COV (0.001)
#define MAXN (720000)
#define PUBFRAME_PERIOD (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
// T1为雷达初始时间戳，s_plot为整个流程耗时，s_plot2特征点数量,s_plot3为kdtree增量时间，s_plot4为kdtree搜索耗时，s_plot5为kdtree删除点数量
// ，s_plot6为kdtree删除耗时，s_plot7为kdtree初始大小，s_plot8为kdtree结束大小,s_plot9为平均消耗时间，s_plot10为添加点数量，s_plot11为点云预处理的总时间
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];

// 定义全局变量，用于记录时间,match_time为匹配时间，solve_time为求解时间，solve_const_H_time为求解H矩阵时间
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
// kdtree_size_st为ikd-tree获得的节点数，kdtree_size_end为ikd-tree结束时的节点数，add_point_size为添加点的数量，kdtree_delete_counter为删除点的数量
int kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
// runtime_pos_log运行时的log是否开启，pcd_save_en是否保存pcd文件，time_sync_en是否同步时间
bool runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0};   // 残差，点到面距离平方和
float DET_RANGE = 300.0f;         // 设置的当前雷达系中心到各个地图边缘的距离
const float MOV_THRESHOLD = 1.5f; // 设置的当前雷达系中心到各个地图边缘的权重
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;                 // 设置根目录
string map_file_path, lid_topic, imu_topic; // 设置地图文件路径，雷达topic，imu topic

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
// 设置立方体长度，视野一半的角度，视野总角度，总距离，雷达结束时间，雷达初始时间
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
// 设置有效特征点数，时间log计数器, scan_count：接收到的激光雷达Msg的总数，publish_count：接收到的IMU的Msg的总数
int effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
// 设置迭代次数，下采样的点数，最大迭代次数，有效点数
int iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool point_selected_surf[100000] = {0}; // 是否为平面特征点
// lidar_pushed：用于判断激光雷达数据是否从缓存队列中拿到meas中的数据, flg_EKF_inited用于判断EKF是否初始化完成
bool lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
// 设置是否发布激光雷达数据，是否发布稠密数据，是否发布激光雷达数据的身体数据
bool scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;

// 设置地图更新方式是fast_lio还是voxel map
int maping_type = 0;

// voxel map的参数
double ranging_sigma = 0, angle_sigma = 0, max_voxel_size = 0;
int max_layer = 3;
int cov_fixed_pts_num = 0, plane_fixed_pts_num = 0;
double plane_min_eigen_value = 0;
vector<int> each_layer_least_pts_num(7, 0);

bool pub_voxel_map = true;
int pub_max_voxel_layer = 1; // only publish 0,1,2 layer's plane
bool pub_point_cloud = true;
bool dense_map_enable = false;
int pub_point_cloud_skip = 5; // publish one points per five points

std::unordered_map<VoxelMap::VoxelLoc, VoxelMap::OctoTree *> voxel_map_; // 八叉树存储结构
std::vector<Eigen::Matrix3d> body_var;
std::vector<Eigen::Matrix3d> crossmat_list;

vector<vector<int>> pointSearchInd_surf;      // 每个点的索引,暂时没用到
vector<BoxPointType> cub_needrm;              // ikd-tree中，地图需要移除的包围盒序列
vector<PointVector> Nearest_Points;           // 每个点的最近点序列
vector<double> extrinT(3, 0.0);               // 雷达相对于IMU的外参T
vector<double> extrinR(9, 0.0);               // 雷达相对于IMU的外参R
deque<double> time_buffer;                    // 激光雷达数据时间戳缓存队列
deque<PointCloudXYZI::Ptr> lidar_buffer;      // 记录特征提取或间隔采样后的lidar（特征）数据
deque<sensor_msgs::Imu::ConstPtr> imu_buffer; // IMU数据缓存队列

// 一些点云变量
PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());           // 提取地图中的特征点，IKD-tree获得
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());        // 去畸变的特征
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());        // 畸变纠正后降采样的单帧点云，lidar系
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());       // 畸变纠正后降采样的单帧点云，w系
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));       // 特征点在地图中对应点的，局部平面参数,w系
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1)); // laserCloudOri是畸变纠正后降采样的单帧点云，body系
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1)); // 对应点法相量
PointCloudXYZI::Ptr _featsArray;                                  // ikd-tree中，map需要移除的点云序列

// 下采样的体素点云
pcl::VoxelGrid<PointType> downSizeFilterSurf; // 单帧内降采样使用voxel grid
pcl::VoxelGrid<PointType> downSizeFilterMap;  // 未使用

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);  // 雷达相对于body系的X轴方向的点
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0); // 雷达相对于world系的X轴方向的点
Eigen::Vector3d euler_cur;                    // 当前的欧拉角
Eigen::Vector3d position_last(Zero3d);        // 上一帧的位置
Eigen::Vector3d Lidar_T_wrt_IMU(Zero3d);      // T lidar to imu (imu = r * lidar + t)
Eigen::Matrix3d Lidar_R_wrt_IMU(Eye3d);       // R lidar to imu (imu = r * lidar + t)

/*** EKF inputs and output ***/
// ESEKF操作
MeasureGroup Measures;
Eigen::Matrix3d rot_var; // 旋转协方差
Eigen::Matrix3d t_var;   // 平移协方差
Eigen::Matrix3d rot_end;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf; // 状态，噪声维度，输入
state_ikfom state_point;                         // 状态

vect3 pos_lid; // world系下lidar坐标

// 输出的路径参数
nav_msgs::Path path;                      // 包含了一系列位姿
nav_msgs::Odometry odomAftMapped;         // 只包含了一个位姿
geometry_msgs::Quaternion geoQuat;        // 四元数
geometry_msgs::PoseStamped msg_body_pose; // 位姿

// 激光和imu处理操作
shared_ptr<Preprocess> p_pre(new Preprocess()); // 定义指向激光雷达数据的预处理类Preprocess的智能指针
shared_ptr<ImuProcess> p_imu(new ImuProcess()); // 定义指向IMU数据预处理类ImuProcess的智能指针

const bool var_contrast(VoxelMap::PointWithCov &x, VoxelMap::PointWithCov &y)
{
    return (x.cov.diagonal().norm() < y.cov.diagonal().norm());
};

// 按下ctrl+c后唤醒所有线程
void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all(); //  会唤醒所有等待队列中阻塞的线程 线程被唤醒后，会通过轮询方式获得锁，获得锁前也一直处理运行状态，不会被再次阻塞。
}

inline void dump_lio_state_to_log(FILE *fp)
{
    Eigen::Vector3d rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                            // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2));    // Pos
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                 // omega
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2));    // Vel
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                 // Acc
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));       // Bias_g
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));       // Bias_a
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a
    fprintf(fp, "\r\n");
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const *const pi, PointType *const po, state_ikfom &s)
{
    Eigen::Vector3d p_body(pi->x, pi->y, pi->z);
    Eigen::Vector3d p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void pointBodyToWorld(PointType const *const pi, PointType *const po)
{
    Eigen::Vector3d p_body(pi->x, pi->y, pi->z);
    Eigen::Vector3d p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template <typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    Eigen::Vector3d p_body(pi[0], pi[1], pi[2]);
    Eigen::Vector3d p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const *const pi, PointType *const po)
{
    Eigen::Vector3d p_body(pi->x, pi->y, pi->z);
    Eigen::Vector3d p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const *const pi, PointType *const po)
{
    Eigen::Vector3d p_body_lidar(pi->x, pi->y, pi->z);
    Eigen::Vector3d p_body_imu(state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}
// 得到被剔除的点
void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history); // 返回被剔除的点
    // for (int i = 0; i < points_history.size(); i++);
    // _featsArray->push_back(points_history[i]); // 存入到缓存中，后面没有用到该数据
}

// 在拿到eskf前馈结果后，动态调整地图区域，防止地图过大而内存溢出，类似LOAM中提取局部地图的方法
BoxPointType LocalMap_Points;      // ikd-tree中,局部地图的包围盒角点
bool Localmap_Initialized = false; // 局部地图是否初始化
void lasermap_fov_segment()
{
    cub_needrm.clear(); // 清空需要移除的区域
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;
    // X轴分界点转换到w系下，好像没有用到
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    // global系下lidar位置
    Eigen::Vector3d pos_LiD = pos_lid;
    // 初始化局部地图包围盒角点，以为w系下lidar位置为中心,得到长宽高200*200*200的局部地图
    if (!Localmap_Initialized)
    { // 系统起始需要初始化局部地图的大小和位置
        for (int i = 0; i < 3; i++)
        {
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    // 各个方向上Lidar与局部地图边界的距离，或者说是lidar与立方体盒子六个面的距离
    float dist_to_map_edge[3][2];
    bool need_move = false;
    // 当前雷达系中心到各个地图边缘的距离
    for (int i = 0; i < 3; i++)
    {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        // 与某个方向上的边界距离（例如1.5*300m）太小，标记需要移除need_move，参考论文Fig3
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
            need_move = true;
    }
    // 不需要挪动就直接退回了
    if (!need_move)
        return;
    // 否则需要计算移动的距离
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    // 新的局部地图盒子边界点
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD - 1)));
    for (int i = 0; i < 3; i++)
    {
        tmp_boxpoints = LocalMap_Points;
        // 与包围盒最小值边界点距离
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints); // 移除较远包围盒
        }
        else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    // 使用Boxs删除指定盒内的点
    if (cub_needrm.size() > 0)
        kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}
// 除了AVIA类型之外的雷达点云回调函数，将数据引入到buffer当中
void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    mtx_buffer.lock(); // 加锁
    scan_count++;
    double preprocess_start_time = omp_get_wtime(); // 记录时间
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);                                       // 点云预处理
    lidar_buffer.push_back(ptr);                                    // 将点云放入缓冲区
    time_buffer.push_back(msg->header.stamp.toSec());               // 将时间放入缓冲区
    last_timestamp_lidar = msg->header.stamp.toSec();               // 记录最后一个时间
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time; // 预处理时间
    mtx_buffer.unlock();
    sig_buffer.notify_all(); // 唤醒所有线程
}

double timediff_lidar_wrt_imu = 0.0;
bool timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();

    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty())
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n", last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);

    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
    publish_count++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
    if (p_pre->lidar_type == MID360)
    {
        msg->linear_acceleration.x *= 9.8;
        msg->linear_acceleration.y *= 9.8;
        msg->linear_acceleration.z *= 9.8;
    }
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp =
            ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}
static void FillPointCloudMsg(const PointCloudXYZI::Ptr &cloud, mvt::protocol::PointCloudXYZI *msg)
{
    msg->mutable_point()->Clear();
    msg->mutable_point()->Reserve((int)cloud->size());

    for (auto &pt : cloud->points)
    {
        auto p = msg->mutable_point()->Add();
        p->set_x(pt.x);
        p->set_y(pt.y);
        p->set_z(pt.z);
        p->set_intensity(pt.intensity);
    }
}

double lidar_mean_scantime = 0.0;
int scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty())
    {
        return false;
    }

    /*** push a lidar scan ***/
    if (!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if (imu_time > lidar_end_time)
            break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int process_increments = 0;
void map_incremental() // 地图的增量更新，主要完成对ikd-tree的地图建立
{
    PointVector PointToAdd;                         // 需要加入到ikd-tree中的点云
    PointVector PointNoNeedDownsample;              // 加入ikd-tree时，不需要降采样的点云
    PointToAdd.reserve(feats_down_size);            // 构建的地图点
    PointNoNeedDownsample.reserve(feats_down_size); // 构建的地图点，不需要降采样的点云
    // 根据点与所在包围盒中心点的距离，分类是否需要降采样
    for (int i = 0; i < feats_down_size; i++)
    {
        // 转换到世界坐标系下
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        // 判断是否有关键点需要加到地图中
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i]; // 获取附近的点云
            bool need_add = true;                               // 是否需要加入到地图中
            BoxPointType Box_of_Point;                          // 点云所在的包围盒
            PointType downsample_result, mid_point;             // 降采样结果，中点
            // filter_size_map_min是地图体素降采样的栅格边长，设为0.1m
            // mid_point即为该特征点所属的栅格的中心点坐标
            mid_point.x = floor(feats_down_world->points[i].x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            // 当前点与box中心的距离
            float dist = calc_dist(feats_down_world->points[i], mid_point);
            // 判断最近点在x、y、z三个方向上，与中心的距离，判断是否加入时需要降采样
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min)
            {
                // 若三个方向距离都大于地图栅格半轴长，无需降采样
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            // 判断当前点的 NUM_MATCH_POINTS 个邻近点与包围盒中心的范围
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) // 若邻近点数小于NUM_MATCH_POINTS，则直接跳出，添加到PointToAdd中
                    break;
                // 如果存在邻近点到中心的距离小于当前点到中心的距离，则不需要添加当前点
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add)
                PointToAdd.push_back(feats_down_world->points[i]); // 加入到PointToAdd中
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]); // 如果周围没有点或者没有初始化EKF，则加入到PointToAdd中
        }
    }

    double st_time = omp_get_wtime();                                  // 记录起始时间
    add_point_size = ikdtree.Add_Points(PointToAdd, true);             // 加入点时需要降采样
    ikdtree.Add_Points(PointNoNeedDownsample, false);                  // 加入点时不需要降采样
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size(); // 计算总共加入ikd-tree的点的数量
    kdtree_incremental_time = omp_get_wtime() - st_time;               // kdtree建立时间更新
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1)); // 创建一个点云用于存储等待发布的点云
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());         // 创建一个点云用于存储等待保存的点云
void publish_frame_world(const ros::Publisher &pubLaserCloudFull)
{
    if (scan_pub_en) // 设置是否发布激光雷达数据，是否发布稠密数据，是否发布激光雷达数据的身体数据
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body); // 判断是否需要降采样
        int size = laserCloudFullRes->points.size();                                             // 获取待转换点云的大小
        PointCloudXYZI::Ptr laserCloudWorld(
            new PointCloudXYZI(size, 1)); // 创建一个点云用于存储转换到世界坐标系的点云

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i],
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }
    // 把结果压入到pcd中
    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(
            new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i],
                                &laserCloudWorld->points[i]); // 转换到世界坐标系
        }
        *pcl_wait_save += *laserCloudWorld; // 把结果压入到pcd中

        static int scan_wait_num = 0;
        scan_wait_num++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
        {
            pcd_index++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void publish_frame_body(const ros::Publisher &pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i],
                               &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(const ros::Publisher &pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld(
        new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i],
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void publish_map(const ros::Publisher &pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

template <typename T>
void set_posestamp(T &out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
}

void publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time); // ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x,
                                    odomAftMapped.pose.pose.position.y,
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "body"));
}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0)
    {
        // pathfile << msg_body_pose.header.stamp.toSec() << " " << msg_body_pose.pose.position.x << " " << msg_body_pose.pose.position.y << " " << msg_body_pose.pose.position.z << endl;
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

// 计算残差信息
void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime(); // 计算匹配的开始时间
    laserCloudOri->clear();               // 将body系的有效点云存储清空
    corr_normvect->clear();               // 将对应的法向量清空
    total_residual = 0.0;

    // 对降采样后的每个特征点进行残差计算

    effct_feat_num = 0; // 有效特征点数

    /** 最接近曲面搜索和残差计算  **/
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body = feats_down_body->points[i];   // 获取降采样后的每个特征点
        PointType &point_world = feats_down_world->points[i]; // 获取降采样后的每个特征点的世界坐标

        /* transform to world frame */
        // 将点转换至世界坐标系下
        Eigen::Vector3d p_body(point_body.x, point_body.y, point_body.z);
        Eigen::Vector3d p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos); // 将点转换至世界坐标系下,从而来计算残差
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge) // 如果收敛了
        {
            /** Find the closest surfaces in the map **/
            // 在已构造的地图上查找特征点的最近邻
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            // 如果最近邻的点数小于NUM_MATCH_POINTS或者最近邻的点到特征点的距离大于5m，则认为该点不是有效点
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false
                                                                                                                                : true;
        }

        if (!point_selected_surf[i]) // 如果该点不是有效点
            continue;

        VF(4)
        pabcd;                          // 平面点信息
        point_selected_surf[i] = false; // 将该点设置为无效点，用来计算是否为平面点

        // 拟合平面方程ax+by+cz+d=0并求解点到平面距离
        if (esti_plane(pabcd, points_near, 0.1f)) // 找平面点法向量寻找，common_lib.h中的函数
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3); // 计算点到平面的距离
            float si = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());                                                  // 计算残差

            if (si > 0.9) // 如果残差大于阈值，则认为该点是有效点
            {
                point_selected_surf[i] = true;   // 再次回复为有效点
                normvec->points[i].x = pabcd(0); // 将法向量存储至normvec
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2; // 将点到平面的距离存储至normvec的intensit中
                res_last[i] = abs(pd2);             // 将残差存储至res_last
            }
        }
    }
    for (int i = 0; i < feats_down_size; i++)
    {
        // 根据point_selected_surf状态判断哪些点是可用的
        if (point_selected_surf[i])
        {
            // body点存到laserCloudOri中
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i]; // 将降采样后的每个特征点存储至laserCloudOri
            // 拟合平面点存到corr_normvect中
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i]; // 计算总残差
            effct_feat_num++;              // 有效特征点数加1
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }
    // res_mean_last = total_residual / effct_feat_num; // 计算残差平均值

    // 维数12是因为其他状态对应的雅可比矩阵值为0
    ekfom_data.h_x = Eigen::MatrixXd::Zero(effct_feat_num, 12);
    ekfom_data.h.resize(effct_feat_num);

    // index.resize(effect_feat_num_);

    const Eigen::Matrix3d Rt = s.rot.toRotationMatrix().transpose();
    const Eigen::Matrix3d offset_R_L_I_T = s.offset_R_L_I.toRotationMatrix().transpose();

    // #ifdef MP_EN
    //   omp_set_num_threads(omp_get_thread_num);
    // #ifdef _MSC_VER
    // #pragma omp parallel for
    // #else
    // #pragma omp parallel for default(none) shared(Rt, offset_R_L_I_T, state_point, ekfom_data)
    // #endif
    // #endif

    // std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const int &i) {
    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &pt = laserCloudOri->points[i];
        Eigen::Vector3d pt_v(pt.x, pt.y, pt.z);
        Eigen::Matrix3d pt_hat;
        pt_hat << SKEW_SYM_MATRX(pt_v);
        Eigen::Vector3d pt_imu = s.offset_R_L_I * pt_v + s.offset_T_L_I;
        Eigen::Matrix3d pt_imu_hat;
        pt_imu_hat << SKEW_SYM_MATRX(pt_imu);

        // get the normal vector of closest surface
        const PointType &norm_p = corr_normvect->points[i];
        Eigen::Vector3d norm_vec(norm_p.x, norm_p.y, norm_p.z);

        // calculate the Measurement Jacobian matrix H
        // norm_vec 是3x1
        Eigen::Vector3d C(Rt * norm_vec);

        // pt_imu_hat = - pt_imu_hat^T
        // A = - pt_imu_hat^T * R_G^T * norm_vec = -(R_G * pt_imu_hat)^T * norm_vec
        Eigen::Vector3d A(pt_imu_hat * C);
        if (extrinsic_est_en)
        {
            // B = - (R_G * R_I * pt_hat)^T * norm_vec
            Eigen::Vector3d B(pt_hat * offset_R_L_I_T * C);
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, A[0], A[1], A[2], B[0], B[1], B[2], C[0], C[1],
                C[2];
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, A[0], A[1], A[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }
        // todo 观测方法与论文不一致
        // Measurement: distance to the closest surface
        // 并不是真正到平面的距离
        // r = z - h = 0 - h
        ekfom_data.h(i) = -norm_p.intensity;
    }
}

void HShareModelVoxelMap(state_ikfom &s, esekfom::esekf<state_ikfom, 12, input_ikfom>::cov &P,
                                          esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    
        laserCloudOri->clear();               // 将body系的有效点云存储清空
        corr_normvect->clear();               // 将对应的法向量清空
        total_residual = 0.0;

        // 对降采样后的每个特征点进行残差计算
        effct_feat_num = 0; // 有效特征点数

        std::vector<VoxelMap::ptpl> ptpl_list;
        vector<VoxelMap::PointWithCov> pv_list;

        for (int i = 0; i < feats_down_size; i++)
        {
        VoxelMap::PointWithCov pv;
        Eigen::Vector3d p_body(feats_down_body->points[i].x,
                               feats_down_body->points[i].y,
                               feats_down_body->points[i].z);
        Eigen::Vector3d point_world(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);

        pv.point << p_body.x(), p_body.y(), p_body.z();
        pv.point_world << point_world.x(), point_world.y(), point_world.z();

        Eigen::Matrix3d body_cov = body_var[i];
        Eigen::Matrix3d point_cross_mat = crossmat_list[i];
        pv.cov = s.rot * body_cov * s.rot.toRotationMatrix().transpose() +
             (-point_cross_mat) * P.block<3, 3>(3, 3) * (-point_cross_mat.transpose()) + P.block<3, 3>(0, 0);
        pv_list.push_back(pv);
        }
        std::vector<Eigen::Vector3d> non_match_list;
        // cout << "pv_list:" << pv_list.size() << endl;
        VoxelMap::BuildResidualListOMP(voxel_map_, max_voxel_size, 3.0, max_layer, pv_list,
                                       ptpl_list, non_match_list);
        // VoxelMap::PubMatchResult("MatchRe/sult", ptpl_list);
        for (int i = 0; i < ptpl_list.size(); i++)
        {
        PointType pi_body;
        PointType pi_world;
        PointType pl;
        VoxelMap::Plane *accordingPlane = reinterpret_cast<VoxelMap::Plane *>(ptpl_list[i].plane_ptr);
        // std::cout << "ptpl_list " << ptpl_list[i].normal.transpose() << std::endl;
        // std::cout << "accordingPlane " << accordingPlane->normal.transpose() << std::endl;
        pi_body.x = ptpl_list[i].point(0);
        pi_body.y = ptpl_list[i].point(1);
        pi_body.z = ptpl_list[i].point(2);

        pl.x = accordingPlane->normal(0);
        pl.y = accordingPlane->normal(1);
        pl.z = accordingPlane->normal(2);

        pl.intensity = ptpl_list[i].dist;
        laserCloudOri->points[i] = pi_body;
        corr_normvect->points[i] = pl;
        effct_feat_num++;
        // total_residual += fabs(dis);
        }

        if (effct_feat_num < 1)
        {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
        }
        // res_mean_last = total_residual / effct_feat_num; // 计算残差平均值

        // 维数12是因为其他状态对应的雅可比矩阵值为0
        ekfom_data.h_x = Eigen::MatrixXd::Zero(effct_feat_num, 12);
        ekfom_data.h.resize(effct_feat_num);

        // index.resize(effect_feat_num_);

        const Eigen::Matrix3d Rt = s.rot.toRotationMatrix().transpose();
        const Eigen::Matrix3d offset_R_L_I_T = s.offset_R_L_I.toRotationMatrix().transpose();

        for (int i = 0; i < effct_feat_num; i++)
        {
        const PointType &pt = laserCloudOri->points[i];
        Eigen::Vector3d pt_v(pt.x, pt.y, pt.z);
        Eigen::Matrix3d pt_hat;
        pt_hat << SKEW_SYM_MATRX(pt_v);
        Eigen::Vector3d pt_imu = s.offset_R_L_I * pt_v + s.offset_T_L_I;
        Eigen::Matrix3d pt_imu_hat;
        pt_imu_hat << SKEW_SYM_MATRX(pt_imu);

        // get the normal vector of closest surface
        const PointType &norm_p = corr_normvect->points[i];
        Eigen::Vector3d norm_vec(norm_p.x, norm_p.y, norm_p.z);

        // calculate the Measurement Jacobian matrix H
        // norm_vec 是3x1
        Eigen::Vector3d C(Rt * norm_vec);

        // pt_imu_hat = - pt_imu_hat^T
        // A = - pt_imu_hat^T * R_G^T * norm_vec = -(R_G * pt_imu_hat)^T * norm_vec
        Eigen::Vector3d A(pt_imu_hat * C);
        if (extrinsic_est_en)
        {
            // B = - (R_G * R_I * pt_hat)^T * norm_vec
            Eigen::Vector3d B(pt_hat * offset_R_L_I_T * C);
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, A[0], A[1], A[2], B[0], B[1], B[2], C[0], C[1],
                C[2];
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, A[0], A[1], A[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }
        // todo 观测方法与论文不一致
        // Measurement: distance to the closest surface
        // 并不是真正到平面的距离
        // r = z - h = 0 - h
        ekfom_data.h(i) = -norm_p.intensity;
        }
}
int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;
    // pathfile.open(filePath); //打开文件
    // if (!pathfile)
    // {
    //     cout << "打开文件失败" << endl;
    //     exit(1);
    // }
    nh.param<bool>("publish/path_en", path_en, true);
    nh.param<bool>("publish/scan_publish_en", scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en", dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en, true);
    nh.param<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
    nh.param<string>("map_file_path", map_file_path, "");
    nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);
    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    nh.param<double>("filter_size_corner", filter_size_corner_min, 0.5);
    nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);
    nh.param<double>("filter_size_map", filter_size_map_min, 0.5);
    nh.param<double>("cube_side_length", cube_len, 200);
    nh.param<float>("mapping/det_range", DET_RANGE, 300.f);
    nh.param<double>("mapping/fov_degree", fov_deg, 180);
    nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);
    nh.param<double>("mapping/acc_cov", acc_cov, 0.1);
    nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
    nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());

    // voxel map
    nh.param<int>("preprocess/maping_type", maping_type, 0);
    nh.param<double>("voxel_map/ranging_sigma", ranging_sigma, 0.02);
    nh.param<double>("voxel_map/angle_sigma", angle_sigma, 0.05);
    nh.param<double>("voxel_map/max_voxel_size", max_voxel_size, 3.2);
    nh.param<int>("voxel_map/max_layer", max_layer, 3);
    nh.param<int>("voxel_map/cov_fixed_pts_num", cov_fixed_pts_num, 200);
    nh.param<int>("voxel_map/plane_fixed_pts_num", plane_fixed_pts_num, 300);
    nh.param<double>("voxel_map/plane_min_eigen_value", plane_min_eigen_value, 0.01);
    nh.param<vector<int>>("voxel_map/each_layer_least_pts_num", each_layer_least_pts_num, vector<int>());
    nh.param<bool>("voxel_map/pub_voxel_map", pub_voxel_map, false);
    nh.param<int>("voxel_map/pub_max_voxel_layer", pub_max_voxel_layer, 0);
    nh.param<bool>("voxel_map/pub_point_cloud", pub_point_cloud, true);
    nh.param<int>("voxel_map/pub_point_cloud_skip", pub_point_cloud_skip, 1);
    nh.param<bool>("voxel_map/dense_map_enable", dense_map_enable, false);
    cout << "p_pre->lidar_type " << p_pre->lidar_type << endl;

    path.header.stamp = ros::Time::now();
    path.header.frame_id = "camera_init";
    int cnt = 0;
    /*** variables definition ***/
    /** 变量定义
     * effct_feat_num          （后面的代码中没有用到该变量）
     * frame_num                雷达总帧数
     * deltaT                   （后面的代码中没有用到该变量）
     * deltaR                   （后面的代码中没有用到该变量）
     * aver_time_consu          每帧平均的处理总时间
     * aver_time_icp            每帧中icp的平均时间
     * aver_time_match          每帧中匹配的平均时间
     * aver_time_incre          每帧中ikd-tree增量处理的平均时间
     * aver_time_solve          每帧中计算的平均时间
     * aver_time_const_H_time   每帧中计算的平均时间（当H恒定时）
     * flg_EKF_converged        EKF收敛标志
     * EKF_stop_flg             EFK停止标识符
     * FOV_DEG                  （后面的代码中没有用到该变量）
     * HALF_FOV_COS             （后面的代码中没有用到该变量）
     * _featsArray              （后面的代码中没有用到该变量）
     */
    int effct_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool EKF_stop_flg = false, flg_EKF_converged = false;
    // FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    // HALF_FOV_COS = cos((FOV_DEG)*0.5 * PI_M / 180.0);

    // _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(Eigen::Vector3d(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(Eigen::Vector3d(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(Eigen::Vector3d(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(Eigen::Vector3d(b_acc_cov, b_acc_cov, b_acc_cov));

    double epsi[23] = {0.001};
    fill(epsi, epsi + 23, 0.001);
    if (maping_type == 0)
    {
        kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);
    }
    else if (maping_type == 1)
    {
        kf.init_voxelmap_share(get_f, df_dx, df_dw, HShareModelVoxelMap, NUM_MAX_ITERATIONS, epsi);  
    }

    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(), "w");

    ofstream fout_pre, fout_out, fout_dbg;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"), ios::out);
    if (fout_pre && fout_out)
        cout << "~~~~" << ROOT_DIR << " file opened" << endl;
    else
        cout << "~~~~" << ROOT_DIR << " doesn't exist" << endl;

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = (p_pre->lidar_type == AVIA || p_pre->lidar_type == MID360) ? nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("/path", 100000);
    // ros::Publisher voxel_map_pub = nh.advertise<visualization_msgs::MarkerArray>("/planes", 10000);
    //------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    // for Plane Map
    bool init_map = false;
    while (status)
    {
        if (flg_exit)
            break;
        ros::spinOnce();
        if (sync_packages(Measures))
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                continue;
            }

            double t0, t1, t2, t3, t4, t5, match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time = 0;
            t0 = omp_get_wtime();
            // std::cout << "lidar size: " << Measures.lidar->points.size() << std::endl;
            p_imu->Process(Measures, kf, feats_undistort);

            state_point = kf.get_x();

            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;
            /*** Segment the map in lidar FOV ***/

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            sort(feats_down_body->points.begin(), feats_down_body->points.end(),
                 time_list);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();
            // std::cout << state_point.grav[0] << state_point.grav[1] << state_point.grav[2] << std::endl;

            if (maping_type == 0)
            {
                /*** initialize the map kdtree ***/
                lasermap_fov_segment();
                if (ikdtree.Root_Node == nullptr)
                {
                    if (feats_down_size > 5)
                    {
                        ikdtree.set_downsample_param(filter_size_map_min);
                        feats_down_world->resize(feats_down_size);
                        for (int i = 0; i < feats_down_size; i++)
                        {
                            pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                        }
                        ikdtree.Build(feats_down_world->points);
                    }
                    continue;
                }
                int featsFromMapNum = ikdtree.validnum();
                kdtree_size_st = ikdtree.size();
            }
            else if (maping_type == 1)
            {
                // Eigen::Vector3d eulerAngle = SO3ToEuler(state_point.rot);

                // geometry_msgs::Quaternion q = tf::createQuaternionMsgFromRollPitchYaw(eulerAngle[0], eulerAngle[1], eulerAngle[2]);
                // Eigen::Quaterniond qq(q.w, q.x, q.y, q.z);
                rot_end = state_point.rot.toRotationMatrix();

                rot_var = kf.get_P().block<3, 3>(3, 3);
                t_var = kf.get_P().block<3, 3>(0, 0);
                if (flg_EKF_inited && !init_map)
                {
                    std::vector<VoxelMap::PointWithCov> pv_list;
                    PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI(feats_undistort->points.size(), 1));
                    for (int i = 0; i < feats_undistort->points.size(); i++)
                    {
                        VoxelMap::PointWithCov pv;

                        pointBodyToWorld(&(feats_undistort->points[i]), &(world_lidar->points[i]));
                        pv.point << feats_undistort->points[i].x, feats_undistort->points[i].y,
                            feats_undistort->points[i].z;
                        pv.point_world << world_lidar->points[i].x, world_lidar->points[i].y,
                            world_lidar->points[i].z;
                        Eigen::Vector3d point_this(feats_undistort->points[i].x,
                                                   feats_undistort->points[i].y,
                                                   feats_undistort->points[i].z);
                        // if z=0, error will occur in calcBodyCov. To be solved
                        if (point_this[2] == 0)
                        {
                            point_this[2] = 0.001;
                        }
                        Eigen::Matrix3d cov;
                        VoxelMap::CalcBodyCov(point_this, ranging_sigma, angle_sigma, cov);
                        // point_this += state_point.offset_T_L_I;
                        Eigen::Matrix3d point_crossmat;
                        point_crossmat << SKEW_SYM_MATRX(point_this);

                        cov = rot_end * cov * rot_end.transpose() + rot_end * (-point_crossmat) * rot_var * (-point_crossmat).transpose() * rot_end.transpose() + t_var;
                        pv.cov = cov;
                        // std::cout << "tnnd:" << pv.point.transpose() << std::enc
                        pv_list.push_back(pv);
                    }
                    VoxelMap::OctoTree::SetLayersParam(each_layer_least_pts_num);
                    VoxelMap::BuildVoxelMap(pv_list, max_voxel_size, max_layer,
                                            plane_fixed_pts_num, cov_fixed_pts_num, plane_min_eigen_value,
                                            voxel_map_);
                    std::cout << "build voxel map" << std::endl;
                    if (pub_voxel_map)
                    {
                        // VoxelMap::pubVoxelMap(voxel_map_, pub_max_voxel_layer, voxel_map_pub);
                        VoxelMap::PubVoxelMap("odoMap", voxel_map_);
                        // 发送点云数据

                        mvt::protocol::KeyFrame keyframe;
                        keyframe.mutable_header()->set_stamp(int64_t(Measures.lidar_beg_time));
                        keyframe.set_id(-1);
                        keyframe.mutable_header()->set_sensor_name("local frame");
                        keyframe.mutable_header()->set_time(PATC_GetTime);
                        auto pos = keyframe.mutable_pose()->mutable_position();
                        pos->set_x(0);
                        pos->set_y(0);
                        pos->set_z(0);
                        auto q = Eigen::Quaterniond::Identity();
                        keyframe.mutable_pose()->mutable_orientation()->set_x(q.x());
                        keyframe.mutable_pose()->mutable_orientation()->set_y(q.y());
                        keyframe.mutable_pose()->mutable_orientation()->set_z(q.z());
                        keyframe.mutable_pose()->mutable_orientation()->set_w(q.w());

                        auto pc = keyframe.mutable_pc();

                        FillPointCloudMsg(world_lidar, pc);
                        PATC_Pub(keyframe);
                    }
                    init_map = true;
                    continue;
                }
            }

            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            Eigen::Vector3d ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose() << " " << ext_euler.transpose() << " " << state_point.offset_T_L_I.transpose() << " " << state_point.vel.transpose()
                     << " " << state_point.bg.transpose() << " " << state_point.ba.transpose() << " " << state_point.grav << endl;

            if (0) // If you need to see map point, change to "if(1)"
            {
                PointVector().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);

            t2 = omp_get_wtime();

            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            if (maping_type == 0)
            {
                kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
                state_point = kf.get_x();
                euler_cur = SO3ToEuler(state_point.rot);
                pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
                geoQuat.x = state_point.rot.coeffs()[0];
                geoQuat.y = state_point.rot.coeffs()[1];
                geoQuat.z = state_point.rot.coeffs()[2];
                geoQuat.w = state_point.rot.coeffs()[3];
                /*** add the feature points to map kdtree ***/
                t3 = omp_get_wtime();
                map_incremental();
                t5 = omp_get_wtime();
            }
            else if (maping_type == 1)
            {
                body_var.resize(feats_down_size);
                crossmat_list.resize(feats_down_size);
                for (int i = 0; i < feats_down_size; i++)
                {
                    Eigen::Vector3d point_this(feats_down_body->points[i].x,
                                               feats_down_body->points[i].y,
                                               feats_down_body->points[i].z);
                    if (point_this[2] == 0)
                    {
                        point_this[2] = 0.001;
                    }
                    Eigen::Matrix3d cov;
                    VoxelMap::CalcBodyCov(point_this, ranging_sigma, angle_sigma, cov);
                    Eigen::Matrix3d point_crossmat;
                    point_crossmat << SKEW_SYM_MATRX(point_this);
                    crossmat_list[i] = point_crossmat;
                    body_var[i] = cov;
                }
                kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time, 1);
                state_point = kf.get_x();
                euler_cur = SO3ToEuler(state_point.rot);
                pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
                geoQuat.x = state_point.rot.coeffs()[0];
                geoQuat.y = state_point.rot.coeffs()[1];
                geoQuat.z = state_point.rot.coeffs()[2];
                geoQuat.w = state_point.rot.coeffs()[3];

                std::vector<VoxelMap::PointWithCov> pv_list;
                PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI(feats_down_size, 1));
                for (int i = 0; i < feats_down_size; i++)
                {
                    VoxelMap::PointWithCov pv;
                    pointBodyToWorld(&(feats_down_body->points[i]), &(world_lidar->points[i]));
                    // pv.point << feats_down_body->points[i].x, feats_down_body->points[i].y,
                    //     feats_down_body->points[i].z;
                    pv.point_world << world_lidar->points[i].x, world_lidar->points[i].y,
                        world_lidar->points[i].z;
                    Eigen::Matrix3d point_crossmat = crossmat_list[i];
                    Eigen::Matrix3d cov = body_var[i];
                    pv.cov = rot_end * cov * rot_end.transpose() +
                             rot_end * (-point_crossmat) * rot_var * (-point_crossmat).transpose() * rot_end.transpose() + t_var;
                    pv_list.push_back(pv);
                }
                std::sort(pv_list.begin(), pv_list.end(), var_contrast);
                VoxelMap::UpdateVoxelMap(pv_list, max_voxel_size, max_layer,
                                         plane_fixed_pts_num, cov_fixed_pts_num, plane_min_eigen_value,
                                         voxel_map_);
            }
            /******* Publish odometry *******/
            // std::vector<double> xyz = {kf.get_x().pos.x(), kf.get_x().pos.y(), kf.get_x().pos.z()};
            // PATC_Plot("predict xyz", xyz);
            publish_odometry(pubOdomAftMapped);

            /******* Publish points *******/
            if (path_en)
                publish_path(pubPath);
            if (scan_pub_en || pcd_save_en)
                publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en)
                publish_frame_body(pubLaserCloudFull_body);
            double t_update_end = omp_get_wtime();
            if (pub_voxel_map)
            {
                // VoxelMap::pubVoxelMap(voxel_map_, pub_max_voxel_layer, voxel_map_pub);
                VoxelMap::PubVoxelMap("odoMap", voxel_map_);
                // 发送点云数据
                PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI(feats_undistort->points.size(), 1));
                for (int i = 0; i < feats_undistort->points.size(); i++)
                {

                    pointBodyToWorld(&(feats_undistort->points[i]), &(world_lidar->points[i]));
                }
                mvt::protocol::KeyFrame keyframe;
                keyframe.mutable_header()->set_stamp(int64_t(Measures.lidar_beg_time));
                keyframe.set_id(-1);
                keyframe.mutable_header()->set_sensor_name("local frame");
                keyframe.mutable_header()->set_time(PATC_GetTime);
                auto pos = keyframe.mutable_pose()->mutable_position();
                pos->set_x(0);
                pos->set_y(0);
                pos->set_z(0);
                auto q = Eigen::Quaterniond::Identity();
                keyframe.mutable_pose()->mutable_orientation()->set_x(q.x());
                keyframe.mutable_pose()->mutable_orientation()->set_y(q.y());
                keyframe.mutable_pose()->mutable_orientation()->set_z(q.z());
                keyframe.mutable_pose()->mutable_orientation()->set_w(q.w());

                auto pc = keyframe.mutable_pc();

                FillPointCloudMsg(world_lidar, pc);
                PATC_Pub(keyframe);
            }
            // publish_effect_world(pubLaserCloudEffect);
            // publish_map(pubLaserCloudMap);

            /*** Debug variables ***/
            //     if (runtime_pos_log)
            //     {
            //         frame_num++;
            //         kdtree_size_end = ikdtree.size();
            //         aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
            //         aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + (t_update_end - t_update_start) / frame_num;
            //         aver_time_match = aver_time_match * (frame_num - 1) / frame_num + (match_time) / frame_num;
            //         aver_time_incre = aver_time_incre * (frame_num - 1) / frame_num + (kdtree_incremental_time) / frame_num;
            //         aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + (solve_time + solve_H_time) / frame_num;
            //         aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) / frame_num + solve_time / frame_num;
            //         T1[time_log_counter] = Measures.lidar_beg_time;
            //         s_plot[time_log_counter] = t5 - t0;                         // 整个流程总时间
            //         s_plot2[time_log_counter] = feats_undistort->points.size(); // 特征点数量
            //         s_plot3[time_log_counter] = kdtree_incremental_time;        // kdtree增量时间
            //         s_plot4[time_log_counter] = kdtree_search_time;             // kdtree搜索耗时
            //         s_plot5[time_log_counter] = kdtree_delete_counter;          // kdtree删除点数量
            //         s_plot6[time_log_counter] = kdtree_delete_time;             // kdtree删除耗时
            //         s_plot7[time_log_counter] = kdtree_size_st;                 // kdtree初始大小
            //         s_plot8[time_log_counter] = kdtree_size_end;                // kdtree结束大小
            //         s_plot9[time_log_counter] = aver_time_consu;                // 平均消耗时间
            //         s_plot10[time_log_counter] = add_point_size;                // 添加点数量
            //         time_log_counter++;
            //         printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n", t1 - t0, aver_time_match, aver_time_solve, t3 - t1, t5 - t3, aver_time_consu, aver_time_icp, aver_time_const_H_time);
            //         ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            //         fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose() << " " << ext_euler.transpose() << " " << state_point.offset_T_L_I.transpose() << " " << state_point.vel.transpose()
            //                  << " " << state_point.bg.transpose() << " " << state_point.ba.transpose() << " " << state_point.grav << " " << feats_undistort->points.size() << endl;
            //         dump_lio_state_to_log(fp);
            //     }
        }

        status = ros::ok();
        rate.sleep();
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name << endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    fout_out.close();
    fout_pre.close();

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(), "w");
        fprintf(fp2, "time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0; i < time_log_counter; i++)
        {
            fprintf(fp2, "%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n", T1[i], s_plot[i], int(s_plot2[i]), s_plot3[i], s_plot4[i], int(s_plot5[i]), s_plot6[i], int(s_plot7[i]), int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}
