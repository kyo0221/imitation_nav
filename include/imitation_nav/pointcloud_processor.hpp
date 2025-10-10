#pragma once

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>

namespace imitation_nav
{

class PointCloudProcessor
{
public:
  PointCloudProcessor(
    double z_min,
    double z_max,
    double angle_min_deg,
    double angle_max_deg,
    double obstacle_distance_threshold,
    double angle_increment_deg = 1.0,
    double range_max = 10.0
  );

  // PointCloud2をLaserScanに変換
  sensor_msgs::msg::LaserScan convertToLaserScan(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg);

  // 障害物検出（指定範囲内の最小距離が閾値以下かチェック）
  bool detectObstacle(const sensor_msgs::msg::LaserScan& scan);

  // 前方の最小距離を計算（collision monitor用）
  double computeMinFrontDistance(const sensor_msgs::msg::LaserScan& scan);

  // 距離に基づく速度スケーリング係数を計算（0.0～1.0）
  static double computeVelocityScale(double min_distance);

private:
  double z_min_;
  double z_max_;
  double angle_min_deg_;
  double angle_max_deg_;
  double obstacle_distance_threshold_;
  double angle_increment_deg_;
  double range_max_;
};

}  // namespace imitation_nav
