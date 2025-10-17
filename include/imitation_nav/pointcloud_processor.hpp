#pragma once

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <opencv2/opencv.hpp>

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
    double collision_zone_stop,
    double collision_zone_slow2,
    double collision_zone_slow1,
    double collision_gain_stop,
    double collision_gain_slow2,
    double collision_gain_slow1,
    double collision_y_width,
    double angle_increment_deg = 1.0,
    double range_max = 10.0
  );

  // PointCloud2をLaserScanに変換
  sensor_msgs::msg::LaserScan convertToLaserScan(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg);

  // collision monitorによる速度ゲイン計算
  double calculateCollisionGain(const sensor_msgs::msg::LaserScan& scan);

  // collision monitor可視化
  void visualizeCollisionMonitor(const sensor_msgs::msg::LaserScan& scan, double current_gain);

private:
  double z_min_;
  double z_max_;
  double collision_zone_stop_;
  double collision_zone_slow2_;
  double collision_zone_slow1_;
  double collision_gain_stop_;
  double collision_gain_slow2_;
  double collision_gain_slow1_;
  double collision_y_width_;
  double angle_increment_deg_;
  double range_max_;
};

}  // namespace imitation_nav
