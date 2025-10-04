#include "imitation_nav/pointcloud_processor.hpp"

#include <cmath>
#include <limits>

namespace imitation_nav
{

PointCloudProcessor::PointCloudProcessor(
    double z_min,
    double z_max,
    double angle_min_deg,
    double angle_max_deg,
    double obstacle_distance_threshold,
    double angle_increment_deg,
    double range_max)
    : z_min_(z_min),
      z_max_(z_max),
      angle_min_deg_(angle_min_deg),
      angle_max_deg_(angle_max_deg),
      obstacle_distance_threshold_(obstacle_distance_threshold),
      angle_increment_deg_(angle_increment_deg),
      range_max_(range_max)
{
}

sensor_msgs::msg::LaserScan PointCloudProcessor::convertToLaserScan(
    const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg)
{
    // PointCloud2をPCLに変換
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    // 高さフィルタリング（z軸）
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(z_min_, z_max_);
    pass.filter(*cloud_filtered);

    sensor_msgs::msg::LaserScan scan_msg;
    scan_msg.header = cloud_msg->header;

    // 角度範囲を設定（-180度から180度）
    double angle_min_rad = -M_PI;
    double angle_max_rad = M_PI;
    double angle_increment_rad = angle_increment_deg_ * M_PI / 180.0;

    int num_readings = static_cast<int>((angle_max_rad - angle_min_rad) / angle_increment_rad);

    scan_msg.angle_min = angle_min_rad;
    scan_msg.angle_max = angle_max_rad;
    scan_msg.angle_increment = angle_increment_rad;
    scan_msg.time_increment = 0.0;
    scan_msg.scan_time = 0.0;
    scan_msg.range_min = 0.0;
    scan_msg.range_max = range_max_;

    scan_msg.ranges.resize(num_readings, std::numeric_limits<float>::infinity());
    scan_msg.intensities.resize(num_readings, 0.0);

    // 点群をLaserScanに変換
    for (const auto& point : cloud_filtered->points)
    {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
            continue;
        }

        // 極座標に変換
        double range = std::sqrt(point.x * point.x + point.y * point.y);
        double angle = std::atan2(point.y, point.x);

        // 範囲チェック
        if (range < scan_msg.range_min || range > scan_msg.range_max) {
            continue;
        }

        // 角度インデックスを計算
        int index = static_cast<int>((angle - angle_min_rad) / angle_increment_rad);

        if (index >= 0 && index < num_readings) {
            // 最小距離を保持
            if (range < scan_msg.ranges[index]) {
                scan_msg.ranges[index] = range;
            }
        }
    }

    return scan_msg;
}

bool PointCloudProcessor::detectObstacle(const sensor_msgs::msg::LaserScan& scan)
{
    double angle_min_check_rad = angle_min_deg_ * M_PI / 180.0;
    double angle_max_check_rad = angle_max_deg_ * M_PI / 180.0;

    for (size_t i = 0; i < scan.ranges.size(); ++i)
    {
        double angle = scan.angle_min + i * scan.angle_increment;

        // 指定角度範囲内かチェック
        if (angle >= angle_min_check_rad && angle <= angle_max_check_rad)
        {
            if (std::isfinite(scan.ranges[i]) && scan.ranges[i] < obstacle_distance_threshold_) {
                return true;
            }
        }
    }

    return false;
}

}  // namespace imitation_nav
