#include "imitation_nav/pointcloud_processor.hpp"

#include <cmath>
#include <limits>

namespace imitation_nav
{

PointCloudProcessor::PointCloudProcessor(
    double z_min,
    double z_max,
    double collision_zone_stop,
    double collision_zone_slow2,
    double collision_zone_slow1,
    double collision_gain_stop,
    double collision_gain_slow2,
    double collision_gain_slow1,
    double collision_y_width,
    double angle_increment_deg,
    double range_max)
    : z_min_(z_min),
      z_max_(z_max),
      collision_zone_stop_(collision_zone_stop),
      collision_zone_slow2_(collision_zone_slow2),
      collision_zone_slow1_(collision_zone_slow1),
      collision_gain_stop_(collision_gain_stop),
      collision_gain_slow2_(collision_gain_slow2),
      collision_gain_slow1_(collision_gain_slow1),
      collision_y_width_(collision_y_width),
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

    // 高さフィルタリング（z軸）- 横幅制限は削除
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

double PointCloudProcessor::calculateCollisionGain(const sensor_msgs::msg::LaserScan& scan)
{
    double min_distance = std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < scan.ranges.size(); ++i)
    {
        if (!std::isfinite(scan.ranges[i])) {
            continue;
        }

        double angle = scan.angle_min + i * scan.angle_increment;
        double range = scan.ranges[i];

        // 極座標からロボット座標系のx, y座標に変換
        double x = range * std::cos(angle);
        double y = range * std::sin(angle);

        // y軸（横）の判定範囲チェック: ±0.35m (合計0.7m)
        if (std::abs(y) > collision_y_width_ / 2.0) {
            continue;
        }

        // x軸（前方）の範囲チェック: 0 < x < collision_zone_slow1_
        if (x > 0.0 && x < collision_zone_slow1_) {
            min_distance = std::min(min_distance, x);
        }
    }

    // 障害物が検出されなければゲイン1.0（制限なし）
    if (!std::isfinite(min_distance)) {
        return 1.0;
    }

    // 距離に応じたゲインを計算
    if (min_distance <= collision_zone_stop_) {
        return collision_gain_stop_;  // 1m以内: gain 0.0
    }
    else if (min_distance <= collision_zone_slow2_) {
        return collision_gain_slow2_;  // 1-2m: gain 0.2
    }
    else if (min_distance <= collision_zone_slow1_) {
        return collision_gain_slow1_;  // 2-3m: gain 0.5
    }
    else {
        return 1.0;  // 3m以上: gain 1.0（制限なし）
    }
}

}  // namespace imitation_nav
