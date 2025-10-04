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

LaserScanWithHeightStats PointCloudProcessor::convertToLaserScanWithStats(
    const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg)
{
    LaserScanWithHeightStats result;

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

    result.scan.header = cloud_msg->header;

    // 角度範囲を設定（-180度から180度）
    double angle_min_rad = -M_PI;
    double angle_max_rad = M_PI;
    double angle_increment_rad = angle_increment_deg_ * M_PI / 180.0;

    int num_readings = static_cast<int>((angle_max_rad - angle_min_rad) / angle_increment_rad);

    result.scan.angle_min = angle_min_rad;
    result.scan.angle_max = angle_max_rad;
    result.scan.angle_increment = angle_increment_rad;
    result.scan.time_increment = 0.0;
    result.scan.scan_time = 0.0;
    result.scan.range_min = 0.0;
    result.scan.range_max = range_max_;

    result.scan.ranges.resize(num_readings, std::numeric_limits<float>::infinity());
    result.scan.intensities.resize(num_readings, 0.0);

    // 高さ統計用の初期化
    result.z_min.resize(num_readings, std::numeric_limits<double>::infinity());
    result.z_max.resize(num_readings, -std::numeric_limits<double>::infinity());
    result.z_variance.resize(num_readings, 0.0);
    result.point_count.resize(num_readings, 0);

    // 各ビンのz値を蓄積
    std::vector<std::vector<double>> z_values_per_bin(num_readings);

    // 点群をLaserScanに変換
    for (const auto& point : cloud_filtered->points)
    {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
            continue;
        }

        // 極座標に変換
        double range = std::sqrt(point.x * point.x + point.y * point.y);
        double angle = std::atan2(point.y, point.x);

        // 範囲チェック
        if (range < result.scan.range_min || range > result.scan.range_max) {
            continue;
        }

        // 角度インデックスを計算
        int index = static_cast<int>((angle - angle_min_rad) / angle_increment_rad);

        if (index >= 0 && index < num_readings) {
            // 最小距離を保持
            if (range < result.scan.ranges[index]) {
                result.scan.ranges[index] = range;
            }

            // 高さ情報を蓄積
            z_values_per_bin[index].push_back(point.z);

            // 最小・最大高さ更新
            result.z_min[index] = std::min(result.z_min[index], static_cast<double>(point.z));
            result.z_max[index] = std::max(result.z_max[index], static_cast<double>(point.z));
            result.point_count[index]++;
        }
    }

    // 各ビンの高さ分散を計算
    for (int i = 0; i < num_readings; ++i) {
        if (z_values_per_bin[i].empty()) {
            result.z_min[i] = 0.0;
            result.z_max[i] = 0.0;
            result.z_variance[i] = 0.0;
            continue;
        }

        // 平均を計算
        double sum = 0.0;
        for (double z : z_values_per_bin[i]) {
            sum += z;
        }
        double mean = sum / z_values_per_bin[i].size();

        // 分散を計算
        double sq_sum = 0.0;
        for (double z : z_values_per_bin[i]) {
            sq_sum += (z - mean) * (z - mean);
        }
        result.z_variance[i] = sq_sum / z_values_per_bin[i].size();
    }

    return result;
}

}  // namespace imitation_nav
