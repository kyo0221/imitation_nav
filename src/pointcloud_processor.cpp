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

double PointCloudProcessor::getObstacleYBias(const sensor_msgs::msg::LaserScan& scan)
{
    double y_sum = 0.0;
    int count = 0;

    // 停止ゾーン内の障害物のy座標を集計
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

        // 停止ゾーン内（0 < x < collision_zone_stop_）の点群のみ対象
        if (x > 0.0 && x < collision_zone_stop_) {
            y_sum += y;
            count++;
        }
    }

    // 停止ゾーン内に障害物がない場合は0.0を返す
    if (count == 0) {
        return 0.0;
    }

    // y座標の平均値を返す（正：左側に障害物、負：右側に障害物）
    return y_sum / count;
}

void PointCloudProcessor::visualizeCollisionMonitor(const sensor_msgs::msg::LaserScan& scan, double current_gain)
{
    // 画像サイズ: x軸-1~9m (10m), y軸3~-3m (6m), 100 pixels/m
    const int img_height = 1000;  // x軸方向
    const int img_width = 600;    // y軸方向
    const double scale = 100.0;   // pixels per meter
    const double x_min = -1.0;
    const double x_max = 9.0;
    const double y_min = -3.0;
    const double y_max = 3.0;

    // 背景画像（黒）
    cv::Mat img = cv::Mat::zeros(img_height, img_width, CV_8UC3);

    // ロボット座標(x, y)からOpenCV座標(col, row)への変換
    auto toImageCoords = [&](double x, double y) -> cv::Point {
        int col = static_cast<int>((y_max - y) * scale);
        int row = static_cast<int>((x_max - x) * scale);
        return cv::Point(col, row);
    };

    // グリッド線を描画（1mごと）
    cv::Scalar grid_color(50, 50, 50);
    for (int x = static_cast<int>(x_min); x <= static_cast<int>(x_max); ++x) {
        cv::Point p1 = toImageCoords(x, y_max);
        cv::Point p2 = toImageCoords(x, y_min);
        cv::line(img, p1, p2, grid_color, 1);
    }
    for (int y = static_cast<int>(y_min); y <= static_cast<int>(y_max); ++y) {
        cv::Point p1 = toImageCoords(x_min, y);
        cv::Point p2 = toImageCoords(x_max, y);
        cv::line(img, p1, p2, grid_color, 1);
    }

    // collision zoneエリアを描画（半透明矩形）
    cv::Mat overlay = img.clone();

    double y_half_width = collision_y_width_ / 2.0;

    // slow1エリア (2-3m): 黄緑色 (BGR: 0, 255, 0)
    if (collision_zone_slow1_ > collision_zone_slow2_) {
        cv::Point p1 = toImageCoords(collision_zone_slow1_, y_half_width);
        cv::Point p2 = toImageCoords(collision_zone_slow2_, -y_half_width);
        cv::rectangle(overlay, p1, p2, cv::Scalar(0, 255, 0), -1);
    }

    // slow2エリア (1-2m): 黄色 (BGR: 0, 255, 255)
    if (collision_zone_slow2_ > collision_zone_stop_) {
        cv::Point p1 = toImageCoords(collision_zone_slow2_, y_half_width);
        cv::Point p2 = toImageCoords(collision_zone_stop_, -y_half_width);
        cv::rectangle(overlay, p1, p2, cv::Scalar(0, 255, 255), -1);
    }

    // stopエリア (0-1m): 赤色 (BGR: 0, 0, 255)
    if (collision_zone_stop_ > 0.0) {
        cv::Point p1 = toImageCoords(collision_zone_stop_, y_half_width);
        cv::Point p2 = toImageCoords(0.0, -y_half_width);
        cv::rectangle(overlay, p1, p2, cv::Scalar(0, 0, 255), -1);
    }

    // エリアに透明度を適用（反応時: 0.6、非反応時: 0.3）
    double alpha = (current_gain < 1.0) ? 0.6 : 0.3;
    cv::addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img);

    // 点群を描画（赤い点）
    for (size_t i = 0; i < scan.ranges.size(); ++i) {
        if (!std::isfinite(scan.ranges[i])) {
            continue;
        }

        double angle = scan.angle_min + i * scan.angle_increment;
        double range = scan.ranges[i];

        // 極座標からロボット座標系のx, y座標に変換
        double x = range * std::cos(angle);
        double y = range * std::sin(angle);

        // 表示範囲内かチェック
        if (x < x_min || x > x_max || y < y_min || y > y_max) {
            continue;
        }

        cv::Point pt = toImageCoords(x, y);
        cv::circle(img, pt, 2, cv::Scalar(0, 0, 255), -1);  // 赤色の点
    }

    // ロボットを描画（青い円）
    cv::Point robot_pos = toImageCoords(0.0, 0.0);
    cv::circle(img, robot_pos, 10, cv::Scalar(255, 0, 0), -1);  // 青色の円
    cv::circle(img, robot_pos, 10, cv::Scalar(255, 255, 255), 2);  // 白い縁

    // gainの情報を表示
    std::stringstream ss;
    ss << "Collision Gain: " << std::fixed << std::setprecision(2) << current_gain;
    cv::putText(img, ss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    // 画像を表示
    cv::imshow("Collision Monitor", img);
    cv::waitKey(1);
}

}  // namespace imitation_nav
