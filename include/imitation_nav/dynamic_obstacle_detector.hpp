#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace imitation_nav
{

// LaserScanに高さ統計を追加した構造体
struct LaserScanWithHeightStats
{
  sensor_msgs::msg::LaserScan scan;
  std::vector<double> z_min;      // 各ビンの最低高さ
  std::vector<double> z_max;      // 各ビンの最高高さ
  std::vector<double> z_variance; // 各ビンの高さ分散
  std::vector<int> point_count;   // 各ビンの点数
};

class DynamicObstacleDetector
{
public:
  enum class State {
    NORMAL,      // 通常走行
    STOPPED,     // 停止中（観測期間）
    WAITING      // 動的障害物待機中
  };

  struct ObstacleCluster
  {
    int bin_start;
    int bin_end;
    double mean_velocity;      // 径方向平均速度
    double std_velocity;       // 速度分散
    double mean_z_variance;    // 高さ分散平均
    double distance;           // 障害物距離
  };

  // コンストラクタ
  DynamicObstacleDetector(
    double v_threshold = 0.15,           // 動的判定速度閾値 [m/s]
    double rigid_std_threshold = 0.10,   // 剛体判定（速度分散上限）
    double veg_z_var_threshold = 0.12,   // 植生判定（高さ分散下限）
    int min_cluster_bins = 3,
    double stop_duration = 10.0,
    double wait_duration = 10.0
  );

  // 停止中の観測を記録
  void recordScan(const LaserScanWithHeightStats& scan);

  // 動的判定実行（停止期間終了時に呼ぶ）
  bool isDynamic();

  // 状態管理
  State getState() const { return state_; }
  void transitionToStopped();
  void transitionToWaiting();
  void transitionToNormal();

  // パラメータ取得
  double getStopDuration() const { return stop_duration_; }
  double getWaitDuration() const { return wait_duration_; }

private:
  // 速度推定（静止ロボット前提）
  std::vector<double> computeVelocity(
    const sensor_msgs::msg::LaserScan& scan1,
    const sensor_msgs::msg::LaserScan& scan2,
    double dt
  );

  // クラスタ化
  std::vector<ObstacleCluster> clusterMovingBins(
    const std::vector<double>& velocities,
    const std::vector<double>& z_variances,
    const sensor_msgs::msg::LaserScan& scan
  );

  // 剛体動的判定
  bool isRigidDynamic(const ObstacleCluster& cluster);

  State state_ = State::NORMAL;
  std::deque<LaserScanWithHeightStats> scan_buffer_; // 停止中の観測バッファ

  // パラメータ
  double v_threshold_;
  double rigid_std_threshold_;
  double veg_z_var_threshold_;
  int min_cluster_bins_;
  double stop_duration_;
  double wait_duration_;
};

}  // namespace imitation_nav
