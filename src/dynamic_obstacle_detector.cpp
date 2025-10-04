#include "imitation_nav/dynamic_obstacle_detector.hpp"

namespace imitation_nav
{

DynamicObstacleDetector::DynamicObstacleDetector(
  double v_threshold,
  double rigid_std_threshold,
  double veg_z_var_threshold,
  int min_cluster_bins,
  double stop_duration,
  double wait_duration)
  : v_threshold_(v_threshold),
    rigid_std_threshold_(rigid_std_threshold),
    veg_z_var_threshold_(veg_z_var_threshold),
    min_cluster_bins_(min_cluster_bins),
    stop_duration_(stop_duration),
    wait_duration_(wait_duration)
{
}

void DynamicObstacleDetector::recordScan(const LaserScanWithHeightStats& scan)
{
  scan_buffer_.push_back(scan);

  if (scan_buffer_.size() > 100) {
    scan_buffer_.pop_front();
  }
}

bool DynamicObstacleDetector::isDynamic()
{
  if (scan_buffer_.size() < 2) {
    return false; // データ不足
  }

  // 最初と最後のスキャンで速度推定
  auto& first = scan_buffer_.front();
  auto& last = scan_buffer_.back();

  // 実際の時間差を計算（stop_duration_を近似として使用）
  double dt = stop_duration_;

  auto velocities = computeVelocity(first.scan, last.scan, dt);

  // クラスタ化
  auto clusters = clusterMovingBins(velocities, last.z_variance, last.scan);

  // いずれかのクラスタが剛体動的なら true
  for (const auto& cluster : clusters) {
    if (isRigidDynamic(cluster)) {
      return true;
    }
  }

  return false;
}

void DynamicObstacleDetector::transitionToStopped()
{
  state_ = State::STOPPED;
  scan_buffer_.clear(); // バッファクリア
}

void DynamicObstacleDetector::transitionToWaiting()
{
  state_ = State::WAITING;
  scan_buffer_.clear();
}

void DynamicObstacleDetector::transitionToNormal()
{
  state_ = State::NORMAL;
  scan_buffer_.clear();
}

std::vector<double> DynamicObstacleDetector::computeVelocity(
  const sensor_msgs::msg::LaserScan& scan1,
  const sensor_msgs::msg::LaserScan& scan2,
  double dt)
{
  std::vector<double> velocities(scan1.ranges.size(), 0.0);

  for (size_t i = 0; i < scan1.ranges.size(); ++i) {
    // 両方有効なレンジのみ
    if (!std::isfinite(scan1.ranges[i]) || !std::isfinite(scan2.ranges[i])) {
      continue;
    }

    // 径方向速度（正=遠ざかる、負=近づく）
    double delta_r = scan2.ranges[i] - scan1.ranges[i];
    velocities[i] = delta_r / dt;
  }

  return velocities;
}

std::vector<DynamicObstacleDetector::ObstacleCluster>
DynamicObstacleDetector::clusterMovingBins(
  const std::vector<double>& velocities,
  const std::vector<double>& z_variances,
  const sensor_msgs::msg::LaserScan& scan)
{
  std::vector<ObstacleCluster> clusters;

  int cluster_start = -1;
  std::vector<double> cluster_velocities;
  std::vector<double> cluster_z_vars;
  std::vector<double> cluster_distances;

  for (size_t i = 0; i < velocities.size(); ++i) {
    // 動的ビンの判定
    if (std::abs(velocities[i]) > v_threshold_ && std::isfinite(scan.ranges[i])) {
      if (cluster_start == -1) {
        // 新規クラスタ開始
        cluster_start = i;
      }
      cluster_velocities.push_back(velocities[i]);
      if (i < z_variances.size()) {
        cluster_z_vars.push_back(z_variances[i]);
      }
      cluster_distances.push_back(scan.ranges[i]);
    } else {
      // クラスタ終了判定
      if (cluster_start != -1 &&
          static_cast<int>(cluster_velocities.size()) >= min_cluster_bins_) {
        ObstacleCluster cluster;
        cluster.bin_start = cluster_start;
        cluster.bin_end = i - 1;

        // 平均速度
        cluster.mean_velocity = std::accumulate(
          cluster_velocities.begin(),
          cluster_velocities.end(),
          0.0) / cluster_velocities.size();

        // 速度分散
        double mean_v = cluster.mean_velocity;
        double sq_sum = std::accumulate(
          cluster_velocities.begin(),
          cluster_velocities.end(),
          0.0,
          [mean_v](double acc, double v) { return acc + (v - mean_v) * (v - mean_v); }
        );
        cluster.std_velocity = std::sqrt(sq_sum / cluster_velocities.size());

        // 高さ分散平均
        if (!cluster_z_vars.empty()) {
          cluster.mean_z_variance = std::accumulate(
            cluster_z_vars.begin(),
            cluster_z_vars.end(),
            0.0) / cluster_z_vars.size();
        } else {
          cluster.mean_z_variance = 0.0;
        }

        // 距離（最小距離）
        cluster.distance = *std::min_element(
          cluster_distances.begin(),
          cluster_distances.end()
        );

        clusters.push_back(cluster);
      }

      // リセット
      cluster_start = -1;
      cluster_velocities.clear();
      cluster_z_vars.clear();
      cluster_distances.clear();
    }
  }

  // 最後のクラスタ処理
  if (cluster_start != -1 &&
      static_cast<int>(cluster_velocities.size()) >= min_cluster_bins_) {
    ObstacleCluster cluster;
    cluster.bin_start = cluster_start;
    cluster.bin_end = velocities.size() - 1;

    cluster.mean_velocity = std::accumulate(
      cluster_velocities.begin(),
      cluster_velocities.end(),
      0.0) / cluster_velocities.size();

    double mean_v = cluster.mean_velocity;
    double sq_sum = std::accumulate(
      cluster_velocities.begin(),
      cluster_velocities.end(),
      0.0,
      [mean_v](double acc, double v) { return acc + (v - mean_v) * (v - mean_v); }
    );
    cluster.std_velocity = std::sqrt(sq_sum / cluster_velocities.size());

    if (!cluster_z_vars.empty()) {
      cluster.mean_z_variance = std::accumulate(
        cluster_z_vars.begin(),
        cluster_z_vars.end(),
        0.0) / cluster_z_vars.size();
    } else {
      cluster.mean_z_variance = 0.0;
    }

    cluster.distance = *std::min_element(
      cluster_distances.begin(),
      cluster_distances.end()
    );

    clusters.push_back(cluster);
  }

  return clusters;
}

bool DynamicObstacleDetector::isRigidDynamic(const ObstacleCluster& cluster)
{
  // 剛体動的（人）の条件
  bool high_mean_velocity = std::abs(cluster.mean_velocity) > v_threshold_;
  bool low_velocity_std = cluster.std_velocity < rigid_std_threshold_;

  // 植生除外：高さがバラバラ & 速度分散大
  bool is_vegetation =
    cluster.mean_z_variance > veg_z_var_threshold_ &&
    cluster.std_velocity > rigid_std_threshold_;

  return high_mean_velocity && low_velocity_std && !is_vegetation;
}

}  // namespace imitation_nav
