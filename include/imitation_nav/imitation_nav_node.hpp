#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>

#include <imitation_nav/topo_localizer.hpp>
#include <imitation_nav/pointcloud_processor.hpp>

#include <imitation_nav/visibility_control.h>

namespace imitation_nav
{

class ImitationNav : public rclcpp::Node
{
public:
  IMITATION_NAV_PUBLIC
  explicit ImitationNav(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

  IMITATION_NAV_PUBLIC
  explicit ImitationNav(const std::string& name_space, const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

private:
  enum class NavigationState {
    NORMAL,
    STOPPED,
    RECOVERY
  };

  void autonomousFlagCallback(const std_msgs::msg::Bool::SharedPtr msg);
  void ImageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

  void ImitationNavigation();

  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr autonomous_flag_subscriber_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr laserscan_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr autonomous_flag_pub_;

  rclcpp::TimerBase::SharedPtr timer_;

  torch::jit::script::Module model_;

  cv::Mat latest_image_;
  sensor_msgs::msg::LaserScan latest_scan_;

  const int interval_ms;
  const int localization_interval_ms;
  const std::string model_name;
  const double linear_max_;
  const double angular_max_;
  const bool visualize_flag_;
  const int window_lower_;
  const int window_upper_;
  const bool use_observation_based_init_;
  const bool enable_stop_deceleration_;

  bool autonomous_flag_=false;
  bool init_flag_=true;
  double collision_gain_=1.0;

  NavigationState nav_state_ = NavigationState::NORMAL;
  rclcpp::Time stopped_time_;
  double recovery_timeout_;
  double recovery_angular_gain_;
  double recovery_direction_ = 0.0;  // リカバリー時の回転方向を保存

  imitation_nav::TopoLocalizer topo_localizer_;
  std::shared_ptr<imitation_nav::PointCloudProcessor> pointcloud_processor_;

  // stop履歴管理
  std::set<int> stopped_node_ids_;

  // stopアクション付きノードのIDリスト
  std::vector<int> stop_node_ids_;

  // 前後10ID範囲にstopがあるかチェック
  bool isNearStoppedNode(int current_node_id) const;

  // stopノードの10個手前にいるかチェック
  bool isApproachingStopNode(int current_node_id) const;
};

}  // namespace imitation_nav