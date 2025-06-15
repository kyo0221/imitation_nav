#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>

#include <imitation_nav/topo_localizer.hpp>

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
  void autonomousFlagCallback(const std_msgs::msg::Bool::SharedPtr msg);
  void RouteCommandCallback(const std_msgs::msg::String::SharedPtr msg);
  void ImageCallback(const sensor_msgs::msg::Image::SharedPtr msg);

  void ImitationNavigation();

  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr autonomous_flag_subscriber_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr route_command_subscriber_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;

  rclcpp::TimerBase::SharedPtr timer_;

  torch::jit::script::Module model_;

  cv::Mat latest_image_;

  const int interval_ms;
  const int localization_interval_ms;
  const std::string model_name;
  const double linear_max_;
  const double angular_max_;
  const int image_width_;
  const int image_height_;
  const bool visualize_flag_;
  const int window_lower_;
  const int window_upper_;

  bool autonomous_flag_=false;
  bool init_flag_=true;
  
  imitation_nav::TopoLocalizer topo_localizer_;
};

}  // namespace imitation_nav