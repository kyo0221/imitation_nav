#include "imitation_nav/imitation_nav_node.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>


namespace imitation_nav
{

ImitationNav::ImitationNav(const rclcpp::NodeOptions &options) : ImitationNav("", options) {}

ImitationNav::ImitationNav(const std::string &name_space, const rclcpp::NodeOptions &options)
: rclcpp::Node("imitation_nav_node", name_space, options),
interval_ms(get_parameter("interval_ms").as_int()),
model_name(get_parameter("model_name").as_string()),
linear_max_(get_parameter("max_linear_vel").as_double()),
angular_max_(get_parameter("max_angular_vel").as_double()),
image_width_(get_parameter("image_width").as_int()),
image_height_(get_parameter("image_height").as_int())
{
    autonomous_flag_subscriber_ = this->create_subscription<std_msgs::msg::Bool>(
        "/autonomous", 10, std::bind(&ImitationNav::autonomousFlagCallback, this, std::placeholders::_1));

    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/image_raw", 10, std::bind(&ImitationNav::ImageCallback, this, std::placeholders::_1));

    cmd_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

    timer_ = this->create_wall_timer(std::chrono::milliseconds(interval_ms),
        std::bind(&ImitationNav::ImitationNavigation, this));

    std::string package_share = ament_index_cpp::get_package_share_directory("imitation_nav");
    std::string model_path_ = package_share + "/weights/" + model_name;

    model_ = torch::jit::load(model_path_);
    model_.to(torch::kCUDA);
    model_.eval();
    RCLCPP_INFO(this->get_logger(), "Model loaded from: %s", model_path_.c_str());
}

void ImitationNav::autonomousFlagCallback(const std_msgs::msg::Bool::SharedPtr msg)
{
    autonomous_flag_ = msg->data;
}

void ImitationNav::ImageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    latest_image_ = cv_ptr->image.clone();
}


void ImitationNav::ImitationNavigation()
{
    if (!autonomous_flag_ || latest_image_.empty()) return;

    cv::Mat resized;
    cv::resize(latest_image_, resized, cv::Size(image_width_, image_height_));
    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);
    at::Tensor image_tensor = torch::from_blob(resized.data, {1, image_height_, image_width_, 3})
        .permute({0, 3, 1, 2})
        .clone()
        .to(torch::kCUDA);

    at::Tensor output = model_.forward({image_tensor}).toTensor();

    double predicted_angular = output.item<float>();
    predicted_angular = std::clamp(predicted_angular, -angular_max_, angular_max_);

    geometry_msgs::msg::Twist cmd_msg;
    cmd_msg.linear.x = linear_max_;
    cmd_msg.angular.z = predicted_angular;
    cmd_pub_->publish(cmd_msg);
}

}  // namespace imitation_nav

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions node_option;
    node_option.allow_undeclared_parameters(true);
    node_option.automatically_declare_parameters_from_overrides(true);

    auto node = std::make_shared<imitation_nav::ImitationNav>(node_option);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}