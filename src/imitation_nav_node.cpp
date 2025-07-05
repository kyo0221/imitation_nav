#include "imitation_nav/imitation_nav_node.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>


namespace imitation_nav
{

ImitationNav::ImitationNav(const rclcpp::NodeOptions &options) : ImitationNav("", options) {}

ImitationNav::ImitationNav(const std::string &name_space, const rclcpp::NodeOptions &options)
: rclcpp::Node("imitation_nav_node", name_space, options),
interval_ms(get_parameter("interval_ms").as_int()),
localization_interval_ms(get_parameter("localization_interval_ms").as_int()),
model_name(get_parameter("model_name").as_string()),
linear_max_(get_parameter("max_linear_vel").as_double()),
angular_max_(get_parameter("max_angular_vel").as_double()),
image_width_(get_parameter("image_width").as_int()),
image_height_(get_parameter("image_height").as_int()),
visualize_flag_(get_parameter("visualize_flag").as_bool()),
window_lower_(get_parameter("window_lower").as_int()),
window_upper_(get_parameter("window_upper").as_int()),
use_observation_based_init_(get_parameter("use_observation_based_init").as_bool()),
obstacle_distance_threshold_(get_parameter("obstacle_distance_threshold").as_double()),
topo_localizer_(
    ament_index_cpp::get_package_share_directory("imitation_nav") + "/config/topo_map/topomap.yaml",
    ament_index_cpp::get_package_share_directory("imitation_nav") + "/weights/placenet/placenet.pt",
    ament_index_cpp::get_package_share_directory("imitation_nav") + "/config/topo_map/images/"
)
{
    autonomous_flag_subscriber_ = this->create_subscription<std_msgs::msg::Bool>(
        "/autonomous", 10, std::bind(&ImitationNav::autonomousFlagCallback, this, std::placeholders::_1));

    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/image_raw", 10, std::bind(&ImitationNav::ImageCallback, this, std::placeholders::_1));

    depth_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/depth_anything_v2/depth_image", 10, std::bind(&ImitationNav::DepthImageCallback, this, std::placeholders::_1));

    cmd_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

    timer_ = this->create_wall_timer(std::chrono::milliseconds(interval_ms),
        std::bind(&ImitationNav::ImitationNavigation, this));

    try {
        std::string package_share = ament_index_cpp::get_package_share_directory("imitation_nav");
        std::string model_path_ = package_share + "/weights/" + model_name;

        model_ = torch::jit::load(model_path_);
        model_.to(torch::kCUDA);
        model_.eval();
        RCLCPP_INFO(this->get_logger(), "Model loaded from: %s", model_path_.c_str());
    } catch (const c10::Error &e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to load model: %s", e.what());
    }
}

void ImitationNav::autonomousFlagCallback(const std_msgs::msg::Bool::SharedPtr msg)
{
    autonomous_flag_ = msg->data;
}

void ImitationNav::ImageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    try {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        latest_image_ = cv_ptr->image.clone();

        if(init_flag_ && autonomous_flag_){
            topo_localizer_.initializeModel(latest_image_, use_observation_based_init_);
            topo_localizer_.setTransitionWindow(window_lower_, window_upper_);
            RCLCPP_INFO(this->get_logger(), "initialize model with observation_based_init: %s", 
                       use_observation_based_init_ ? "true" : "false");
            init_flag_=false;
        }
    } catch (const cv_bridge::Exception &e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
}

void ImitationNav::DepthImageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    try {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        latest_depth_image_ = cv_ptr->image.clone();
        
        // Check for obstacles in the path
        obstacle_detected_ = checkObstacleInPath(latest_depth_image_);
        
    } catch (const cv_bridge::Exception &e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception in depth callback: %s", e.what());
    }
}

bool ImitationNav::checkObstacleInPath(const cv::Mat& depth_image)
{
    if (depth_image.empty()) {
        return false;
    }
    
    // Focus on upper half of the image
    int height = depth_image.rows;
    int width = depth_image.cols;
    int upper_half_height = height / 2;
    
    // Create ROI for upper half
    cv::Rect roi(0, 0, width, upper_half_height);
    cv::Mat upper_half = depth_image(roi);
    
    // Convert to grayscale if it's a color image
    cv::Mat gray_image;
    if (upper_half.channels() == 3) {
        cv::cvtColor(upper_half, gray_image, cv::COLOR_BGR2GRAY);
    } else {
        gray_image = upper_half.clone();
    }
    
    // Calculate the number of pixels representing close objects
    // Assuming depth visualization: darker pixels = closer objects, lighter = farther
    // For typical depth colormap, closer objects appear in warmer colors (red/orange)
    // We'll use a simple threshold approach
    
    double threshold_intensity = obstacle_distance_threshold_ * 255.0; // Convert to pixel intensity
    cv::Mat mask;
    cv::threshold(gray_image, mask, threshold_intensity, 255, cv::THRESH_BINARY_INV);
    
    // Count pixels below threshold (close objects)
    int close_pixels = cv::countNonZero(mask);
    int total_pixels = gray_image.rows * gray_image.cols;
    
    // If more than 5% of upper half pixels represent close objects, consider it an obstacle
    double close_pixel_ratio = static_cast<double>(close_pixels) / total_pixels;
    bool obstacle_found = close_pixel_ratio > 0.05;
    
    if (obstacle_found) {
        RCLCPP_WARN(this->get_logger(), "Obstacle detected! Close pixel ratio: %.2f%%", close_pixel_ratio * 100.0);
    }
    
    return obstacle_found;
}

void ImitationNav::ImitationNavigation()
{
    if (!autonomous_flag_ || init_flag_ || latest_image_.empty()) return;

    try {
        int node_id_ = topo_localizer_.inferNode(latest_image_);

        std::string action = topo_localizer_.getNodeAction(node_id_);
        int command_idx = 0;

        RCLCPP_INFO(this->get_logger(), "current node id : %d, action command : %s", node_id_, action.c_str());

        if (action == "roadside") command_idx = 0;
        else if (action == "straight") command_idx = 1;
        else if (action == "left") command_idx = 2;
        else if (action == "right") command_idx = 3;
        else {
            RCLCPP_WARN(this->get_logger(), "Unknown matched action: %s. Defaulting to 'roadside'", action.c_str());
            command_idx = 0;
        }

        cv::Mat resized;
        cv::resize(latest_image_, resized, cv::Size(image_width_, image_height_));
        resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);
        at::Tensor image_tensor = torch::from_blob(resized.data, {1, image_height_, image_width_, 3})
            .permute({0, 3, 1, 2})
            .clone()
            .to(torch::kCUDA);

        at::Tensor cmd_tensor = torch::zeros({1, 4}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        cmd_tensor[0][command_idx] = 1.0;

        at::Tensor output = model_.forward({image_tensor, cmd_tensor}).toTensor();

        double predicted_angular = output.item<float>();
        predicted_angular = std::clamp(predicted_angular, -angular_max_, angular_max_);

        geometry_msgs::msg::Twist cmd_msg;
        // Set linear velocity to 0 if obstacle is detected, otherwise use max linear velocity
        cmd_msg.linear.x = obstacle_detected_ ? 0.0 : linear_max_;
        cmd_msg.angular.z = predicted_angular;
        cmd_pub_->publish(cmd_msg);
        
        if (obstacle_detected_) {
            RCLCPP_WARN(this->get_logger(), "Obstacle detected! Setting linear velocity to 0");
        }

    } catch (const c10::Error &e) {
        RCLCPP_ERROR(this->get_logger(), "TorchScript inference error: %s", e.what());
    }
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