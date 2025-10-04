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
visualize_flag_(get_parameter("visualize_flag").as_bool()),
window_lower_(get_parameter("window_lower").as_int()),
window_upper_(get_parameter("window_upper").as_int()),
use_observation_based_init_(get_parameter("use_observation_based_init").as_bool()),
topo_localizer_(
    ament_index_cpp::get_package_share_directory("imitation_nav") + "/config/topo_map/topomap.yaml",
    ament_index_cpp::get_package_share_directory("imitation_nav") + "/weights/placenet/placenet.pt",
    ament_index_cpp::get_package_share_directory("imitation_nav") + "/config/topo_map/images/"
)
{
    autonomous_flag_subscriber_ = this->create_subscription<std_msgs::msg::Bool>(
        "/autonomous", 10, std::bind(&ImitationNav::autonomousFlagCallback, this, std::placeholders::_1));

    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/zed/zed_node/left/image_rect_color", 10, std::bind(&ImitationNav::ImageCallback, this, std::placeholders::_1));

    pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/zed/zed_node/point_cloud/cloud_registered", 10, std::bind(&ImitationNav::pointCloudCallback, this, std::placeholders::_1));

    cmd_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
    laserscan_pub_ = this->create_publisher<sensor_msgs::msg::LaserScan>("/scan", 10);

    timer_ = this->create_wall_timer(std::chrono::milliseconds(interval_ms),
        std::bind(&ImitationNav::ImitationNavigation, this));

    // PointCloudProcessorを初期化
    double z_min = this->declare_parameter("z_min", -0.5);
    double z_max = this->declare_parameter("z_max", 0.5);
    double angle_min_deg = this->declare_parameter("angle_min_deg", -7.5);
    double angle_max_deg = this->declare_parameter("angle_max_deg", 7.5);
    double obstacle_distance_threshold = this->declare_parameter("obstacle_distance_threshold", 2.0);
    double angle_increment_deg = this->declare_parameter("angle_increment_deg", 1.0);
    double range_max = this->declare_parameter("range_max", 10.0);

    pointcloud_processor_ = std::make_shared<imitation_nav::PointCloudProcessor>(
        z_min, z_max, angle_min_deg, angle_max_deg,
        obstacle_distance_threshold, angle_increment_deg, range_max
    );

    // DynamicObstacleDetectorを初期化
    double v_threshold = this->declare_parameter("v_threshold", 0.15);
    double rigid_std_threshold = this->declare_parameter("rigid_std_threshold", 0.10);
    double veg_z_var_threshold = this->declare_parameter("veg_z_var_threshold", 0.12);
    int min_cluster_bins = this->declare_parameter("min_cluster_bins", 3);
    double stop_duration = this->declare_parameter("stop_duration", 10.0);
    double wait_duration = this->declare_parameter("wait_duration", 10.0);

    detector_ = std::make_shared<imitation_nav::DynamicObstacleDetector>(
        v_threshold, rigid_std_threshold, veg_z_var_threshold,
        min_cluster_bins, stop_duration, wait_duration
    );

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
        cv::Mat rgb_image;
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
        cv::cvtColor(cv_ptr->image, rgb_image, cv::COLOR_BGRA2RGB);
        latest_image_ = rgb_image.clone();

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

void ImitationNav::pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    // 高さ統計付きLaserScan生成
    auto scan_with_stats = pointcloud_processor_->convertToLaserScanWithStats(msg);

    // LaserScanをパブリッシュ
    laserscan_pub_->publish(scan_with_stats.scan);

    // 障害物検出（基本的な距離チェック）
    bool obstacle = pointcloud_processor_->detectObstacle(scan_with_stats.scan);

    auto state = detector_->getState();

    switch (state) {
        case DynamicObstacleDetector::State::NORMAL:
            if (obstacle) {
                // 障害物検出 → 停止開始
                detector_->transitionToStopped();
                stop_start_time_ = this->now();
                RCLCPP_WARN(this->get_logger(), "Obstacle detected. Starting %.1fs observation.",
                           detector_->getStopDuration());
            }
            obstacle_detected_ = obstacle;
            break;

        case DynamicObstacleDetector::State::STOPPED: {
            // 停止中の観測を記録
            detector_->recordScan(scan_with_stats);

            auto elapsed = (this->now() - stop_start_time_).seconds();
            if (elapsed >= detector_->getStopDuration()) {
                // 停止期間経過 → 動的判定
                bool is_dynamic = detector_->isDynamic();

                if (is_dynamic) {
                    // 動的障害物 → 待機状態へ
                    detector_->transitionToWaiting();
                    wait_start_time_ = this->now();
                    RCLCPP_INFO(this->get_logger(), "Dynamic obstacle detected. Waiting %.1fs...",
                               detector_->getWaitDuration());
                } else {
                    // 静的障害物 → 走行再開
                    detector_->transitionToNormal();
                    obstacle_detected_ = false;
                    RCLCPP_INFO(this->get_logger(), "Static obstacle detected. Resuming navigation.");
                }
            }
            obstacle_detected_ = true; // 停止維持
            break;
        }

        case DynamicObstacleDetector::State::WAITING: {
            auto elapsed = (this->now() - wait_start_time_).seconds();
            if (elapsed >= detector_->getWaitDuration()) {
                // 待機期間経過 → 再度停止状態で判定
                detector_->transitionToStopped();
                stop_start_time_ = this->now();
                RCLCPP_INFO(this->get_logger(), "Re-checking obstacle status...");
            }
            obstacle_detected_ = true; // 停止維持
            break;
        }
    }
}

void ImitationNav::ImitationNavigation()
{
    if (!autonomous_flag_ || init_flag_ || latest_image_.empty()) return;

    try {
        cv::Mat cropped, imitation_img, topomap_img;

        int x_start = (latest_image_.cols - latest_image_.rows) / 2;
        int y_start = (latest_image_.rows - latest_image_.rows) / 2;
        
        cv::Rect crop_rect(x_start, y_start, latest_image_.rows, latest_image_.rows);
        cropped = latest_image_(crop_rect).clone();
        cv::resize(cropped, topomap_img, cv::Size(85, 85));
        cv::resize(cropped, imitation_img, cv::Size(224, 224));

        int node_id_ = topo_localizer_.inferNode(topomap_img);
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

        at::Tensor image_tensor = torch::from_blob(
        imitation_img.data, 
        {1, 224, 224, 3}, 
        torch::kUInt8)
        .permute({0, 3, 1, 2})
        .clone()
        .to(torch::kFloat32)
        .div(255.0)
        .to(torch::kCUDA);

        at::Tensor cmd_tensor = torch::zeros({1, 4}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        cmd_tensor[0][command_idx] = 1.0;

        at::Tensor output = model_.forward({image_tensor, cmd_tensor}).toTensor();

        double predicted_angular = output.item<float>();
        predicted_angular = std::clamp(predicted_angular, -angular_max_, angular_max_);

        geometry_msgs::msg::Twist cmd_msg;

        // 障害物が検出された場合は速度を0にする
        if (obstacle_detected_) {
            cmd_msg.linear.x = 0.0;
            cmd_msg.angular.z = 0.0;
            RCLCPP_WARN(this->get_logger(), "Obstacle detected! Stopping robot.");
        } else {
            cmd_msg.linear.x = linear_max_;
            cmd_msg.angular.z = predicted_angular;
        }

        cmd_pub_->publish(cmd_msg);

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