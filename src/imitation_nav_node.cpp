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

    path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/path", 10);
    laserscan_pub_ = this->create_publisher<sensor_msgs::msg::LaserScan>("/scan", 10);
    cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

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

    // PurePursuitパラメータの初期化
    lookahead_distance_ = this->declare_parameter("lookahead_distance", 1.0);
    target_linear_velocity_ = this->declare_parameter("target_linear_velocity", 0.5);
    max_angular_velocity_ = this->declare_parameter("max_angular_velocity", 1.0);
    goal_tolerance_ = this->declare_parameter("goal_tolerance", 0.3);

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
    // PointCloud2をLaserScanに変換
    sensor_msgs::msg::LaserScan scan_msg = pointcloud_processor_->convertToLaserScan(msg);
    laserscan_pub_->publish(scan_msg);
    latest_scan_ = scan_msg;

    // 前方の最小距離を計算（collision monitor用）
    min_front_distance_ = pointcloud_processor_->computeMinFrontDistance(scan_msg);

    // 障害物検出
    obstacle_detected_ = pointcloud_processor_->detectObstacle(scan_msg);
}

geometry_msgs::msg::Twist ImitationNav::computePurePursuitControl(
    const nav_msgs::msg::Path& path)
{
    geometry_msgs::msg::Twist cmd_vel;
    cmd_vel.linear.x = 0.0;
    cmd_vel.angular.z = 0.0;

    if (path.poses.empty()) {
        return cmd_vel;
    }

    double closest_dist = std::numeric_limits<double>::max();
    size_t lookahead_idx = 0;
    bool found_lookahead = false;

    for (size_t i = 0; i < path.poses.size(); ++i) {
        double px = path.poses[i].pose.position.x;
        double py = path.poses[i].pose.position.y;
        double dist = std::sqrt(px * px + py * py);

        // Look-ahead距離に最も近い点を探す
        if (std::abs(dist - lookahead_distance_) < closest_dist) {
            closest_dist = std::abs(dist - lookahead_distance_);
            lookahead_idx = i;
            found_lookahead = true;
        }
    }

    if (!found_lookahead) {
        // 経路が見つからない場合は停止
        return cmd_vel;
    }

    // 目標点の座標
    double goal_x = path.poses[lookahead_idx].pose.position.x;
    double goal_y = path.poses[lookahead_idx].pose.position.y;
    double goal_dist = std::sqrt(goal_x * goal_x + goal_y * goal_y);

    // ゴール到達判定
    if (goal_dist < goal_tolerance_) {
        RCLCPP_INFO(this->get_logger(), "Goal reached!");
        return cmd_vel;
    }

    // 目標点への角度
    double theta = std::atan2(goal_y, goal_x);

    // PurePursuitの角速度計算
    // ω = 2 * v * sin(θ) / L
    double angular_vel = 2.0 * target_linear_velocity_ * std::sin(theta) / lookahead_distance_;

    // 角速度を制限
    angular_vel = std::max(-max_angular_velocity_,
                          std::min(max_angular_velocity_, angular_vel));

    // Collision monitorによる速度スケーリング
    double velocity_scale = PointCloudProcessor::computeVelocityScale(min_front_distance_);

    // 速度指令
    cmd_vel.linear.x = target_linear_velocity_ * velocity_scale;
    cmd_vel.angular.z = angular_vel;

    if (velocity_scale == 0.0) {
        RCLCPP_WARN(this->get_logger(), "Collision Monitor: Obstacle within 1m (%.2fm) - STOPPED", min_front_distance_);
    }

    return cmd_vel;
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

        auto output_cpu = output.to(torch::kCPU);
        auto output_accessor = output_cpu.accessor<float, 2>();

        nav_msgs::msg::Path path_msg;
        path_msg.header.stamp = this->now();
        path_msg.header.frame_id = "base_link";

        for (int i = 0; i < 30; ++i) {
            geometry_msgs::msg::PoseStamped pose;
            pose.header.stamp = this->now();
            pose.header.frame_id = "base_link";
            pose.pose.position.x = output_accessor[0][i * 2];
            pose.pose.position.y = output_accessor[0][i * 2 + 1];
            pose.pose.position.z = 0.0;
            pose.pose.orientation.w = 1.0;

            path_msg.poses.push_back(pose);
        }

        // パスを出版
        path_pub_->publish(path_msg);

        // PurePursuitで速度指令を計算
        geometry_msgs::msg::Twist cmd_vel = computePurePursuitControl(path_msg);
        cmd_vel_pub_->publish(cmd_vel);

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