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
    autonomous_flag_pub_ = this->create_publisher<std_msgs::msg::Bool>("/autonomous", 10);

    timer_ = this->create_wall_timer(std::chrono::milliseconds(interval_ms),
        std::bind(&ImitationNav::ImitationNavigation, this));

    // PointCloudProcessorを初期化
    double z_min = this->declare_parameter("z_min", -0.5);
    double z_max = this->declare_parameter("z_max", 0.5);

    // collision monitor パラメータ
    double collision_zone_stop = this->declare_parameter("collision_zone_stop", 1.0);
    double collision_zone_slow2 = this->declare_parameter("collision_zone_slow2", 2.0);
    double collision_zone_slow1 = this->declare_parameter("collision_zone_slow1", 3.0);
    double collision_gain_stop = this->declare_parameter("collision_gain_stop", 0.0);
    double collision_gain_slow2 = this->declare_parameter("collision_gain_slow2", 0.2);
    double collision_gain_slow1 = this->declare_parameter("collision_gain_slow1", 0.5);
    double collision_y_width = this->declare_parameter("collision_y_width", 0.7);

    double angle_increment_deg = this->declare_parameter("angle_increment_deg", 1.0);
    double range_max = this->declare_parameter("range_max", 10.0);

    // recovery パラメータ
    recovery_timeout_ = this->declare_parameter("recovery_timeout", 10.0);
    recovery_angular_gain_ = this->declare_parameter("recovery_angular_gain", 0.3);

    pointcloud_processor_ = std::make_shared<imitation_nav::PointCloudProcessor>(
        z_min, z_max,
        collision_zone_stop, collision_zone_slow2, collision_zone_slow1,
        collision_gain_stop, collision_gain_slow2, collision_gain_slow1,
        collision_y_width, angle_increment_deg, range_max
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
    // PointCloud2をLaserScanに変換
    sensor_msgs::msg::LaserScan scan_msg = pointcloud_processor_->convertToLaserScan(msg);

    // LaserScanをパブリッシュ
    laserscan_pub_->publish(scan_msg);

    // collision gainを計算
    collision_gain_ = pointcloud_processor_->calculateCollisionGain(scan_msg);

    // collision monitorを可視化
    pointcloud_processor_->visualizeCollisionMonitor(scan_msg, collision_gain_);
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

        // 前後10ID範囲にstop履歴がある場合、actionをroadsideに上書き
        if (action == "stop" && isNearStoppedNode(node_id_)) {
            RCLCPP_INFO(this->get_logger(),
                "Node %d has stop action but is near previous stop location. Treating as roadside.", node_id_);
            action = "roadside";
        }

        // stopアクションの処理
        if (action == "stop") {
            RCLCPP_WARN(this->get_logger(), "Stop action detected at node %d. Stopping navigation.", node_id_);

            // stop履歴に追加
            stopped_node_ids_.insert(node_id_);

            // autonomous_flag_をfalseにして停止
            autonomous_flag_ = false;

            // 外部にautonomous_flag_の変更を通知
            auto flag_msg = std_msgs::msg::Bool();
            flag_msg.data = false;
            autonomous_flag_pub_->publish(flag_msg);

            // 速度を0にして停止
            geometry_msgs::msg::Twist stop_cmd;
            stop_cmd.linear.x = 0.0;
            stop_cmd.angular.z = 0.0;
            cmd_pub_->publish(stop_cmd);

            return;
        }

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

        // 状態遷移ロジック
        if (collision_gain_ == 0.0) {
            if (nav_state_ == NavigationState::NORMAL) {
                // 停止検出 → STOPPED状態へ
                nav_state_ = NavigationState::STOPPED;
                stopped_time_ = this->now();
                RCLCPP_WARN(this->get_logger(), "Collision detected! Entered STOPPED state");
            }
            else if (nav_state_ == NavigationState::STOPPED) {
                // 停止時間チェック
                double elapsed = (this->now() - stopped_time_).seconds();
                if (elapsed >= recovery_timeout_) {
                    nav_state_ = NavigationState::RECOVERY;
                    RCLCPP_WARN(this->get_logger(),
                        "Timeout (%.1fs)! Entering RECOVERY mode", elapsed);
                }
            }
        } else {
            // collision_gain回復 → NORMAL状態へ
            if (nav_state_ != NavigationState::NORMAL) {
                RCLCPP_INFO(this->get_logger(), "Collision cleared! Returning to NORMAL");
                nav_state_ = NavigationState::NORMAL;
            }
        }

        // 速度指令生成
        geometry_msgs::msg::Twist cmd_msg;
        switch (nav_state_) {
            case NavigationState::NORMAL:
                cmd_msg.linear.x = linear_max_ * collision_gain_;
                cmd_msg.angular.z = predicted_angular * collision_gain_;
                if (collision_gain_ < 1.0) {
                    RCLCPP_WARN(this->get_logger(),
                        "Collision zone detected! Applying gain: %.2f (linear: %.2f, angular: %.2f)",
                        collision_gain_, cmd_msg.linear.x, cmd_msg.angular.z);
                }
                break;

            case NavigationState::STOPPED:
                cmd_msg.linear.x = 0.0;
                cmd_msg.angular.z = 0.0;
                break;

            case NavigationState::RECOVERY:
                cmd_msg.linear.x = 0.0;  // 並進速度0
                cmd_msg.angular.z = predicted_angular * recovery_angular_gain_;  // 低速旋回
                RCLCPP_WARN(this->get_logger(),
                    "RECOVERY: rotating with angular=%.2f", cmd_msg.angular.z);
                break;
        }

        cmd_pub_->publish(cmd_msg);

    } catch (const c10::Error &e) {
        RCLCPP_ERROR(this->get_logger(), "TorchScript inference error: %s", e.what());
    }
}

bool ImitationNav::isNearStoppedNode(int current_node_id) const {
    for (int stopped_id : stopped_node_ids_) {
        int diff = std::abs(current_node_id - stopped_id);
        if (diff <= 10) {
            return true;
        }
    }
    return false;
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