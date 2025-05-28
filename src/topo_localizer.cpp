#include "imitation_nav/topo_localizer.hpp"

#include <yaml-cpp/yaml.h>
#include <iostream>
#include <cmath>

namespace imitation_nav
{

TopoLocalizer::TopoLocalizer(const std::string& map_path, const std::string& model_path)
    : device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
{
    model_ = torch::jit::load(model_path);
    model_.to(device_);
    model_.eval();
    loadMap(map_path);
}

void TopoLocalizer::loadMap(const std::string& map_path) {
    YAML::Node root = YAML::LoadFile(map_path);
    for (const auto& node : root["nodes"]) {
        TopoNode topo;
        topo.id = node["id"].as<int>();
        topo.image_path = node["image"].as<std::string>();
        topo.action = node["edges"][0]["action"].as<std::string>();
        for (const auto& f : node["feature"]) {
            topo.feature.push_back(f.as<float>());
        }
        map_.emplace_back(topo);
    }
}

torch::Tensor TopoLocalizer::extractFeature(const cv::Mat& image) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(200, 88));
    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

    resized = resized.clone();  // ensure contiguous memory
    auto input_tensor = torch::from_blob(resized.data, {1, 88, 200, 3}, torch::kFloat32)
                            .permute({0, 3, 1, 2})  // [N, C, H, W]
                            .clone()
                            .to(device_);

    input_tensor[0][0] = input_tensor[0][0].sub(0.485).div(0.229);
    input_tensor[0][1] = input_tensor[0][1].sub(0.456).div(0.224);
    input_tensor[0][2] = input_tensor[0][2].sub(0.406).div(0.225);

    return model_.forward(std::vector<torch::jit::IValue>{input_tensor})
                .toTensor()
                .cpu()
                .squeeze(0);
}

int TopoLocalizer::inferNode(const cv::Mat& input_image) {
    auto query_feature = extractFeature(input_image);
    int best_idx = -1;
    float best_dist = std::numeric_limits<float>::max();

    for (size_t i = 0; i < map_.size(); ++i) {
        const auto& feat = map_[i].feature;
        auto ref_tensor = torch::from_blob((void*)feat.data(), {static_cast<long>(feat.size())}, torch::kFloat32);
        float dist = torch::nn::functional::pairwise_distance(query_feature, ref_tensor).item<float>();
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = static_cast<int>(i);
        }
    }

    return map_[best_idx].id;
}

std::string TopoLocalizer::getNodeAction(int node_id) const {
    for (const auto& node : map_) {
        if (node.id == node_id) {
            return node.action;
        }
    }
    return "straight";
}

} // namespace imitation_nav