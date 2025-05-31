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

void TopoLocalizer::initializeModel(const cv::Mat& image) {
    torch::Tensor query_feature = extractFeature(image);

    std::vector<float> dists;
    for (const auto& node : map_) {
        auto ref_tensor = torch::from_blob((void*)node.feature.data(), {static_cast<long>(node.feature.size())}, torch::kFloat32);
        float dist = std::sqrt(2.0f - 2.0f * torch::dot(query_feature, ref_tensor).item<float>());
        dists.push_back(dist);
    }

    // 分位点（2.5%と97.5%）を計算
    std::vector<float> sorted_dists = dists;
    std::sort(sorted_dists.begin(), sorted_dists.end());
    float q025 = sorted_dists[static_cast<size_t>(0.025 * sorted_dists.size())];
    float q975 = sorted_dists[static_cast<size_t>(0.975 * sorted_dists.size())];

    lambda1_ = std::log(delta_) / (q975 - q025);

    belief_.resize(dists.size());
    float sum = 0.0f;
    for (size_t i = 0; i < dists.size(); ++i) {
        belief_[i] = std::exp(-lambda1_ * dists[i]);
        sum += belief_[i];
    }
    if (sum > 1e-6) {
        for (float& b : belief_) {
            b /= sum;
        }
    }
}

void TopoLocalizer::setTransitionWindow(int window_lower, int window_upper) {
    window_lower_ = window_lower;
    window_upper_ = window_upper;
    int window_size = window_upper_ - window_lower_;
    transition_.assign(window_size, 1.0f);  // 一様分布で初期化
    float sum = static_cast<float>(window_size);
    for (float& v : transition_) v /= sum;
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
    std::vector<float> query_vec(query_feature.data_ptr<float>(), query_feature.data_ptr<float>() + query_feature.size(0));

    std::vector<float> obs_lhood(map_.size(), 0.0f);
    for (size_t i = 0; i < map_.size(); ++i) {
        const auto& feat = map_[i].feature;
        float dot = std::inner_product(feat.begin(), feat.end(), query_vec.begin(), 0.0f);
        float dist = std::sqrt(std::max(0.0f, 2.0f - 2.0f * dot));
        obs_lhood[i] = std::exp(-lambda1_ * dist);
    }

    const int w_size = static_cast<int>(transition_.size());
    std::vector<float> padded_belief;
    padded_belief.reserve(belief_.size() + 2 * (w_size - 1));

    for (int i = w_size - 1; i > 0; --i)
        padded_belief.push_back(belief_[i]);
    padded_belief.insert(padded_belief.end(), belief_.begin(), belief_.end());
    for (int i = belief_.size() - 2; i >= static_cast<int>(belief_.size()) - w_size + 1; --i)
        padded_belief.push_back(belief_[i]);

    std::vector<float> conv_result(belief_.size(), 0.0f);
    for (size_t i = 0; i < belief_.size(); ++i) {
        for (size_t j = 0; j < transition_.size(); ++j) {
            conv_result[i] += padded_belief[i + j] * transition_[j];
        }
    }

    for (size_t i = 0; i < belief_.size(); ++i) {
        belief_[i] = conv_result[i] * obs_lhood[i];
    }

    // 正規化
    float sum = std::accumulate(belief_.begin(), belief_.end(), 0.0f);
    if (sum > 1e-6f) {
        for (float& b : belief_) {
            b /= sum;
        }
    }

    auto max_iter = std::max_element(belief_.begin(), belief_.end());
    int best_idx = std::distance(belief_.begin(), max_iter);


    std::vector<std::pair<size_t, float>> indexed_belief;
    for (size_t i = 0; i < obs_lhood.size(); ++i) {
        indexed_belief.emplace_back(i, obs_lhood[i]);
    }

    std::sort(indexed_belief.begin(), indexed_belief.end(),
            [](const auto& a, const auto& b) {
                return a.second > b.second;
            });

    std::cout << "[TopoLocalizer] Top 5 belief nodes:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), indexed_belief.size()); ++i) {
        size_t idx = indexed_belief[i].first;
        std::cout << "  Rank " << (i + 1)
                << ": Node ID = " << map_[idx].id
                << ", Belief = " << belief_[idx]
                << ", ObsLhood = " << obs_lhood[idx]
                << std::endl;
    }

    std::cout << "best_idx : " << best_idx << std::endl;

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