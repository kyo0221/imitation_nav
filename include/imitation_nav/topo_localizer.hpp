#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

namespace imitation_nav
{

struct TopoNode {
    int id;
    std::string image_path;
    std::vector<float> feature;
    std::string action;
    std::vector<int> edges;
};

class TopoLocalizer {
public:
    TopoLocalizer(const std::string& map_path, const std::string& model_path);
    int inferNode(const cv::Mat& input_image);
    void initializeModel(const cv::Mat& image);
    void setTransitionWindow(int window_lower, int window_upper);

    std::string getNodeAction(int node_id) const;

private:
    std::vector<TopoNode> map_;
    torch::jit::script::Module model_;
    torch::Device device_;

    torch::Tensor extractFeature(const cv::Mat& image);
    void loadMap(const std::string& map_path);

    std::vector<float> belief_;     // 各ノードへの確率（正規化済み）
    float lambda1_ = 0.0f;
    float delta_ = 5.0f;
    int window_lower_;
    int window_upper_;
    
    std::vector<float> transition_;


    torch::Tensor last_feature_;
};

} // namespace imitation_nav
