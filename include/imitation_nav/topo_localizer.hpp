#pragma once

#include <string>
#include <vector>
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
    std::string getNodeAction(int node_id) const;

private:
    std::vector<TopoNode> map_;
    torch::jit::script::Module model_;
    torch::Device device_;
    torch::Tensor extractFeature(const cv::Mat& image);
    void loadMap(const std::string& map_path);

    int current_idx_=0;
};

} // namespace imitation_nav
