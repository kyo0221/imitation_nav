#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include <torch/script.h>

namespace imitation_nav
{

struct TopoNode {
    int id;
    std::string image_path;
    std::string action;
    std::vector<float> feature;
};

class TopoLocalizer {
public:
    TopoLocalizer(const std::string& map_path, const std::string& model_path, const std::string& image_dir);
    
    void initializeModel(const cv::Mat& image, bool use_observation_based_init = false);
    void setTransitionWindow(int window_lower, int window_upper);
    int inferNode(const cv::Mat& input_image);
    
    std::string getNodeAction(int node_id) const;
    
    float getMaxBelief() const;
    float getBeliefEntropy() const;

private:
    // 内部メソッド
    void loadMap(const std::string& map_path);
    torch::Tensor extractFeature(const cv::Mat& image);
    void initializeBelief(const std::vector<float>& distances, bool use_observation_based_init = false);
    void initializePlot();
    void setupGaussianTransition();
    float computeFeatureDistance(const torch::Tensor& feat1, const torch::Tensor& feat2);
    
    std::vector<float> computeObservationLikelihood(const torch::Tensor& query_feature);
    std::vector<float> applyTransitionModel();
    void updateBelief(const std::vector<float>& predicted_belief, const std::vector<float>& obs_likelihood);
    void displayPredictedNode(int best_idx) const;
    cv::Mat addCommandOverlay(const cv::Mat& image, const std::string& command) const;
    // void displayBliefHist() const;
    void displayCombinedHist(const std::vector<float>& obs_likelihood) const;

    std::vector<TopoNode> map_;
    std::vector<float> belief_;
    std::vector<float> transition_;
    
    torch::jit::script::Module model_;
    torch::Device device_;
    
    std::string image_path_;
    float lambda1_;                      // 観測モデルパラメータ
    float delta_;                        // 信頼度パラメータ
    
    int window_lower_;
    int window_upper_;
    
    bool is_initialized_;
};

} // namespace imitation_nav