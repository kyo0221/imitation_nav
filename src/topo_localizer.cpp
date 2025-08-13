#include "imitation_nav/topo_localizer.hpp"

#include <yaml-cpp/yaml.h>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace imitation_nav
{

TopoLocalizer::TopoLocalizer(const std::string& map_path, const std::string& model_path, const std::string& image_dir)
    : device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      delta_(5.0f),
      is_initialized_(false)
{
    image_path_ = image_dir;
    model_ = torch::jit::load(model_path);
    model_.to(device_);
    model_.eval();
    loadMap(map_path);
    
    setTransitionWindow(-2, 3);
}

void TopoLocalizer::initializeModel(const cv::Mat& image, bool use_observation_based_init) {
    torch::Tensor query_feature = extractFeature(image);

    std::vector<float> dists;
    dists.reserve(map_.size());
    
    for (const auto& node : map_) {
        auto ref_tensor = torch::from_blob((void*)node.feature.data(), 
                                         {static_cast<long>(node.feature.size())}, 
                                         torch::kFloat32);
        float dist = computeFeatureDistance(query_feature, ref_tensor);
        dists.push_back(dist);
    }

    // 統計的に妥当な分位点計算
    std::vector<float> sorted_dists = dists;
    std::sort(sorted_dists.begin(), sorted_dists.end());
    
    size_t n = sorted_dists.size();
    float q025 = sorted_dists[static_cast<size_t>(0.025 * n)];
    float q975 = sorted_dists[static_cast<size_t>(0.975 * n)];
    
    float range = q975 - q025;
    if (range < 1e-6f) {
        lambda1_ = 1.0f;  // デフォルト値
    } else {
        lambda1_ = std::log(delta_) / range;
    }

    // 初期信念分布を設定
    belief_.resize(map_.size());
    initializeBelief(dists, use_observation_based_init);
    
    is_initialized_ = true;
    
    std::cout << "[TopoLocalizer] Initialized with lambda1 = " << lambda1_ 
              << ", observation_based_init = " << (use_observation_based_init ? "true" : "false") << std::endl;
}

void TopoLocalizer::initializeBelief(const std::vector<float>& distances, bool use_observation_based_init) {
    // 全ての信念値を0で初期化
    std::fill(belief_.begin(), belief_.end(), 0.0f);
    
    if (use_observation_based_init) {
        // 観測尤度ベースの初期化：最も類似度の高いノードを中心に初期化
        auto min_iter = std::min_element(distances.begin(), distances.end());
        int best_idx = std::distance(distances.begin(), min_iter);
        
        const size_t uniform_range = 5;
        float uniform_prob = 1.0f / uniform_range;
        
        // 最も類似度の高いノードを中心に初期化
        int start_idx = std::max(0, best_idx - static_cast<int>(uniform_range) / 2);
        int end_idx = std::min(static_cast<int>(belief_.size()), start_idx + static_cast<int>(uniform_range));
        
        for (int i = start_idx; i < end_idx; ++i) {
            belief_[i] = uniform_prob;
        }
        
        std::cout << "[TopoLocalizer] Observation-based initialization centered at node " << best_idx << std::endl;
    } else {
        // ID0中心の初期化（従来の方法）
        const size_t uniform_range = 5;
        float uniform_prob = 1.0f / uniform_range;
        
        for (size_t i = 0; i < std::min(uniform_range, belief_.size()); ++i) {
            belief_[i] = uniform_prob;
        }
        
        std::cout << "[TopoLocalizer] ID0-centered initialization" << std::endl;
    }
}

void TopoLocalizer::setTransitionWindow(int window_lower, int window_upper) {
    window_lower_ = window_lower;
    window_upper_ = window_upper;
    
    int window_size = window_upper_ - window_lower_ + 1;
    transition_.resize(window_size);
    
    // ガウシアンライクな遷移確率を設定
    setupGaussianTransition();
}

void TopoLocalizer::setupGaussianTransition() {
    float sigma = 1.0f;  // 標準偏差
    float sum = 0.0f;
    
    // 各インデックスに対して遷移確率を計算
    for (size_t i = 0; i < transition_.size(); ++i) {
        // インデックス i に対応するオフセット（現在ノードからの相対位置）
        int offset = static_cast<int>(i) + window_lower_;
        
        float gaussian = std::exp(-0.5f * std::pow(offset / sigma, 2));
        transition_[i] = gaussian;
        sum += gaussian;
    }
    
    // 正規化（確率の総和を1にする）
    if (sum > 1e-8f) {
        for (float& t : transition_) {
            t /= sum;
        }
    } else {
        // フォールバック: 一様分布
        float uniform_prob = 1.0f / transition_.size();
        std::fill(transition_.begin(), transition_.end(), uniform_prob);
    }
    
    std::cout << "[TopoLocalizer] Transition probabilities:" << std::endl;
    for (size_t i = 0; i < transition_.size(); ++i) {
        int offset = static_cast<int>(i) + window_lower_;
        std::cout << "  Offset " << offset << ": " << transition_[i] << std::endl;
    }
}

float TopoLocalizer::computeFeatureDistance(const torch::Tensor& feat1, const torch::Tensor& feat2) {
    float dot_product = torch::dot(feat1, feat2).item<float>();
    float distance = std::sqrt(std::max(0.0f, 2.0f - 2.0f * dot_product));
    return distance;
}

void TopoLocalizer::loadMap(const std::string& map_path) {
    YAML::Node root = YAML::LoadFile(map_path);
    map_.clear();
    map_.reserve(root["nodes"].size());
    
    for (const auto& node : root["nodes"]) {
        TopoNode topo;
        topo.id = node["id"].as<int>();
        topo.image_path = node["image"].as<std::string>();
        
        if (node["edges"] && node["edges"].size() > 0) {
            topo.action = node["edges"][0]["action"].as<std::string>();
        } else {
            topo.action = "roadside";
        }
        
        topo.feature.clear();
        topo.feature.reserve(node["feature"].size());
        for (const auto& f : node["feature"]) {
            topo.feature.push_back(f.as<float>());
        }
        
        map_.emplace_back(std::move(topo));
    }
    
    std::cout << "[TopoLocalizer] Loaded " << map_.size() << " nodes" << std::endl;
}

torch::Tensor TopoLocalizer::extractFeature(const cv::Mat& image) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(85, 85));
    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

    // メモリの連続性を保証
    resized = resized.clone();
    
    auto input_tensor = torch::from_blob(resized.data, {1, 85, 85, 3}, torch::kFloat32)
                            .permute({0, 3, 1, 2})  // [N, C, H, W]
                            .clone()
                            .to(device_);

    // ImageNet正規化
    input_tensor[0][0] = input_tensor[0][0].sub(0.485).div(0.229);
    input_tensor[0][1] = input_tensor[0][1].sub(0.456).div(0.224);
    input_tensor[0][2] = input_tensor[0][2].sub(0.406).div(0.225);

    torch::NoGradGuard no_grad;  // 勾配計算を無効化
    return model_.forward(std::vector<torch::jit::IValue>{input_tensor})
                .toTensor()
                .cpu()
                .squeeze(0);
}

int TopoLocalizer::inferNode(const cv::Mat& input_image) {
    if (!is_initialized_) {
        std::cerr << "[TopoLocalizer] Error: Model not initialized. Call initializeModel() first." << std::endl;
        return -1;
    }
    
    // 特徴量抽出と観測尤度計算
    auto query_feature = extractFeature(input_image);
    std::vector<float> obs_likelihood = computeObservationLikelihood(query_feature);
    
    // 予測ステップ（遷移モデル適用）
    std::vector<float> predicted_belief = applyTransitionModel();
    
    // 更新ステップ（観測モデル適用）
    updateBelief(predicted_belief, obs_likelihood);
    
    // 最尤推定
    auto max_iter = std::max_element(belief_.begin(), belief_.end());
    int best_idx = std::distance(belief_.begin(), max_iter);
    
    // 結果画像の表示
    displayPredictedNode(best_idx);
    
    return map_[best_idx].id;
}

std::vector<float> TopoLocalizer::computeObservationLikelihood(const torch::Tensor& query_feature) {
    std::vector<float> query_vec(query_feature.data_ptr<float>(), 
                                query_feature.data_ptr<float>() + query_feature.size(0));
    
    std::vector<float> likelihood(map_.size());
    
    for (size_t i = 0; i < map_.size(); ++i) {
        const auto& feat = map_[i].feature;
        float dot = std::inner_product(feat.begin(), feat.end(), query_vec.begin(), 0.0f);
        float dist = std::sqrt(std::max(0.0f, 2.0f - 2.0f * dot));
        
        float exp_arg = -lambda1_ * dist;
        if (exp_arg < -50.0f) exp_arg = -50.0f;  // アンダーフロー防止
        
        likelihood[i] = std::exp(exp_arg);
    }
    
    return likelihood;
}

std::vector<float> TopoLocalizer::applyTransitionModel() {
    std::vector<float> predicted(belief_.size(), 0.0f);
    
    // 循環境界条件を適用した畳み込み
    for (size_t i = 0; i < belief_.size(); ++i) {
        for (size_t j = 0; j < transition_.size(); ++j) {
            int src_idx = static_cast<int>(i) - window_lower_ - static_cast<int>(j);
            
            if (src_idx < 0) {
                src_idx += static_cast<int>(belief_.size());
            } else if (src_idx >= static_cast<int>(belief_.size())) {
                src_idx -= static_cast<int>(belief_.size());
            }
            
            predicted[i] += belief_[src_idx] * transition_[j];
        }
    }
    
    return predicted;
}

void TopoLocalizer::updateBelief(const std::vector<float>& predicted_belief,
                                 const std::vector<float>& obs_likelihood) {
    size_t n = predicted_belief.size();
    belief_.resize(n);

    auto max_iter = std::max_element(predicted_belief.begin(), predicted_belief.end());
    int best_idx = std::distance(predicted_belief.begin(), max_iter);

    for (size_t i = 0; i < n; ++i) {
        int offset = static_cast<int>(i) - best_idx;
        if (offset < window_lower_ || offset > window_upper_) {
            belief_[i] = 0.0f;  // 遷移ウィンドウ外は信念をゼロに
        } else {
            belief_[i] = predicted_belief[i] * obs_likelihood[i];
        }
    }

    // 正規化
    float sum = std::accumulate(belief_.begin(), belief_.end(), 0.0f);
    if (sum > 1e-6f) {
        for (auto& b : belief_) {
            b /= sum;
        }
    } else {
        std::cerr << "[TopoLocalizer] Warning: Belief normalization failed (sum too small)." << std::endl;
    }
}

cv::Mat TopoLocalizer::addCommandOverlay(const cv::Mat& image, const std::string& command) const {
    cv::Mat result = image.clone();
    
    // オーバーレイエリアの設定
    int overlay_height = 80;
    int button_width = 100;
    int button_height = 60;
    int margin = 20;
    
    // 下部にオーバーレイエリアを作成
    cv::Rect overlay_rect(0, result.rows - overlay_height, result.cols, overlay_height);
    cv::Mat overlay = result(overlay_rect);
    overlay = cv::Scalar(30, 30, 30); // 暗いグレー背景
    
    // ボタンの位置を計算
    int total_width = 4 * button_width + 5 * margin;
    int start_x = (result.cols - total_width) / 2;
    int button_y = result.rows - overlay_height + 10;
    
    // 各コマンドボタンを描画
    std::vector<std::string> commands = {"roadside", "straight", "left", "right"};
    std::vector<cv::Scalar> colors = {
        cv::Scalar(100, 100, 100),  // roadside - グレー
        cv::Scalar(100, 100, 100),  // straight - グレー
        cv::Scalar(100, 100, 100),  // left - グレー
        cv::Scalar(100, 100, 100)   // right - グレー
    };
    
    // アクティブなコマンドを特定して色を変更
    for (size_t i = 0; i < commands.size(); ++i) {
        if (commands[i] == command) {
            colors[i] = cv::Scalar(0, 255, 0); // アクティブは緑色
        }
    }
    
    // ボタンを描画
    for (size_t i = 0; i < commands.size(); ++i) {
        int x = start_x + i * (button_width + margin);
        cv::Rect button_rect(x, button_y, button_width, button_height);
        
        // ボタンの背景
        cv::rectangle(result, button_rect, colors[i], -1);
        cv::rectangle(result, button_rect, cv::Scalar(255, 255, 255), 2);
        
        // テキスト
        cv::Point text_pos(x + 10, button_y + 35);
        cv::putText(result, commands[i], text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                   cv::Scalar(255, 255, 255), 2);
    }
    
    return result;
}

void TopoLocalizer::displayPredictedNode(int best_idx) const {
    cv::Mat best_node_image = cv::imread(image_path_ + map_[best_idx].image_path);
    if (!best_node_image.empty()) {
        cv::Mat resized;
        cv::resize(best_node_image, resized, cv::Size(480, 480));
        
        // 指令情報オーバーレイを追加
        cv::Mat display_image = addCommandOverlay(resized, map_[best_idx].action);
        
        cv::imshow("Predicted Node Image", display_image);
        cv::waitKey(1);
    } else {
        std::cerr << "[TopoLocalizer] Failed to load image at: " 
                 << image_path_ + map_[best_idx].image_path << std::endl;
    }
}



std::string TopoLocalizer::getNodeAction(int node_id) const {
    for (const auto& node : map_) {
        if (node.id == node_id) {
            return node.action;
        }
    }
    return "roadside";  // デフォルトアクション
}

float TopoLocalizer::getMaxBelief() const {
    if (belief_.empty()) return 0.0f;
    return *std::max_element(belief_.begin(), belief_.end());
}

float TopoLocalizer::getBeliefEntropy() const {
    float entropy = 0.0f;
    for (const float& b : belief_) {
        if (b > 1e-8f) {
            entropy -= b * std::log(b);
        }
    }
    return entropy;
}

} // namespace imitation_nav