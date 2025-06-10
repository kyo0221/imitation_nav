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

void TopoLocalizer::initializeModel(const cv::Mat& image) {
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
    initializeBelief(dists);
    
    is_initialized_ = true;
    
    std::cout << "[TopoLocalizer] Initialized with lambda1 = " << lambda1_ << std::endl;
}

void TopoLocalizer::initializeBelief(const std::vector<float>& distances) {
    float sum = 0.0f;
    const float sigma = 1.0f;  // ガウス分布の標準偏差（必要に応じて調整）
    
    // 0を基準としたガウス分布で初期信念分布を設定
    for (size_t i = 0; i < distances.size(); ++i) {
        // ガウス分布: exp(-x^2 / (2*σ^2))
        float gaussian_value = std::exp(-(distances[i] * distances[i]) / (2.0f * sigma * sigma));
        belief_[i] = gaussian_value;
        sum += belief_[i];
    }
    
    // 正規化
    if (sum > 1e-8f) {
        for (float& b : belief_) {
            b /= sum;
        }
    } else {
        float uniform_prob = 1.0f / belief_.size();
        std::fill(belief_.begin(), belief_.end(), uniform_prob);
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
            topo.action = "straight";
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
    
    // デバッグ情報出力
    printTopBeliefs(5);
    
    // 結果画像の表示
    displayPredictedNode(best_idx);
    displayBliefHist();
    
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
    float sum = 0.0f;
    
    // ベイズ更新: P(state|obs) ∝ P(obs|state) * P(state)
    for (size_t i = 0; i < belief_.size(); ++i) {
        belief_[i] = predicted_belief[i] * obs_likelihood[i];
        sum += belief_[i];
    }
    
    // 正規化
    if (sum > 1e-8f) {
        for (float& b : belief_) {
            b /= sum;
        }
    } else {
        // 数値的に不安定な場合は観測尤度のみを使用
        std::cout << "[TopoLocalizer] Warning: Numerical instability detected, using observation likelihood only" << std::endl;
        sum = 0.0f;
        for (size_t i = 0; i < belief_.size(); ++i) {
            belief_[i] = obs_likelihood[i];
            sum += belief_[i];
        }
        if (sum > 1e-8f) {
            for (float& b : belief_) {
                b /= sum;
            }
        }
    }
}

void TopoLocalizer::printTopBeliefs(int top_k) const {
    std::vector<std::pair<size_t, float>> indexed_belief;
    for (size_t i = 0; i < belief_.size(); ++i) {
        indexed_belief.emplace_back(i, belief_[i]);
    }
    
    std::sort(indexed_belief.begin(), indexed_belief.end(),
            [](const auto& a, const auto& b) {
                return a.second > b.second;
            });
    
    std::cout << "[TopoLocalizer] Top " << top_k << " belief nodes:" << std::endl;
    for (int i = 0; i < std::min(top_k, static_cast<int>(indexed_belief.size())); ++i) {
        size_t idx = indexed_belief[i].first;
        std::cout << "  Rank " << (i + 1)
                << ": Node ID = " << map_[idx].id
                << ", Belief = " << indexed_belief[i].second
                << std::endl;
    }
}

void TopoLocalizer::displayPredictedNode(int best_idx) const {
    cv::Mat best_node_image = cv::imread(image_path_ + map_[best_idx].image_path);
    if (!best_node_image.empty()) {
        cv::Mat resized;
        cv::resize(best_node_image, resized, cv::Size(480, 480));
        cv::imshow("Predicted Node Image", resized);
        cv::waitKey(1);
    } else {
        std::cerr << "[TopoLocalizer] Failed to load image at: " 
                 << image_path_ + map_[best_idx].image_path << std::endl;
    }
}

void TopoLocalizer::displayBliefHist() const {
    if (belief_.empty()) {
        std::cerr << "[TopoLocalizer] No belief data to display" << std::endl;
        return;
    }
    
    // ヒストグラム画像のサイズとパラメータ
    const int img_width = 1200;
    const int img_height = 600;
    const int margin = 80;
    const int hist_width = img_width - 2 * margin;
    const int hist_height = img_height - 2 * margin;
    
    // 白い背景の画像を作成
    cv::Mat hist_img(img_height, img_width, CV_8UC3, cv::Scalar(255, 255, 255));
    
    // 最大信念値を取得（スケーリング用）
    float max_belief = *std::max_element(belief_.begin(), belief_.end());
    if (max_belief < 1e-8f) max_belief = 1.0f;
    
    // バーの幅を計算
    int bar_width = hist_width / static_cast<int>(belief_.size());
    if (bar_width < 1) bar_width = 1;
    
    // 最大信念値のインデックスを特定
    auto max_iter = std::max_element(belief_.begin(), belief_.end());
    int max_idx = std::distance(belief_.begin(), max_iter);
    
    // ヒストグラムのバーを描画
    for (size_t i = 0; i < belief_.size(); ++i) {
        // バーの高さを計算（信念値に比例）
        int bar_height = static_cast<int>((belief_[i] / max_belief) * hist_height);
        
        // バーの位置を計算
        int x = margin + static_cast<int>(i) * bar_width;
        int y = margin + hist_height - bar_height;
        
        // バーの色を決定（最大値は赤、それ以外は青）
        cv::Scalar bar_color = (static_cast<int>(i) == max_idx) ? 
                              cv::Scalar(0, 0, 255) : cv::Scalar(255, 100, 100);
        
        // バーを描画
        cv::rectangle(hist_img, 
                     cv::Point(x, y), 
                     cv::Point(x + bar_width - 1, margin + hist_height), 
                     bar_color, 
                     cv::FILLED);
        
        // バーの輪郭を描画
        cv::rectangle(hist_img, 
                     cv::Point(x, y), 
                     cv::Point(x + bar_width - 1, margin + hist_height), 
                     cv::Scalar(0, 0, 0), 
                     1);
    }
    
    // 軸とグリッドを描画
    // X軸
    cv::line(hist_img, 
             cv::Point(margin, margin + hist_height), 
             cv::Point(margin + hist_width, margin + hist_height), 
             cv::Scalar(0, 0, 0), 2);
    
    // Y軸
    cv::line(hist_img, 
             cv::Point(margin, margin), 
             cv::Point(margin, margin + hist_height), 
             cv::Scalar(0, 0, 0), 2);
    
    // グリッドライン（Y軸方向）
    for (int i = 1; i <= 10; ++i) {
        int y = margin + (hist_height * i) / 10;
        cv::line(hist_img, 
                 cv::Point(margin, y), 
                 cv::Point(margin + hist_width, y), 
                 cv::Scalar(200, 200, 200), 1);
    }
    
    // ラベルとテキストを追加
    // タイトル
    cv::putText(hist_img, "Belief Distribution - Topological Localization", 
                cv::Point(margin, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    
    cv::putText(hist_img, "Belief", 
                cv::Point(10, margin + hist_height/2), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
    
    cv::putText(hist_img, "Node ID", 
                cv::Point(margin + hist_width/2 - 40, img_height - 20), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
    
    for (int i = 0; i <= 10; ++i) {
        float value = (max_belief * i) / 10.0f;
        int y = margin + hist_height - (hist_height * i) / 10;
        
        std::string label = std::to_string(value);
        label = label.substr(0, label.find('.') + 4);
        
        cv::putText(hist_img, label, 
                    cv::Point(margin - 70, y + 5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
    }
    
    int label_interval = std::max(1, static_cast<int>(belief_.size()) / 20);
    for (size_t i = 0; i < belief_.size(); i += label_interval) {
        int x = margin + static_cast<int>(i) * bar_width + bar_width/2;
        std::string node_id = std::to_string(map_[i].id);
        
        cv::putText(hist_img, node_id, 
                    cv::Point(x - 10, margin + hist_height + 20), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
    }
    
    // 統計情報を表示
    float entropy = getBeliefEntropy();
    std::string info_text = "Max Belief: " + std::to_string(max_belief).substr(0, 6) + 
                           "  Entropy: " + std::to_string(entropy).substr(0, 6) + 
                           "  Best Node: " + std::to_string(map_[max_idx].id);
    
    cv::putText(hist_img, info_text, 
                cv::Point(margin, img_height - 40), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
    
    // 最大信念値のノードに注釈を追加
    if (max_idx < static_cast<int>(belief_.size())) {
        int max_x = margin + max_idx * bar_width + bar_width/2;
        int max_y = margin + hist_height - static_cast<int>((belief_[max_idx] / max_belief) * hist_height);
        
        // 矢印を描画
        cv::arrowedLine(hist_img, 
                       cv::Point(max_x, max_y - 30), 
                       cv::Point(max_x, max_y - 5), 
                       cv::Scalar(0, 0, 255), 2);
        
        std::string max_text = "MAX: " + std::to_string(belief_[max_idx]).substr(0, 6);
        cv::putText(hist_img, max_text, 
                    cv::Point(max_x - 30, max_y - 35), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    }
    
    cv::imshow("Belief Distribution Histogram", hist_img);
    cv::waitKey(1);
}

std::string TopoLocalizer::getNodeAction(int node_id) const {
    for (const auto& node : map_) {
        if (node.id == node_id) {
            return node.action;
        }
    }
    return "straight";  // デフォルトアクション
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