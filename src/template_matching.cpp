#include "imitation_nav/template_matching.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <iostream>
#include <fstream>

namespace imitation_nav
{

TemplateMatcher::TemplateMatcher(const std::string& template_dir, const std::string& map_yaml_path)
    : last_action_("straight")
{
    if (!loadTemplates(template_dir)) {
        throw std::runtime_error("Failed to load templates from directory: " + template_dir);
    }
    if (!loadMap(map_yaml_path)) {
        throw std::runtime_error("Failed to load map YAML file: " + map_yaml_path);
    }
    current_index_ = 0;
}

bool TemplateMatcher::loadTemplates(const std::string& dir)
{
    templates_.clear();

    for (int i = 1;; ++i)
    {
        std::ostringstream filename;
        filename << "img" << i << ".png";
        std::filesystem::path file_path = std::filesystem::path(dir) / filename.str();

        if (!std::filesystem::exists(file_path)) {
            break;
        }

        cv::Mat img = cv::imread(file_path.string(), cv::IMREAD_COLOR);
        if (!img.empty()) {
            templates_.push_back(img);
        } else {
            std::cerr << "Failed to load template: " << file_path << std::endl;
        }
    }

    return !templates_.empty();
}

bool TemplateMatcher::loadMap(const std::string& yaml_path)
{
    try {
        YAML::Node root = YAML::LoadFile(yaml_path);
        map_actions_.clear();

        if (!root["map_list"]) {
            std::cerr << "map_list not found in YAML." << std::endl;
            return false;
        }

        for (const auto& entry : root["map_list"]) {
            if (entry["edge"]) {
                int id = entry["edge"]["ID"].as<int>();
                std::string action = entry["edge"]["action"].as<std::string>();
                map_actions_.emplace_back(id, action);
            }
        }
        return !map_actions_.empty();
    }
    catch (const YAML::Exception& e) {
        std::cerr << "YAML error: " << e.what() << std::endl;
        return false;
    }
}

void TemplateMatcher::matchAndAdvance(const cv::Mat& input_img, double threshold)
{
    if (current_index_ >= templates_.size() || current_index_ >= map_actions_.size()) {
        std::cerr << "[Matcher] No more templates or map actions to process." << std::endl;
    }

    const cv::Mat& tmpl = templates_[current_index_];
    if (input_img.cols < tmpl.cols || input_img.rows < tmpl.rows) {
        std::cerr << "[Matcher] Input image smaller than template." << std::endl;
    }

    cv::Mat result;
    cv::matchTemplate(input_img, tmpl, result, cv::TM_CCOEFF_NORMED);
    double min_val, max_val;
    cv::minMaxLoc(result, &min_val, &max_val);

    std::cerr << "🧩 Matching (ID=" << map_actions_[current_index_].first
              << ", action=" << map_actions_[current_index_].second
              << "): score=" << max_val << std::endl;

    if (max_val >= threshold) {
        last_action_ = map_actions_[current_index_].second;
        ++current_index_;
    }
}

std::string TemplateMatcher::getCurrentAction() const
{
    return last_action_;
}

int TemplateMatcher::getCurrentID() const
{
    if (current_index_ == 0) return -1;
    return map_actions_[current_index_ - 1].first;
}

}  // namespace imitation_nav