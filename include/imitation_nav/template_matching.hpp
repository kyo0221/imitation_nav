#pragma once

#include <iostream>
#include <sstream>
#include <filesystem>

#include <std_msgs/msg/string.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace imitation_nav
{

class TemplateMatcher
{
public:
    TemplateMatcher(const std::string& template_dir, const std::string& map_yaml_path);
    void matchAndAdvance(const cv::Mat& input_img, double threshold);
    std::string getCurrentAction() const;
    int getCurrentID() const;

private:
    bool loadTemplates(const std::string& dir);
    bool loadMap(const std::string& yaml_path);

    std::vector<cv::Mat> templates_;
    std::vector<std::pair<int, std::string>> map_actions_;
    std::size_t current_index_;
    std::string last_action_;
};

}  // namespace imitation_nav