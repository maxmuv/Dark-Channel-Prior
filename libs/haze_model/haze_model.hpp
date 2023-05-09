#pragma once
#ifndef HAZE_MODEL_HPP
#define HAZE_MODEL_HPP

#include <opencv2/core/mat.hpp>

namespace haze {

void CreateTransmission(cv::Mat &transmission, const cv::Mat &depth_map,
                        const double beta);

class HazeModel {
 private:
  cv::Mat transmission;
  cv::Mat atmospheric_light;
  const double t0;

 public:
  HazeModel() = delete;
  HazeModel(const HazeModel &) = delete;
  HazeModel(HazeModel &&) = delete;
  HazeModel &operator=(const HazeModel &) = delete;
  HazeModel &operator=(HazeModel &&) = delete;
  HazeModel(const cv::Mat &tr, const cv::Mat &al, const double t0 = 0.1);
  void AugmentImage(cv::Mat &result, const cv::Mat &scene_radiance) const;
  void RecoverImage(cv::Mat &result, const cv::Mat &observed_intensity) const;
  ~HazeModel() = default;
};

}  // namespace haze
#endif  // HAZE_MODEL_HPP
