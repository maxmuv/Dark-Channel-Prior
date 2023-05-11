#pragma once
#ifndef EXECUTOR_HPP
#define EXECUTOR_HPP

#include <opencv2/core/mat.hpp>
#include <random>
#include <vector>

namespace exec {

enum ProcessType { DEHAZING, AUGMENTING };

class Executor {
 private:
  Executor() = delete;
  Executor(const Executor&) = delete;
  Executor(Executor&&) = delete;
  Executor& operator=(const Executor&) = delete;
  Executor& operator=(Executor&&) = delete;
  cv::Mat Augment() const;
  cv::Mat Dehaze() const;

 public:
  Executor(const std::vector<cv::Mat>& images, const ProcessType type);
  cv::Mat Process() const;
  ~Executor() = default;

 private:
  double min_depth_val = 0.2;
  mutable std::uniform_real_distribution<> beta;  // (0.5, 4.0);
  mutable std::uniform_real_distribution<>
      atmospheric_light_val;  // (0.0, 1.0);
  mutable std::mt19937 gen;
  cv::Mat img;
  cv::Mat depth_map;
  const ProcessType type;
};

}  // namespace exec
#endif  // EXECUTOR_HPP
