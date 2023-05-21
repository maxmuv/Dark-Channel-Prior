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
  std::vector<cv::Mat> Dehaze() const;

 public:
  Executor(const std::vector<cv::Mat>& images, const ProcessType type);
  std::vector<cv::Mat> Process() const;
  ~Executor() = default;

 private:
  double min_depth_val = 0.3;
  mutable std::uniform_real_distribution<> beta;  // (1.5, 3.0);
  mutable std::uniform_real_distribution<>
      atmospheric_light_val;  // (0.3, 0.7);
  mutable std::mt19937 gen;
  cv::Mat img;
  cv::Mat depth_map;
  const ProcessType type;
};

void Produce(const std::vector<std::string>& input_pathes,
             std::string& result_path);

}  // namespace exec
#endif  // EXECUTOR_HPP
