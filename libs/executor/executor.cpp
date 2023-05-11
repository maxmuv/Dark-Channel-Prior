#include <algorithm>
#include <executor.hpp>
#include <haze_model.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace exec {

Executor::Executor(const std::vector<cv::Mat>& images, const ProcessType type)
    : beta(0.5, 4.0),
      atmospheric_light_val(0.0, 1.0),
      gen(std::random_device()()),
      type(type) {
  if (images.size() <= static_cast<size_t>(type))
    throw std::invalid_argument(
        "Executor::Executor(...): num of images is incorrect");
  cv::Size img_size = images.front().size();
  std::for_each(images.begin(), images.end(), [&](const cv::Mat& m) {
    if (m.type() != CV_64FC3)
      throw std::invalid_argument(
          "Executor::Executor(...): image types are incorrect");
    if (m.size() != img_size)
      throw std::invalid_argument(
          "Executor::Executor(...): image sizes are incorrect");
    try {
      cv::checkRange(m, false, 0, -std::numeric_limits<double>::epsilon(),
                     1.0 + std::numeric_limits<double>::epsilon());
    } catch (...) {
      throw std::invalid_argument(
          "Executor::Executor(...): images are out of range");
    }
  });
  img = images.front().clone();
  if (type == AUGMENTING) {
    depth_map = images[1].clone();
  }
}

cv::Mat Executor::Process() const {
  if (type == AUGMENTING)
    return Augment();
  else
    return Dehaze();
}

cv::Mat Executor::Augment() const {
  cv::Mat transmission(depth_map.size(), CV_64FC1);
  double v = atmospheric_light_val(gen);
  cv::Mat atmospheric_light(1, 1, CV_64FC3, cv::Scalar(v, v, v));

  std::vector<cv::Mat> maps_1c;
  cv::split(depth_map, maps_1c);
  auto& map1c = maps_1c.front();
  cv::Mat blured_depth_map;
  cv::blur(map1c, blured_depth_map, cv::Size(30, 30));
  cv::Mat clipped_blured_depth_map;
  cv::max(blured_depth_map, min_depth_val, clipped_blured_depth_map);
  haze::CreateTransmission(transmission, clipped_blured_depth_map, beta(gen));
  haze::HazeModel model(transmission, atmospheric_light);
  cv::Mat result(img.size(), CV_64FC3);
  model.AugmentImage(result, img);
}
cv::Mat Executor::Dehaze() const { return cv::Mat(img.size(), CV_64FC3); }

}  // namespace exec
