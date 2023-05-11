#include <algorithm>
#include <executor.hpp>
#include <haze_model.hpp>
#include <image_loader/image_loader.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace exec {

Executor::Executor(const std::vector<cv::Mat>& images, const ProcessType type)
    : beta(1.5, 3.0),
      atmospheric_light_val(0.3, 0.7),
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
  return result;
}
cv::Mat Executor::Dehaze() const { return cv::Mat(img.size(), CV_64FC3); }

static std::string ResultErrorMessage(const std::string& lwhat,
                                      const std::string& rwhat) {
  return lwhat + rwhat + "\n";
}

void Produce(const std::vector<std::string>& input_pathes,
             std::string& result_path) {
  std::vector<load::PathWrapper> images_pathes;
  std::vector<load::PathWrapper> depth_map_pathes;

  ProcessType type = DEHAZING;

  try {
    images_pathes = load::LoadDir(input_pathes.front());
    if (input_pathes.size() > 1) {
      type = AUGMENTING;
      depth_map_pathes = load::LoadDir(input_pathes[1]);
    }
  } catch (const std::exception& ex) {
    throw std::runtime_error(ResultErrorMessage(
        "Produce(): cannot load content of input dirs:\n", ex.what()));
  }

  size_t size = images_pathes.size();
  if (depth_map_pathes.size() != size)
    throw std::runtime_error(
        "Produce(): input dirs has different numbers of files\n");

  load::PathWrapper result(result_path);
  try {
    if (!result.Empty())
      throw std::runtime_error("Produce(): result dir isn't empty");
  } catch (const std::exception& ex) {
    throw std::runtime_error(
        ResultErrorMessage("Produce(): incorrect result dir:\n", ex.what()));
  }

  try {
    for (size_t i = 0; i < size; ++i) {
      if (images_pathes[i].name != depth_map_pathes[i].name)
        throw std::runtime_error("Produce(): files must have equal filename");
      std::string name = images_pathes[i].name;
      std::vector<cv::Mat> images{load::LoadImg(images_pathes[i])};
      if (type == AUGMENTING)
        images.push_back(load::LoadImg(depth_map_pathes[i]));
      Executor ex(images, type);
      cv::Mat result_image = ex.Process();
      load::PathWrapper result_file_path;
      result_file_path.path = result.path / name;
      cv::Mat ui_result_image;
      result_image.convertTo(ui_result_image, CV_8UC3, 255.);
      cv::imwrite(result_file_path.ToString(), ui_result_image);
    }
  } catch (const std::exception& ex) {
    throw std::runtime_error(ResultErrorMessage(
        "Produce(): cannot augment/dehaze image:\n", ex.what()));
  }
}

}  // namespace exec
