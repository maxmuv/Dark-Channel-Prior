#include <haze_model.hpp>
#include <opencv2/core.hpp>
#include <stdexcept>

namespace haze {

HazeModel::HazeModel(const cv::Mat& tr, const cv::Mat& al, const double t0)
    : t0(t0) {
  if (tr.type() != CV_64FC1 || al.type() != CV_64FC3)
    throw std::invalid_argument("HazeModel::HazeModel(...): incorrect type");
  if (tr.size() == cv::Size(0, 0))
    throw std::invalid_argument(
        "HazeModel::HazeModel(...): incorrect matrices' sizes");
  if (al.size() != cv::Size(1, 1))
    throw std::invalid_argument(
        "HazeModel::HazeModel(...): incorrect matrices' sizes");
  transmission = tr.clone();
  atmospheric_light = al.clone();
}

void HazeModel::AugmentImage(cv::Mat& result,
                             const cv::Mat& scene_radiance) const {
  if (scene_radiance.type() != CV_64FC3)
    throw std::invalid_argument(
        "HazeModel::AugmentImage(...): incorrect type of input");
  if (result.type() != CV_64FC3)
    throw std::invalid_argument(
        "HazeModel::AugmentImage(...): incorrect type of result");
  if (scene_radiance.size() != transmission.size())
    throw std::invalid_argument(
        "HazeModel::AugmentImage(...): incorrect size of input");
  if (result.size() != transmission.size())
    throw std::invalid_argument(
        "HazeModel::AugmentImage(...): incorrect size of result");
  cv::Mat transmission_3_ch;
  cv::Mat channels[3] = {transmission, transmission, transmission};
  cv::merge(channels, 3, transmission_3_ch);
  cv::Mat inverse_transmission = cv::Mat(transmission.size(), CV_64FC3,
                                         cv::Vec<double, 3>(1.0, 1.0, 1.0)) -
                                 transmission_3_ch;
  cv::Mat atmospheric_light_image(
      transmission.size(), CV_64FC3,
      atmospheric_light.at<cv::Vec<double, 3>>(0, 0));
  result = transmission_3_ch.mul(scene_radiance) +
           inverse_transmission.mul(atmospheric_light_image);
}

void HazeModel::RecoverImage(cv::Mat& result,
                             const cv::Mat& observed_intensity) const {
  if (observed_intensity.type() != CV_64FC3)
    throw std::invalid_argument(
        "HazeModel::RecoverImage(...): incorrect type of input");
  if (result.type() != CV_64FC3)
    throw std::invalid_argument(
        "HazeModel::RecoverImage(...): incorrect type of result");
  if (observed_intensity.size() != transmission.size())
    throw std::invalid_argument(
        "HazeModel::RecoverImage(...): incorrect size of input");
  if (result.size() != transmission.size())
    throw std::invalid_argument(
        "HazeModel::RecoverImage(...): incorrect size of result");
}

}  // namespace haze
