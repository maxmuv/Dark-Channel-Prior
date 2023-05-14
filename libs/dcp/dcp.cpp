#include <dcp.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <vector>

namespace dcp {

cv::Mat DarkChannel(const cv::Mat& image, const int patch_size) {
  if (patch_size % 2 == 0)
    throw std::invalid_argument("DarkChannel(...): patch size can't be even");
  if (image.type() != CV_64FC3)
    throw std::invalid_argument("DarkChannel(...): image has incorrect type");
  std::vector<cv::Mat> colors;
  cv::Mat min_bg(image.size(), CV_64FC1);
  cv::split(image, colors);
  cv::Mat min(image.size(), CV_64FC1);
  cv::min(colors[0], colors[1], min_bg);
  cv::min(min_bg, colors[2], min);
  cv::Mat struct_el = cv::getStructuringElement(
      cv::MORPH_RECT, cv::Size(patch_size, patch_size));
  cv::Mat dark_channel(image.size(), CV_64FC1);
  cv::erode(min, dark_channel, struct_el, cv::Point(-1, -1), 1,
            cv::BORDER_REPLICATE);
  return dark_channel;
}

cv::Mat EstimateTransmission(const cv::Mat& hazy_image,
                             const cv::Mat& atmospheric_light,
                             const int patch_size, const double omega) {
  if (patch_size % 2 == 0)
    throw std::invalid_argument(
        "EstimateTransmission(...): patch size can't be even");
  if (hazy_image.type() != CV_64FC3)
    throw std::invalid_argument(
        "EstimateTransmission(...): hazy_image has incorrect type");
  if (atmospheric_light.type() != CV_64FC3)
    throw std::invalid_argument(
        "EstimateTransmission(...): atmospheric_light has incorrect type");
  if (atmospheric_light.size() != cv::Size(1, 1))
    throw std::invalid_argument(
        "EstimateTransmission(...): atmospheric_light has incorrect type");
  cv::Mat atmospheric_light_image(hazy_image.size(), CV_64FC3,
                                  atmospheric_light.at<cv::Scalar>(0, 0));
  cv::Mat norm_hazy_image_by_al(hazy_image.size(), CV_64FC3);
  cv::divide(hazy_image, atmospheric_light_image, norm_hazy_image_by_al);
  return cv::Mat(hazy_image.size(), CV_64FC1, cv::Scalar(1.)) -
         omega * DarkChannel(norm_hazy_image_by_al, patch_size);
}

cv::Mat SoftMatting(const cv::Mat& transmission, const cv::Mat& hazy_image,
                    const int patch_size, const double eps,
                    const double lambda) {
  /*
  cv::Mat mean_i(hazy_image.size(), CV_64FC3);
  cv::boxFilter(hazy_image, mean_i, -1, cv::Size(patch_size, patch_size));
  cv::Mat mean_ii(hazy_image.size(), CV_64FC3);
  cv::boxFilter(hazy_image.mul(hazy_image), mean_ii, -1,
                cv::Size(patch_size, patch_size));

  cv::Mat cov = mean_ii - mean_i.mul(mean_i);
  cv::Mat cov_eps = cov + cv::Mat(hazy_image.size(), CV_64FC3,
                                  cv::Scalar(eps / (patch_size * patch_size),
                                             eps / (patch_size * patch_size),
                                             eps / (patch_size * patch_size)));
  cv::Mat L(hazy_image.size(), CV_64FC3);
  cv::boxFilter(
      -1 * (cv::Mat(hazy_image.size(), CV_64FC3, cv::Scalar(1., 1., 1.)) +
            (hazy_image - mean_i).mul(hazy_image - mean_i).mul(1 / cov_eps)),
      L, -1, cv::Size(patch_size, patch_size));
  cv::Mat div = (cv::Mat(hazy_image.size(), CV_64FC3,
                         cv::Scalar(patch_size, patch_size, patch_size)) +
                 L);
  std::vector<cv::Mat> div_channels;
  cv::split(div, div_channels);
  cv::Mat res_div =
      div_channels[0].mul(div_channels[1]).mul(div_channels[1]) +
      cv::Mat(transmission.size(), CV_64FC1, cv::Scalar(lambda * patch_size));
  cv::Mat result = transmission + transmission.mul(1 / (res_div * lambda));
  return result;*/
  cv::Mat result(transmission.size(), CV_64FC1);
  cv::boxFilter(transmission, result, -1, cv::Size(patch_size, patch_size));
  return result;
}

cv::Mat EstimateAtmospericLight(const cv::Mat& hazy_image, const int patch_size,
                                const double brightest_share) {
  if (patch_size % 2 == 0)
    throw std::invalid_argument(
        "EstimateAtmospericLight(...): patch size can't be even");
  if (hazy_image.type() != CV_64FC3)
    throw std::invalid_argument(
        "EstimateAtmospericLight(...): hazy_image has incorrect type");
  if (brightest_share < 0 || brightest_share > 1)
    throw std::invalid_argument(
        "EstimateAtmospericLight(...): brightest_share is out of range");
  cv::Mat dark_channel = DarkChannel(hazy_image, patch_size);

  if (dark_channel.size() != hazy_image.size())
    throw std::invalid_argument(
        "EstimateAtmospericLight(...): size of hazy_image is not equal size of "
        "dark_channel");

  struct coordval {
    coordval(const int i, const int j, const double val, const cv::Vec3d& pix)
        : i(i), j(j), val(val) {
      intensity = pix[0] + pix[1] + pix[2];
    }
    int i = 0;
    int j = 0;
    double val;
    double intensity;
  };

  cv::Mat hazy_image_clone = hazy_image.clone();
  std::vector<coordval> pixel_intensities;
  pixel_intensities.reserve(dark_channel.rows * dark_channel.cols);
  for (int i = 0; i < dark_channel.rows; ++i) {
    for (int j = 0; j < dark_channel.cols; ++j) {
      pixel_intensities.emplace_back(i, j, dark_channel.at<double>(i, j),
                                     hazy_image_clone.at<cv::Vec3d>(i, j));
    }
  }
  std::stable_sort(pixel_intensities.begin(), pixel_intensities.end(),
                   [](const coordval& lhs, const coordval& rhs) {
                     return lhs.val > rhs.val;
                   });
  int border = static_cast<int>(
      std::max(1.0, 1.0 * pixel_intensities.size() * brightest_share));
  cv::Scalar atmospheric_light_val(0, 0, 0);
  auto comp_float = [](const double lhs, const double rhs) -> bool {
    double max = std::max({fabs(lhs), fabs(rhs), 1.0});
    if (fabs(rhs - lhs) < max * std::numeric_limits<double>::epsilon())
      return true;
    return false;
  };
  int al_num = 0;
  double max_intensity = 0;
  for (int i = 0; i < border; ++i) {
    auto& coords = pixel_intensities[i];
    if (max_intensity < coords.intensity) max_intensity = coords.intensity;
  }
  for (int i = 0; i < border; ++i) {
    auto& coords = pixel_intensities[i];
    if (comp_float(max_intensity, coords.intensity)) {
      atmospheric_light_val +=
          hazy_image_clone.at<cv::Scalar>(coords.i, coords.j);
      ++al_num;
    }
  }
  if (al_num == 0)
    throw std::runtime_error(
        "EstimateAtmospericLight(...): must be at least one pixel with max "
        "intensity");
  atmospheric_light_val /= al_num;
  cv::Mat atmospheric_light(1, 1, CV_64FC3, atmospheric_light_val);
  return atmospheric_light;
}

}  // namespace dcp
