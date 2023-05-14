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

/*
cv::Mat SoftMatting(const cv::Mat& transmission, const cv::Mat& hazy_image,
                    const int patch_size, const double eps,
                    const double lambda) {
  cv::Mat result(transmission.size(), transmission.type());
  for (int i = 0; i < transmission.rows; ++i) {
    for (int j = 0; j < transmission.cols; ++j) {
      int win_y1 = std::min(i + patch_size / 2, transmission.rows);
      int win_x1 = std::min(j + patch_size / 2, transmission.cols);
      int win_x0 = std::max(j - patch_size / 2, 0);
      int win_y0 = std::max(i - patch_size / 2, 0);
      int center_x = i - win_x0;
      int center_y = j - win_y0;
      cv::Mat tr_reg =
          transmission(cv::Range(win_y0, win_y1), cv::Range(win_x0, win_x1));
      cv::Mat ha_reg =
          hazy_image(cv::Range(win_y0, win_y1), cv::Range(win_x0, win_x1));
      std::vector<cv::Mat> ha_reg_ch;
      cv::split(ha_reg, ha_reg_ch);

      cv::Mat L(tr_reg.size(), CV_64FC1, cv::Scalar(0));
      cv::Mat mean(1, 3, CV_64FC1);
      mean.at<double>(0, 0) = cv::mean(ha_reg_ch[0])[0];
      mean.at<double>(0, 1) = cv::mean(ha_reg_ch[1])[0];
      mean.at<double>(0, 2) = cv::mean(ha_reg_ch[2])[0];
      cv::Mat mean_1;
      cv::transpose(mean, mean_1);
      cv::Mat covar(3, 3, CV_64FC1);
      covar.at<double>(0, 0) =
          cv::mean(ha_reg_ch[0].mul(ha_reg_ch[0]) -
                   mean.at<double>(0, 0) * mean.at<double>(0, 0))[0] +
          eps / (tr_reg.cols * tr_reg.rows);
      covar.at<double>(0, 1) =
          cv::mean(ha_reg_ch[0].mul(ha_reg_ch[1]) -
                   mean.at<double>(0, 0) * mean.at<double>(0, 1))[0];
      covar.at<double>(0, 2) =
          cv::mean(ha_reg_ch[0].mul(ha_reg_ch[2]) -
                   mean.at<double>(0, 0) * mean.at<double>(0, 2))[0];
      covar.at<double>(1, 0) =
          cv::mean(ha_reg_ch[1].mul(ha_reg_ch[0]) -
                   mean.at<double>(0, 1) * mean.at<double>(0, 0))[0];
      covar.at<double>(1, 1) =
          cv::mean(ha_reg_ch[1].mul(ha_reg_ch[1]) -
                   mean.at<double>(1, 0) * mean.at<double>(0, 1))[0] +
          eps / (tr_reg.cols * tr_reg.rows);
      covar.at<double>(1, 2) =
          cv::mean(ha_reg_ch[1].mul(ha_reg_ch[2]) -
                   mean.at<double>(0, 1) * mean.at<double>(0, 2))[0];
      covar.at<double>(2, 0) =
          cv::mean(ha_reg_ch[2].mul(ha_reg_ch[0]) -
                   mean.at<double>(0, 1) * mean.at<double>(0, 0))[0];
      covar.at<double>(2, 1) =
          cv::mean(ha_reg_ch[2].mul(ha_reg_ch[1]) -
                   mean.at<double>(1, 0) * mean.at<double>(0, 1))[0];
      covar.at<double>(2, 2) =
          cv::mean(ha_reg_ch[2].mul(ha_reg_ch[2]) -
                   mean.at<double>(0, 1) * mean.at<double>(0, 2))[0] +
          eps / (tr_reg.cols * tr_reg.rows);
      for (int Li = 0; Li < tr_reg.cols; ++Li) {
        for (int Lj = 0; Lj < tr_reg.rows; ++Lj) {
          if (Li == Lj) {
            L.at<double>(Li, Lj) += (1. + lambda);
          }
          cv::Mat pix(1, 3, CV_64FC1);
          pix.at<double>(0, 0) = ha_reg_ch[0].at<double>(Li, Lj);
          pix.at<double>(0, 1) = ha_reg_ch[1].at<double>(Li, Lj);
          pix.at<double>(0, 2) = ha_reg_ch[2].at<double>(Li, Lj);
          cv::Mat pix_1;
          cv::transpose(pix, pix_1);
          cv::Mat expr = ((pix - mean) * covar.inv() * (pix_1 - mean_1));
          std::cout << expr;
          L.at<double>(Li, Lj) -=
              1. / (tr_reg.cols * tr_reg.rows) * (1 + expr.at<double>(0, 0));
        }
      }
      cv::Mat res;
      cv::solve(L, lambda * tr_reg, res);
      res
    }
  }
  return result;
}*/

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
    coordval(const int i, const int j, const double val, const cv::Scalar& pix)
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
  for (int i = 0; i < dark_channel.rows; ++i) {
    for (int j = 0; j < dark_channel.cols; ++j) {
      pixel_intensities.emplace_back(i, j, dark_channel.at<double>(i, j),
                                     hazy_image_clone.at<cv::Scalar>(i, j));
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
