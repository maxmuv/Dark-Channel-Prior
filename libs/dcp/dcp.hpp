#pragma once
#ifndef DCP_HPP
#define DCP_HPP

#include <opencv2/core/mat.hpp>

namespace dcp {

cv::Mat SoftMatting(const cv::Mat& transmission, const cv::Mat& hazy_image,
                    const int patch_size, const double eps,
                    const double lambda = 1e-4);

cv::Mat DarkChannel(const cv::Mat& image, const int patch_size);

cv::Mat EstimateTransmission(const cv::Mat& hazy_image,
                             const cv::Mat& atmospheric_light,
                             const int patch_size, const double omega = 0.95);

cv::Mat EstimateAtmospericLight(const cv::Mat& hazy_image, const int patch_size,
                                const double brightest_share = 1e-3);

}  // namespace dcp
#endif  // DCP_HPP
