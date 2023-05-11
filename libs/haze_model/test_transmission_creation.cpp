#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <math.h>

#include <haze_model.hpp>

static bool IsDoubleMatsEqual(const cv::Mat& lhs, const cv::Mat& rhs) {
  // (hypothesis) it seems that in OpenCV cv::compare is breaked for zero filled
  // mats.
  double result = true;
  for (int i = 0; i < lhs.rows; ++i) {
    for (int j = 0; j < rhs.cols; ++j) {
      double l = lhs.at<double>(i, j);
      double r = rhs.at<double>(i, j);
      double max_l_r_1 = std::max({1.0, fabs(r), fabs(l)});
      if (fabs(l - r) >
          max_l_r_1 * std::numeric_limits<double>::epsilon()) {
        result = false;
      }
    }
  }
  return result;
}

TEST_CASE("1x1 matrix") {
  cv::Mat depth_0(1, 1, CV_64FC1, cv::Scalar(0));
  double beta = 12.3;  // random number
  cv::Mat transmission(1, 1, CV_64FC1);
  REQUIRE_NOTHROW(haze::CreateTransmission(transmission, depth_0, beta));
  cv::Mat ideal(1, 1, CV_64FC1, cv::Scalar(1));
  CHECK(IsDoubleMatsEqual(transmission, ideal));

  cv::Mat depth_1(1, 1, CV_64FC1, cv::Scalar(1));
  double beta_0 = 0.0;
  REQUIRE_NOTHROW(haze::CreateTransmission(transmission, depth_1, beta_0));
  CHECK(IsDoubleMatsEqual(transmission, ideal));

  double beta_1 = 1.0;
  REQUIRE_NOTHROW(haze::CreateTransmission(transmission, depth_1, beta_1));
  cv::Mat ideal_1(1, 1, CV_64FC1, cv::Scalar(exp(-1.0)));
  CHECK(IsDoubleMatsEqual(transmission, ideal_1));
}

TEST_CASE("1x2 matrix") {
  cv::Mat depth(1, 2, CV_64FC1, cv::Scalar(1.0));
  depth.at<double>(0, 1) = 0.0;
  double beta = 1;
  cv::Mat transmission(1, 2, CV_64FC1);
  REQUIRE_NOTHROW(haze::CreateTransmission(transmission, depth, beta));
  cv::Mat ideal(1, 2, CV_64FC1, cv::Scalar(exp(-1.0), 1.0));
  ideal.at<double>(0, 1) = 1.0;
  CHECK(IsDoubleMatsEqual(transmission, ideal));
}
