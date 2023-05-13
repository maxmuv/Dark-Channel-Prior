#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include <dcp.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>

static bool IsDoubleMatsEqual(const cv::Mat& lhs, const cv::Mat& rhs) {
  // (hypothesis) it seems that in OpenCV cv::compare is breaked for zero filled
  // mats.
  double result = true;
  for (int i = 0; i < lhs.rows; ++i) {
    for (int j = 0; j < rhs.cols; ++j) {
      double l = lhs.at<double>(i, j);
      double r = rhs.at<double>(i, j);
      double max_l_r_1 = std::max({1.0, fabs(r), fabs(l)});
      if (fabs(l - r) > max_l_r_1 * std::numeric_limits<double>::epsilon()) {
        result = false;
      }
    }
  }
  return result;
}

class DCPFixture {
 protected:
  static std::unique_ptr<cv::Mat> transmission;
};

std::unique_ptr<cv::Mat> DCPFixture::transmission = nullptr;

TEST_CASE("DarkChannel") {
  cv::Mat test(2, 3, CV_64FC3, cv::Scalar(1, 1, 1));
  cv::Mat atmospheric_light(1, 1, CV_64FC3, cv::Scalar(1.0, 1.0, 1.0));
  cv::Mat dark_channel_ideal(2, 3, CV_64FC1, cv::Scalar(0));

  test.at<cv::Vec3d>(1, 2) = cv::Vec3d(1, 0, 1);
  dark_channel_ideal.at<double>(0, 0) = 1;
  dark_channel_ideal.at<double>(1, 0) = 1;
  REQUIRE_THROWS_WITH_AS([&]() { dcp::DarkChannel(test, 2); }(),
                         "DarkChannel(...): patch size can't be even",
                         const std::invalid_argument&);
  CHECK(IsDoubleMatsEqual(dcp::DarkChannel(test, 3), dark_channel_ideal));
}

TEST_CASE_FIXTURE(DCPFixture, "EstimateTransmission") {
  cv::Mat test(2, 3, CV_64FC3, cv::Scalar(1, 1, 1));
  cv::Mat atmospheric_light(1, 1, CV_64FC3, cv::Scalar(1.0, 1.0, 1.0));
  cv::Mat dark_channel_ideal(2, 3, CV_64FC1, cv::Scalar(1));

  test.at<cv::Vec3d>(1, 2) = cv::Vec3d(1, 0, 1);
  dark_channel_ideal.at<double>(0, 0) = 0.05;
  dark_channel_ideal.at<double>(1, 0) = 0.05;
  REQUIRE_THROWS_WITH_AS(
      [&]() { dcp::EstimateTransmission(test, atmospheric_light, 2); }(),
      "EstimateTransmission(...): patch size can't be even",
      const std::invalid_argument&);
  transmission.reset(new cv::Mat());
  *transmission.get() =
      dcp::EstimateTransmission(test, atmospheric_light, 3).clone();
  CHECK(IsDoubleMatsEqual(*transmission.get(), dark_channel_ideal));
}

/*TEST_CASE_FIXTURE(DCPFixture, "SoftMatting") {
  cv::Mat test(2, 2, CV_64FC3, cv::Scalar(1, 1, 1));
  cv::Mat atmospheric_light(1, 1, CV_64FC3, cv::Scalar(1.0, 1.0, 1.0));
  cv::Mat ideal(2, 2, CV_64FC1, cv::Scalar(0.375));
  transmission.reset(new cv::Mat());
  *transmission.get() = dcp::SoftMatting(
      dcp::EstimateTransmission(test, atmospheric_light, 3, 0.5), test, 3, 0,
      0.25);
  std::cout << *transmission.get();
  CHECK(IsDoubleMatsEqual(*transmission.get(), ideal));
}*/

TEST_CASE("EstimateAtmospericLight") {
  cv::Mat test(2, 3, CV_64FC3, cv::Scalar(1, 1, 1));
  test.at<cv::Vec3d>(0, 2) = cv::Vec3d(1, 0, 1);
  cv::Mat atm_light = dcp::EstimateAtmospericLight(test, 3);
  for (int i = 0; i < 3; ++i) {
    CHECK_EQ(doctest::Approx(atm_light.at<cv::Vec3d>(0, 0)[i]), 1.0);
  }
}
