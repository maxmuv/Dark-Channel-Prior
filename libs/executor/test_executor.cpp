#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include <executor.hpp>
#include <opencv2/core.hpp>
#include <stdexcept>

TEST_CASE("image_processor") {
  std::vector<cv::Mat> mats;
  REQUIRE_THROWS_WITH_AS([&]() { exec::Executor ex(mats, exec::DEHAZING); }(),
                         "Executor::Executor(...): num of images is incorrect",
                         const std::invalid_argument&);
  for (int i = 0; i < 3; ++i)
    mats.emplace_back(20, 40, CV_64FC3, cv::Scalar(0.5, 0.5, 0.5));
  mats.emplace_back(20, 40, CV_64FC3, cv::Scalar(12, 0, 0));
  REQUIRE_THROWS_WITH_AS([&]() { exec::Executor ex(mats, exec::DEHAZING); }(),
                         "Executor::Executor(...): images are out of range",
                         const std::invalid_argument&);
  mats.pop_back();
  mats.emplace_back(20, 40, CV_64FC2, cv::Scalar(0, 0, 0));
  REQUIRE_THROWS_WITH_AS([&]() { exec::Executor ex(mats, exec::DEHAZING); }(),
                         "Executor::Executor(...): image types are incorrect",
                         const std::invalid_argument&);
  mats.pop_back();
  mats.emplace_back(10, 10, CV_64FC3, cv::Scalar(0, 0, 0));
  REQUIRE_THROWS_WITH_AS([&]() { exec::Executor ex(mats, exec::DEHAZING); }(),
                         "Executor::Executor(...): image sizes are incorrect",
                         const std::invalid_argument&);
  mats.pop_back();

  exec::Executor processor_dehazing(mats, exec::DEHAZING);
  std::vector<cv::Mat> result_dehazing;
  REQUIRE_NOTHROW(result_dehazing = processor_dehazing.Process());
  cv::Size s(40, 20);
  CHECK(result_dehazing.back().size() == s);
  CHECK_EQ(result_dehazing.back().type(), CV_64FC3);

  exec::Executor processor_augmenting(mats, exec::DEHAZING);
  std::vector<cv::Mat> result_augmenting;
  REQUIRE_NOTHROW(result_augmenting = processor_dehazing.Process());
  CHECK(result_augmenting.back().size() == s);
  CHECK_EQ(result_augmenting.back().type(), CV_64FC3);
}
