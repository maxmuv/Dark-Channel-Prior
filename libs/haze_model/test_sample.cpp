#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include <fstream>
#include <haze_model.hpp>
#include <image_loader.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

TEST_CASE("Sample") {
  load::PathWrapper img_path(std::string(CSDIR) +
                             "/sample/00022_00193_outdoor_000_000.png");
  // loading with utf-8
  cv::Mat img = load::LoadImgUTF8(img_path);
  load::PathWrapper map_path(std::string(CSDIR) +
                             "/sample/00022_00193_outdoor_000_000_depth.png");
  cv::Mat map = load::LoadImgUTF8(map_path);
  CHECK_EQ(cv::Size(1024, 768), img.size());
  CHECK_EQ(cv::Size(1024, 768), map.size());
  CHECK_EQ(img.type(), CV_8UC3);
  CHECK_EQ(map.type(), CV_8UC3);
  std::vector<cv::Mat> maps_1c;
  cv::split(map, maps_1c);
  auto& map1c = maps_1c.front();
  CHECK_EQ(map1c.type(), CV_8UC1);
  cv::Mat norm_d_img;
  cv::Mat d_depth;
  img.convertTo(norm_d_img, CV_64FC3, 1.0 / 255.0);
  map1c.convertTo(d_depth, CV_64FC1, 1.0 / 255.0);
  cv::Mat transmission(map.size(), CV_64FC1);
  cv::Mat clipped_d_depth;
  cv::max(d_depth, 0.2, clipped_d_depth);
  cv::Mat blured_clipped_d_depth;
  cv::blur(clipped_d_depth, blured_clipped_d_depth, cv::Size(30, 30));
  REQUIRE_NOTHROW(
      haze::CreateTransmission(transmission, blured_clipped_d_depth,
                               2.0));  // beta \in (0.5, 4) - tr \in (1%, 90%)
  cv::Scalar mean = cv::mean(norm_d_img);
  double max_mean = std::max({mean[0], mean[1], mean[2]});
  std::cout << max_mean;
  cv::Mat atmospheric_light(1, 1, CV_64FC3,
                            cv::Scalar(max_mean, max_mean, max_mean));
  haze::HazeModel model(transmission, atmospheric_light);
  cv::Mat augmented_image(img.size(), CV_64FC3);
  REQUIRE_NOTHROW(model.AugmentImage(augmented_image, norm_d_img));
  cv::Mat ui_augmented_image;
  augmented_image.convertTo(ui_augmented_image, CV_8UC3, 255.);
  cv::imwrite(std::string(CSDIR) + "/sample/augmented_image.png",
              ui_augmented_image);
  cv::Mat recovered_image(img.size(), CV_64FC3);
  REQUIRE_NOTHROW(model.RecoverImage(recovered_image, augmented_image));
  cv::Mat ui_recovered_image;
  recovered_image.convertTo(ui_recovered_image, CV_8UC3, 255.);
  cv::imwrite(std::string(CSDIR) + "/sample/recovered_image.png",
              ui_recovered_image);
}
