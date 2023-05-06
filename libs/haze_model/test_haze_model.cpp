#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include <haze_model.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

static bool IsDoubleMatsEqual(const cv::Mat& lhs, const cv::Mat& rhs) {
  // (hypothesis) it seems that in OpenCV cv::compare is breaked for zero filled
  // mats.
  double result = true;
  for (int i = 0; i < lhs.rows; ++i) {
    for (int j = 0; j < rhs.cols; ++j) {
      cv::Vec3d l = lhs.at<cv::Vec3d>(i, j);
      cv::Vec3d r = rhs.at<cv::Vec3d>(i, j);
      for (int c = 0; c < 3; ++c) {
        double max_l_r_1 = std::max({1.0, fabs(r[c]), fabs(l[c])});
        if (fabs(l[c] - r[c]) >
            max_l_r_1 * std::numeric_limits<double>::epsilon()) {
          result = false;
        }
      }
    }
  }
  return result;
}

class HazeModelFixture {
 protected:
  static std::unique_ptr<haze::HazeModel> haze_model;
  static std::unique_ptr<cv::Mat> hazy_image;
  static std::unique_ptr<cv::Mat> scene_radiance;
};

std::unique_ptr<haze::HazeModel> HazeModelFixture::haze_model = nullptr;
std::unique_ptr<cv::Mat> HazeModelFixture::hazy_image = nullptr;
std::unique_ptr<cv::Mat> HazeModelFixture::scene_radiance = nullptr;

/// further tests use 1x1 Double 3-channel Matriсes

TEST_CASE_FIXTURE(HazeModelFixture, "HazeModelConstructor1") {
  // In further tests:
  // transmision = 0.5
  // atmospheric_light = (0., 0.5, 1.)
  cv::Mat t_transmission(1, 1, CV_64FC1, cv::Scalar(0.5));
  cv::Mat t_atmospheric_light(1, 1, CV_64FC3, cv::Scalar(0., 0.5, 1.));
  SUBCASE("correct") {
    REQUIRE_NOTHROW(haze_model.reset(
        new haze::HazeModel(t_transmission, t_atmospheric_light)));
  }
  SUBCASE("incorrect type") {
    cv::Mat f_atmospheric_light(1, 1, CV_16UC2);
    CHECK_THROWS_WITH_AS(
        [&]() { new haze::HazeModel(t_transmission, f_atmospheric_light); }(),
        "HazeModel::HazeModel(...): incorrect type",
        const std::invalid_argument&);
    cv::Mat f_transmission(1, 1, CV_32SC3);
    CHECK_THROWS_WITH_AS(
        [&]() { new haze::HazeModel(f_transmission, t_atmospheric_light); }(),
        "HazeModel::HazeModel(...): incorrect type",
        const std::invalid_argument&);
  }
  SUBCASE("incorrect matrices' sizes") {
    cv::Mat f_atmospheric_light(2, 1, CV_64FC3);
    CHECK_THROWS_WITH_AS(
        [&]() { new haze::HazeModel(t_transmission, f_atmospheric_light); }(),
        "HazeModel::HazeModel(...): incorrect matrices' sizes",
        const std::invalid_argument&);
    cv::Mat t_transmission(1, 2, CV_64FC1);
    CHECK_NOTHROW([&]() {
      haze::HazeModel* ptr =
          new haze::HazeModel(t_transmission, t_atmospheric_light);
      delete ptr;
    }());
  }
}

TEST_CASE_FIXTURE(HazeModelFixture, "AugmentImage1") {
  // scene_radiance = (1., 0.5, 0.)
  scene_radiance.reset(new cv::Mat(1, 1, CV_64FC3, cv::Scalar(1., 0.5, 0)));
  // hazy_image = (1.*0.5 + 0*0.5, 0.5*0.5+0.5*0.5, 0.*0.5 + 1*0.5) = (0.5, 0.5,
  // 0.5)
  hazy_image.reset(new cv::Mat(1, 1, CV_64FC3, cv::Scalar(0.5, 0.5, 0.5)));
  cv::Mat zero_augmented_image;
  REQUIRE_THROWS_AS(
      haze_model->AugmentImage(zero_augmented_image, *scene_radiance.get()),
      const std::invalid_argument&);
  cv::Mat augmented_image(1, 1, CV_64FC3);
  REQUIRE_NOTHROW(
      haze_model->AugmentImage(augmented_image, *scene_radiance.get()));
  // cv::Mat dst(1, 1, CV_64FC3);
  //  cv::compare(*hazy_image.get(), augmented_image, dst, cv::CMP_NE);
  //  std::vector<cv::Mat> channels;
  //  cv::split(dst, channels);
  //  std::for_each(channels.begin(), channels.end(),
  //                [&](const cv::Mat& ch) { CHECK_EQ(cv::countNonZero(ch), 0);
  //                });
  CHECK(IsDoubleMatsEqual(*hazy_image.get(), augmented_image));
  SUBCASE("incorrect type") {
    cv::Mat f_radiance(1, 1, CV_16FC3);
    cv::Mat dst(1, 1, CV_64FC3);
    CHECK_THROWS_WITH_AS(
        haze_model->AugmentImage(dst, f_radiance),
        "HazeModel::AugmentImage(...): incorrect type of input",
        const std::invalid_argument&);
    cv::Mat f_dst(1, 1, CV_16FC3);
    CHECK_THROWS_WITH_AS(
        haze_model->AugmentImage(f_dst, *scene_radiance.get()),
        "HazeModel::AugmentImage(...): incorrect type of result",
        const std::invalid_argument&);
  }
  SUBCASE("incorrect syze") {
    cv::Mat f_radiance(1, 2, CV_64FC3);
    cv::Mat dst(1, 1, CV_64FC3);
    CHECK_THROWS_WITH_AS(
        haze_model->AugmentImage(dst, f_radiance),
        "HazeModel::AugmentImage(...): incorrect size of input",
        const std::invalid_argument&);
    cv::Mat f_dst(1, 2, CV_64FC3);
    CHECK_THROWS_WITH_AS(
        haze_model->AugmentImage(f_dst, *scene_radiance.get()),
        "HazeModel::AugmentImage(...): incorrect size of result",
        const std::invalid_argument&);
  }
}

TEST_CASE_FIXTURE(HazeModelFixture, "RecoverImage1") {
  // scene_radiance = (1., 0.5, 0.)
  // hazy_image = (1.*0.5 + 0*0.5, 0.5*0.5+0.5*0.5, 0.*0.5 + 1*0.5) = (0.5, 0.5,
  // 0.5)
  cv::Mat zero_recovered_image;
  REQUIRE_THROWS_AS(
      haze_model->RecoverImage(zero_recovered_image, *hazy_image.get()),
      const std::invalid_argument&);
  cv::Mat recovered_image(1, 1, CV_64FC3);
  REQUIRE_NOTHROW(haze_model->RecoverImage(recovered_image, *hazy_image.get()));
  // cv::Mat dst(1, 1, CV_64FC3);
  // cv::compare(*scene_radiance.get(), recovered_image, dst, cv::CMP_NE);
  // std::vector<cv::Mat> channels;
  // cv::split(dst, channels);
  // std::for_each(channels.begin(), channels.end(),
  //              [&](const cv::Mat& ch) { CHECK_EQ(cv::countNonZero(ch), 0);
  //              });
  CHECK(IsDoubleMatsEqual(*scene_radiance.get(), recovered_image));
  SUBCASE("incorrect type") {
    cv::Mat f_image(1, 1, CV_16FC3);
    cv::Mat dst(1, 1, CV_64FC3);
    CHECK_THROWS_WITH_AS(
        haze_model->RecoverImage(dst, f_image),
        "HazeModel::RecoverImage(...): incorrect type of input",
        const std::invalid_argument&);
    cv::Mat f_dst(1, 1, CV_16FC3);
    CHECK_THROWS_WITH_AS(
        haze_model->RecoverImage(f_dst, *hazy_image.get()),
        "HazeModel::RecoverImage(...): incorrect type of result",
        const std::invalid_argument&);
  }
  SUBCASE("incorrect syze") {
    cv::Mat f_image(1, 2, CV_64FC3);
    cv::Mat dst(1, 1, CV_64FC3);
    CHECK_THROWS_WITH_AS(
        haze_model->RecoverImage(dst, f_image),
        "HazeModel::RecoverImage(...): incorrect size of input",
        const std::invalid_argument&);
    cv::Mat f_dst(1, 2, CV_64FC3);
    CHECK_THROWS_WITH_AS(
        haze_model->RecoverImage(f_dst, *hazy_image.get()),
        "HazeModel::RecoverImage(...): incorrect size of result",
        const std::invalid_argument&);
  }
}

TEST_CASE_FIXTURE(HazeModelFixture, "Destructor1") {
  CHECK_NE(nullptr, haze_model.release());
  CHECK_NE(nullptr, hazy_image.release());
  CHECK_NE(nullptr, scene_radiance.release());
}

/// further test use 2x2 Double 3-channel Matriсes

TEST_CASE_FIXTURE(HazeModelFixture, "HazeModelConstructor2") {
  // In further tests:
  // transmision = (0.1
  //                0.5,
  //                1.0)
  // atmospheric_light = (0., 0.5, 1.)
  cv::Mat t_transmission(3, 1, CV_64FC1, cv::Scalar(0.5));
  t_transmission.at<double>(0) = 0.1;
  t_transmission.at<double>(2) = 1.0;
  cv::Mat t_atmospheric_light(1, 1, CV_64FC3, cv::Scalar(0., 0.5, 1.));
  SUBCASE("correct") {
    REQUIRE_NOTHROW(haze_model.reset(
        new haze::HazeModel(t_transmission, t_atmospheric_light, 1e-9)));
  }
  SUBCASE("incorrect type") {
    cv::Mat f_atmospheric_light(1, 1, CV_16UC2);
    CHECK_THROWS_WITH_AS(
        [&]() { new haze::HazeModel(t_transmission, f_atmospheric_light); }(),
        "HazeModel::HazeModel(...): incorrect type",
        const std::invalid_argument&);
    cv::Mat f_transmission(3, 1, CV_32SC3);
    CHECK_THROWS_WITH_AS(
        [&]() { new haze::HazeModel(f_transmission, t_atmospheric_light); }(),
        "HazeModel::HazeModel(...): incorrect type",
        const std::invalid_argument&);
  }
  SUBCASE("incorrect matrices' sizes") {
    cv::Mat f_atmospheric_light(3, 1, CV_64FC3);
    CHECK_THROWS_WITH_AS(
        [&]() { new haze::HazeModel(t_transmission, f_atmospheric_light); }(),
        "HazeModel::HazeModel(...): incorrect matrices' sizes",
        const std::invalid_argument&);
  }
}

TEST_CASE_FIXTURE(HazeModelFixture, "AugmentImage2") {
  // scene_radiance = ((1., 0.5, 0.),
  //                   (1., 0.5, 0.),
  //                   (1., 0.5, 0.))
  scene_radiance.reset(new cv::Mat(3, 1, CV_64FC3, cv::Scalar(1., 0.5, 0)));
  // hazy_image = ((0.1, 0.5, 0.9),
  //               (0.5, 0.5, 0.5),
  //               (1.0, 0.5, 0.0))
  hazy_image.reset(new cv::Mat(3, 1, CV_64FC3, cv::Scalar(0.5, 0.5, 0.5)));
  hazy_image->at<cv::Vec<double, 3>>(0, 0) = cv::Vec<double, 3>(0.1, 0.5, 0.9);
  hazy_image->at<cv::Vec<double, 3>>(2, 0) = cv::Vec<double, 3>(1.0, 0.5, 0.0);
  cv::Mat zero_augmented_image;
  REQUIRE_THROWS_AS(
      haze_model->AugmentImage(zero_augmented_image, *scene_radiance.get()),
      const std::invalid_argument&);
  cv::Mat augmented_image(3, 1, CV_64FC3);
  REQUIRE_NOTHROW(
      haze_model->AugmentImage(augmented_image, *scene_radiance.get()));
  // cv::Mat dst(3, 1, CV_64FC3);
  // cv::compare(*hazy_image.get(), augmented_image, dst, cv::CMP_NE);
  // std::vector<cv::Mat> channels;
  // cv::split(dst, channels);
  // std::for_each(channels.begin(), channels.end(),
  //               [&](const cv::Mat& ch) { CHECK_EQ(cv::countNonZero(ch), 0);
  //               });
  CHECK(IsDoubleMatsEqual(*hazy_image.get(), augmented_image));
  SUBCASE("incorrect type") {
    cv::Mat f_radiance(3, 1, CV_16FC3);
    cv::Mat dst(3, 1, CV_64FC3);
    CHECK_THROWS_WITH_AS(
        haze_model->AugmentImage(dst, f_radiance),
        "HazeModel::AugmentImage(...): incorrect type of input",
        const std::invalid_argument&);
    cv::Mat f_dst(3, 1, CV_16FC3);
    CHECK_THROWS_WITH_AS(
        haze_model->AugmentImage(f_dst, *scene_radiance.get()),
        "HazeModel::AugmentImage(...): incorrect type of result",
        const std::invalid_argument&);
  }
  SUBCASE("incorrect syze") {
    cv::Mat f_radiance(1, 2, CV_64FC3);
    cv::Mat dst(3, 1, CV_64FC3);
    CHECK_THROWS_WITH_AS(
        haze_model->AugmentImage(dst, f_radiance),
        "HazeModel::AugmentImage(...): incorrect size of input",
        const std::invalid_argument&);
    cv::Mat f_dst(3, 2, CV_64FC3);
    CHECK_THROWS_WITH_AS(
        haze_model->AugmentImage(f_dst, *scene_radiance.get()),
        "HazeModel::AugmentImage(...): incorrect size of result",
        const std::invalid_argument&);
  }
}

TEST_CASE_FIXTURE(HazeModelFixture, "RecoverImage2") {
  // scene_radiance = ((1., 0.5, 0.),
  //                   (1., 0.5, 0.),
  //                   (1., 0.5, 0.))
  // hazy_image = ((0.1, 0.5, 0.9),
  //               (0.5, 0.5, 0.5),
  //               (1.0, 0.5, 0.0))
  cv::Mat zero_recovered_image;
  REQUIRE_THROWS_AS(
      haze_model->RecoverImage(zero_recovered_image, *hazy_image.get()),
      const std::invalid_argument&);
  cv::Mat recovered_image(3, 1, CV_64FC3);
  REQUIRE_NOTHROW(haze_model->RecoverImage(recovered_image, *hazy_image.get()));
  // cv::Mat dst(3, 1, CV_64FC3);
  // cv::compare(*scene_radiance.get(), recovered_image, dst, cv::CMP_NE);
  // std::vector<cv::Mat> channels;
  // cv::split(dst, channels);
  // std::for_each(channels.begin(), channels.end(),
  //               [&](const cv::Mat& ch) { CHECK_EQ(cv::countNonZero(ch), 0);
  //               });
  CHECK(IsDoubleMatsEqual(*scene_radiance.get(), recovered_image));
  SUBCASE("incorrect type") {
    cv::Mat f_image(3, 1, CV_16FC3);
    cv::Mat dst(3, 1, CV_64FC3);
    CHECK_THROWS_WITH_AS(
        haze_model->RecoverImage(dst, f_image),
        "HazeModel::RecoverImage(...): incorrect type of input",
        const std::invalid_argument&);
    cv::Mat f_dst(3, 1, CV_16FC3);
    CHECK_THROWS_WITH_AS(
        haze_model->RecoverImage(f_dst, *hazy_image.get()),
        "HazeModel::RecoverImage(...): incorrect type of result",
        const std::invalid_argument&);
  }
  SUBCASE("incorrect syze") {
    cv::Mat f_image(3, 2, CV_64FC3);
    cv::Mat dst(3, 1, CV_64FC3);
    CHECK_THROWS_WITH_AS(
        haze_model->RecoverImage(dst, f_image),
        "HazeModel::RecoverImage(...): incorrect size of input",
        const std::invalid_argument&);
    cv::Mat f_dst(3, 2, CV_64FC3);
    CHECK_THROWS_WITH_AS(
        haze_model->RecoverImage(f_dst, *hazy_image.get()),
        "HazeModel::RecoverImage(...): incorrect size of result",
        const std::invalid_argument&);
  }
}

TEST_CASE_FIXTURE(HazeModelFixture, "Destructor2") {
  CHECK_NE(nullptr, haze_model.release());
  CHECK_NE(nullptr, hazy_image.release());
  CHECK_NE(nullptr, scene_radiance.release());
}

TEST_CASE("t0 is not zero") {
  // transmission = (0.01)
  // atmospheric_light = (10, 10, 10)
  // scene_radiance = (100, 100, 100)
  // hasy_image = (10.9, 10.9, 10.9)
  // t0 = 0.1
  // recovered_image = (19, 19, 19)
  cv::Mat transmission(1, 1, CV_64FC1, cv::Scalar(0.01));
  cv::Mat atmospheric_light(1, 1, CV_64FC3, cv::Scalar(10, 10, 10));
  cv::Mat scene_radiance(1, 1, CV_64FC3, cv::Scalar(100, 100, 100));
  haze::HazeModel model(transmission, atmospheric_light);
  cv::Mat ideal_hazy_image(1, 1, CV_64FC3, cv::Scalar(10.9, 10.9, 10.9));
  cv::Mat hazy_image(1, 1, CV_64FC3);
  model.AugmentImage(hazy_image, scene_radiance);
  CHECK(IsDoubleMatsEqual(hazy_image, ideal_hazy_image));
  cv::Mat ideal_recovered_image(1, 1, CV_64FC3, cv::Scalar(19, 19, 19));
  cv::Mat recovered_image(1, 1, CV_64FC3);
  model.RecoverImage(recovered_image, hazy_image);
  CHECK(IsDoubleMatsEqual(recovered_image, ideal_recovered_image));
}
