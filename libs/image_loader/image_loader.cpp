#include <algorithm>
#include <fstream>
#include <image_loader.hpp>
#include <iterator>
#include <opencv2/imgcodecs.hpp>
#include <vector>

namespace fs = std::filesystem;

namespace load {

bool PathWrapper::operator<(const PathWrapper& rhs) const {
  return std::lexicographical_compare(name.begin(), name.end(),
                                      rhs.name.begin(), rhs.name.end());
}

void PathWrapper::UpdateName() { name = path.filename().u8string(); }

std::vector<PathWrapper> LoadDir(const PathWrapper& path) {
  if (!fs::exists(path.path))
    throw std::runtime_error("LoadDir(...): path doesn't exist");
  if (!fs::is_directory(path.path))
    throw std::runtime_error("LoadDir(...): path isn't a dir");
  std::vector<PathWrapper> result;
  for (auto const& dir_entry : fs::directory_iterator{path.path}) {
    if (fs::is_directory(dir_entry))
      throw std::runtime_error("LoadDir(...): subpath isn't a file");
    result.emplace_back();
    result.back().path = dir_entry;
    result.back().UpdateName();
  }
  std::stable_sort(result.begin(), result.end());
  return result;
}

cv::Mat LoadImg(const PathWrapper& path) {
  cv::Mat result = cv::imread(path.ToString());
  if (result.empty()) {
    result = LoadImgUTF8(path);
  }
  if (result.empty()) {
    throw std::runtime_error("LoadImg(...): a path isn't Unicode");
  }
  // converting img to right format
  cv::Mat right_result;
  result.convertTo(right_result, CV_64FC3, 1.0 / 255.0);
  return right_result;
}

cv::Mat LoadImgUTF8(const PathWrapper& path) {
  std::ifstream file(path.path, std::ios::binary);
  file.unsetf(std::ios::skipws);

  std::streampos size;
  file.seekg(0, std::ios::end);
  size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<unsigned char> img_vec;
  img_vec.reserve(size);
  img_vec.insert(img_vec.begin(), std::istream_iterator<unsigned char>(file),
                 std::istream_iterator<unsigned char>());
  return cv::imdecode(img_vec, cv::IMREAD_COLOR);
}

}  // namespace load
