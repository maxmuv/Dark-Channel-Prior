#pragma once
#ifndef IMAGE_LOADER_HPP
#define IMAGE_LOADER_HPP

#include <filesystem>
#include <opencv2/core/mat.hpp>
#include <vector>

namespace fs = std::filesystem;

namespace load {
struct PathWrapper {
  PathWrapper() = default;
  PathWrapper(const std::string& str) : path(str) {}
  void UpdateName();
  const std::string ToString() const {
    return static_cast<std::string>(path.u8string());
  }
  bool Empty() const {
    int i = 0;
    if (!fs::exists(path))
      throw std::runtime_error("Empty(...): path doesn't exist");
    if (!fs::is_directory(path))
      throw std::runtime_error("Empty(...): path isn't a dir");
    for (auto const& dir_entry : fs::directory_iterator{path}) ++i;
    if (i > 0) return false;
    return true;
  }
  fs::path path;
  std::string name;
  bool operator<(const PathWrapper& rhs) const;
};

cv::Mat LoadImg(const PathWrapper& path);

cv::Mat LoadImgUTF8(const PathWrapper& path);

std::vector<PathWrapper> LoadDir(const PathWrapper& path);

}  // namespace load

#endif  // IMAGE_LOADER_HPP
