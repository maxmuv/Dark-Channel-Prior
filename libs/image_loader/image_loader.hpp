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
  fs::path path;
  std::string name;
  bool operator<(const PathWrapper& rhs) const;
};

cv::Mat LoadImg(const PathWrapper& path);

cv::Mat LoadImgUTF8(const PathWrapper& path);

std::vector<PathWrapper> LoadDir(const PathWrapper& path);

}  // namespace load

#endif  // IMAGE_LOADER_HPP
