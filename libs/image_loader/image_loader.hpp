#pragma once
#ifndef IMAGE_LOADER_HPP
#define IMAGE_LOADER_HPP

#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

namespace load {
struct PathWrapper {
  PathWrapper() = default;
  PathWrapper(const std::string& str) : path(str) {}
  void UpdateName();
  const std::string ToString() const { return static_cast<std::string>(path); }
  fs::path path;
  std::string name;
  bool operator<(const PathWrapper& rhs) const;
};

std::vector<PathWrapper> LoadDir(const PathWrapper& path);

}  // namespace load

#endif  // IMAGE_LOADER_HPP