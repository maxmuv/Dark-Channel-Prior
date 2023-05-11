#include <algorithm>
#include <image_loader.hpp>

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

}  // namespace load
