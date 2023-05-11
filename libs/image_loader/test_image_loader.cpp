#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include "image_loader.hpp"

TEST_CASE("ImageLoader") {
  load::PathWrapper input(std::string(CSDIR) + "/test_dir");
  std::vector<load::PathWrapper> result;
  result = load::LoadDir(input);
  CHECK_EQ(result.size(), 3);
  std::vector<std::string> names({"1.txt", "2.txt", "3.txt"});
  for (int i = 0; i < 3; ++i) CHECK_EQ(names[i], result[i].name);
}
