#include <executor/executor.hpp>
#include <iostream>

std::vector<std::string> ParseArgs(int argc, char* argv[]) {
  std::string help_message(
      "HazeMachine <output_dir> <input_dirs> [1..2]\n\n"
      "Positional arguments:\n"
      "\toutput_dir   	empty output dir\n"
      "\tinput_dirs   	gets one image directory to dehaze or two to augment"
      "[nargs=1..2] \n");
  if (argc <= 2 || argc > 4) throw std::runtime_error(help_message);
  std::vector<std::string> args;
  for (int i = 1; i < argc; ++i) {
    args.emplace_back(argv[i]);
  }
  return args;
}

int main(int argc, char* argv[]) {
  std::vector<std::string> args;
  try {
    args = ParseArgs(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    return 1;
  }
  try {
    auto output = args.front();
    std::vector<std::string> input;
    for (size_t i = 1; i < args.size(); ++i) input.push_back(args[i]);
    exec::Produce(input, output);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    return 1;
  }
}
