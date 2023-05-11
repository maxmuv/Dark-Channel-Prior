#include <argparse.hpp>
#include <executor/executor.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
  argparse::ArgumentParser program("HazeMachine");
  program.add_argument("output_dir").help("empty output dir").nargs(1);
  program.add_argument("input_dirs")
      .help("gets one image directory to dehaze or two to augment")
      .nargs(1, 2);
  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }
  try {
    auto output = program.get<std::string>("output_dir");
    auto input = program.get<std::vector<std::string>>("input_dirs");
    exec::Produce(input, output);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    return 1;
  }
}
