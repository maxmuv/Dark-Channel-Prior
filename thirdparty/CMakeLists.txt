project(thirdparty)

add_library(doctest doctest.h)
set_target_properties(doctest PROPERTIES LINKER_LANGUAGE CXX)

add_library(argparse argparse.hpp)
set_target_properties(argparse PROPERTIES LINKER_LANGUAGE CXX)
