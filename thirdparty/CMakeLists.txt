project(thirdparty)

add_library(doctest doctest.h)
set_target_properties(doctest PROPERTIES LINKER_LANGUAGE CXX)
