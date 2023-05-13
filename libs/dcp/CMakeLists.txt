project(dcp)

add_library(DarkChannelPrior dcp.hpp dcp.cpp)
target_link_libraries(DarkChannelPrior ${OpenCV_LIBS})

add_executable(test_dcp test_dcp.cpp)
target_link_libraries(test_dcp ${OpenCV_LIBS} DarkChannelPrior)

enable_testing()
add_test(NAME test_dcp COMMAND test_dcp)
