project(image_loader)

add_library(ImageLoader image_loader.cpp image_loader.hpp)
target_link_libraries(ImageLoader ${OpenCV_LIBS})

add_executable(test_image_loader test_image_loader.cpp)
add_definitions(-DCSDIR=\"${CMAKE_CURRENT_SOURCE_DIR}\")
target_link_libraries(test_image_loader ImageLoader)

enable_testing()
add_test(NAME test_image_loader COMMAND test_image_loader)
