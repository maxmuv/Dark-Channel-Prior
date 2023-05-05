cmake_minimum_required(VERSION 3.12)

project(DarkChannelPrior)

set_property(GLOBAL PROPERTY CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

if (MSVC)
	add_compile_options(/W4)
else()
        add_compile_options(-Wall -Wextra -Wpedantic -Wno-c11-extensions)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/bin)

find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(
	libs
	apps
	thirdparty
)

add_subdirectory(libs)
add_subdirectory(apps)
add_subdirectory(thirdparty)

