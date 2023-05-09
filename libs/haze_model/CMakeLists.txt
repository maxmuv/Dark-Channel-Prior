project(haze_model)
add_library(HazeModel 
        haze_model.hpp
        haze_model.cpp
)
target_link_libraries(HazeModel ${OpenCV_LIBS})

add_executable(test_haze_model test_haze_model.cpp)
target_link_libraries(test_haze_model HazeModel ${OpenCV_LIBS})

add_executable(test_transmission_creation test_transmission_creation.cpp)
target_link_libraries(test_transmission_creation HazeModel ${OpenCV_LIBS})

add_executable(test_sample test_sample.cpp)
add_definitions(-DCSDIR=\"${CMAKE_CURRENT_SOURCE_DIR}\")
target_link_libraries(test_sample HazeModel ${OpenCV_LIBS})
