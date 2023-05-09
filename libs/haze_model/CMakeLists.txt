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
