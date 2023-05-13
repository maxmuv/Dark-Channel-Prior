project(executor)

add_library(Executor executor.hpp executor.cpp)
target_link_libraries(Executor HazeModel ImageLoader DarkChannelPrior)

add_executable(test_executor test_executor.cpp)
target_link_libraries(test_executor Executor)

enable_testing()
add_test(NAME test_executor COMMAND test_executor)
