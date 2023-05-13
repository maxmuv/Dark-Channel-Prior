project(libs)
include_directories(
        haze_model
        executor
        image_loader
        dcp
)

add_subdirectory(haze_model)
add_subdirectory(executor)
add_subdirectory(image_loader)
add_subdirectory(dcp)

enable_testing()
