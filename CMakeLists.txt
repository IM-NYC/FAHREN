# CMakeLists.txt (root)

cmake_minimum_required(VERSION 3.15)
project(Fahren C)

# Optional compilation flags
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O2")

# Optionally allow users to toggle static/shared library
option(BUILD_SHARED_LIBS "Build shared library (dylib/so)" ON)

# Include src CMake to get source files
add_subdirectory(src)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Create the library (shared or static)
add_library(${PROJECT_NAME} ${FAHREN_SOURCES})

# Set library versioning (optional)
set_target_properties(${PROJECT_NAME} PROPERTIES
    VERSION 1.0
    SOVERSION 1
)

# Optional linking: e.g., pthread
find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)

# Add test executable
add_executable(${PROJECT_NAME}_test test/test_write_weights.c)

# Link the library to test
target_link_libraries(${PROJECT_NAME}_test PRIVATE ${PROJECT_NAME})

# Optional: run test after build
add_custom_target(run_test
    COMMAND ${PROJECT_NAME}_test
    DEPENDS ${PROJECT_NAME}_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Quick-start example
add_executable(quick_start examples/quick_start.c)
target_include_directories(quick_start PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(quick_start PRIVATE ${PROJECT_NAME})

add_custom_target(quick_start_run
    COMMAND quick_start
    DEPENDS quick_start
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
