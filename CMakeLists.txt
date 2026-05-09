# CMakeLists.txt - FAHREN Neural Network Library

cmake_minimum_required(VERSION 3.15)
project(Fahren C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O2")

option(BUILD_SHARED_LIBS "Build shared library (dylib/so)" ON)

add_subdirectory(src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Main FAHREN library
add_library(${PROJECT_NAME} ${FAHREN_SOURCES})
set_target_properties(${PROJECT_NAME} PROPERTIES
    VERSION 1.0
    SOVERSION 1
)

# Threading support
find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)

# Test executable
if(EXISTS ${CMAKE_SOURCE_DIR}/test/test.c)
    add_executable(${PROJECT_NAME}_test test/test.c)
    target_link_libraries(${PROJECT_NAME}_test PRIVATE ${PROJECT_NAME})

    add_custom_target(run_test
        COMMAND ${PROJECT_NAME}_test
        DEPENDS ${PROJECT_NAME}_test
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endif()
target_link_libraries(${PROJECT_NAME}_test PRIVATE ${PROJECT_NAME})

# Test target
add_custom_target(run_test
    COMMAND ${PROJECT_NAME}_test
    DEPENDS ${PROJECT_NAME}_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)