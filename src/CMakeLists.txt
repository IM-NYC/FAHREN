# src/CMakeLists.txt

# Minimum required CMake version
cmake_minimum_required(VERSION 3.15)

# Create a list for all sources
set(FAHREN_SOURCES)

# Platform-specific sources
if(UNIX)
    message(STATUS "Adding POSIX sources")
    list(APPEND FAHREN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/posix.c)
endif()

if(WIN32)
    message(FATAL_ERROR "Windows OS is not supported")
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(STATUS "Building for 64-bit")
else()
    message(STATUS "Building for 32-bit")
endif()

# Export variables to parent CMakeLists
set(FAHREN_SOURCES ${FAHREN_SOURCES} PARENT_SCOPE)
set(FAHREN_INCLUDE_DIRS ${FAHREN_INCLUDE_DIRS} PARENT_SCOPE)
