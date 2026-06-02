# FAHREN — Fast and Readable Neural Network Library
# Cross-platform CMake (Windows, Linux, macOS)

cmake_minimum_required(VERSION 3.18)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(FAHRENOptions)

if(FAHREN_ENABLE_CUDA)
    project(Fahren
        VERSION 1.0.0
        DESCRIPTION "C neural network library for building and training models"
        LANGUAGES C CUDA
    )
else()
    project(Fahren
        VERSION 1.0.0
        DESCRIPTION "C neural network library for building and training models"
        LANGUAGES C
    )
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(FAHRENCompileFlags)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# Honor BUILD_SHARED_LIBS from option when set
if(FAHREN_BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
else()
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
endif()

add_subdirectory(src)

# ---------------------------------------------------------------------------
# Library target
# ---------------------------------------------------------------------------
add_library(fahren ${FAHREN_SOURCES})
add_library(Fahren::fahren ALIAS fahren)

set_target_properties(fahren PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME fahren
    POSITION_INDEPENDENT_CODE ON
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
)

# MSVC: shared builds need exported symbols or no import .lib is produced for consumers.
if(WIN32 AND BUILD_SHARED_LIBS)
    set_target_properties(fahren PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

target_include_directories(fahren
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

fahren_apply_compile_flags(fahren)

find_package(Threads REQUIRED)
target_link_libraries(fahren PUBLIC Threads::Threads)

if(UNIX AND NOT APPLE)
    target_link_libraries(fahren PUBLIC m)
elseif(UNIX AND APPLE)
    target_link_libraries(fahren PUBLIC m)
endif()

if(WIN32)
    target_link_libraries(fahren PRIVATE bcrypt)
endif()

if(FAHREN_ENABLE_CUDA)
    set(FAHREN_ENABLE_ADDONS ON CACHE BOOL "" FORCE)
endif()

if(FAHREN_ENABLE_OPENBLAS)
    find_package(OpenBLAS)
    if(OpenBLAS_FOUND)
        target_compile_definitions(fahren PRIVATE FAHREN_USE_OPENBLAS)
        target_link_libraries(fahren PRIVATE OpenBLAS::OpenBLAS)
        message(STATUS "FAHREN: OpenBLAS enabled")
    else()
        message(WARNING "FAHREN_ENABLE_OPENBLAS is ON but OpenBLAS was not found")
    endif()
endif()

if(FAHREN_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_C_FOUND)
        target_link_libraries(fahren PRIVATE OpenMP::OpenMP_C)
        message(STATUS "FAHREN: OpenMP enabled")
    endif()
endif()

# Optional add-ons hook (CUDA, etc.)
if(FAHREN_ENABLE_ADDONS)
    include(Addons)
endif()

if(FAHREN_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(FAHREN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# ---------------------------------------------------------------------------
# Examples (future)
# ---------------------------------------------------------------------------
if(FAHREN_BUILD_EXAMPLES AND EXISTS "${CMAKE_SOURCE_DIR}/examples/CMakeLists.txt")
    add_subdirectory(examples)
endif()

# ---------------------------------------------------------------------------
# Install + package config (for find_package(Fahren))
# ---------------------------------------------------------------------------
if(FAHREN_INSTALL)
    install(TARGETS fahren
        EXPORT FahrenTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    install(DIRECTORY include/fahren
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h"
    )
    install(FILES docs/weights-format.md DESTINATION ${CMAKE_INSTALL_DOCDIR} OPTIONAL)

    install(EXPORT FahrenTargets
        FILE FahrenTargets.cmake
        NAMESPACE Fahren::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Fahren
    )

    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FahrenConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/FahrenConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Fahren
    )

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/FahrenConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/FahrenConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/FahrenConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Fahren
    )
endif()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "FAHREN ${PROJECT_VERSION} configuration:")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  Shared libs:    ${BUILD_SHARED_LIBS}")
message(STATUS "  Tests:          ${FAHREN_BUILD_TESTS}")
message(STATUS "  Verbose API:    ${FAHREN_ENABLE_VERBOSE}")
message(STATUS "  Add-ons:        ${FAHREN_ENABLE_ADDONS}")
message(STATUS "  CUDA:           ${FAHREN_ENABLE_CUDA}")
message(STATUS "  OpenBLAS:       ${FAHREN_ENABLE_OPENBLAS}")
message(STATUS "  Install rules:  ${FAHREN_INSTALL}")
if(WIN32)
    message(STATUS "  Platform:       Windows")
elseif(APPLE)
    message(STATUS "  Platform:       macOS")
else()
    message(STATUS "  Platform:       Unix")
endif()
message(STATUS "")
