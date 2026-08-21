# Novaflow — Neural Network Framework
# Cross-platform CMake (Windows, Linux, macOS)

cmake_minimum_required(VERSION 3.18)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(NovaOptions)

project(Novaflow
    VERSION 1.0.0
    DESCRIPTION "C neural network framework for building and training models"
    LANGUAGES C
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(NovaCompileFlags)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

if(NOVA_BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
else()
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
endif()

add_subdirectory(src)

add_library(nova ${NOVA_SOURCES})
add_library(Novaflow::nova ALIAS nova)

set_target_properties(nova PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME nova
    POSITION_INDEPENDENT_CODE ON
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
)

if(WIN32 AND BUILD_SHARED_LIBS)
    set_target_properties(nova PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

target_include_directories(nova
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

nova_apply_compile_flags(nova)

find_package(Threads REQUIRED)
target_link_libraries(nova PUBLIC Threads::Threads)

if(UNIX AND NOT APPLE)
    target_link_libraries(nova PUBLIC m)
elseif(UNIX AND APPLE)
    target_link_libraries(nova PUBLIC m)
endif()

if(WIN32)
    target_link_libraries(nova PRIVATE bcrypt)
endif()

if(NOVA_ENABLE_OPENBLAS)
    find_package(OpenBLAS)
    if(OpenBLAS_FOUND)
        target_compile_definitions(nova PRIVATE NOVA_USE_OPENBLAS)
        target_link_libraries(nova PRIVATE OpenBLAS::OpenBLAS)
        message(STATUS "Novaflow: OpenBLAS enabled")
    else()
        message(WARNING "NOVA_ENABLE_OPENBLAS is ON but OpenBLAS was not found")
    endif()
endif()

if(NOVA_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_C_FOUND)
        target_link_libraries(nova PRIVATE OpenMP::OpenMP_C)
        message(STATUS "Novaflow: OpenMP enabled")
    endif()
endif()

if(NOVA_ENABLE_CUDA)
    enable_language(CUDA)
    add_subdirectory(addons/cuda)
    message(STATUS "Novaflow: CUDA enabled")
endif()

if(NOVA_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if(NOVA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(NOVA_BUILD_EXAMPLES AND EXISTS "${CMAKE_SOURCE_DIR}/examples/CMakeLists.txt")
    add_subdirectory(examples)
endif()

if(NOVA_INSTALL)
    install(TARGETS nova
        EXPORT NovaTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    install(DIRECTORY include/nova
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h"
    )

    install(EXPORT NovaTargets
        FILE NovaflowTargets.cmake
        NAMESPACE Novaflow::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Novaflow
    )

    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/NovaConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/NovaflowConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Novaflow
    )

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/NovaflowConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/NovaflowConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/NovaflowConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Novaflow
    )
endif()

message(STATUS "")
message(STATUS "Novaflow ${PROJECT_VERSION} configuration:")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  Shared libs:    ${BUILD_SHARED_LIBS}")
message(STATUS "  Tests:          ${NOVA_BUILD_TESTS}")
message(STATUS "  Tools:          ${NOVA_BUILD_TOOLS}")
message(STATUS "  OpenBLAS:       ${NOVA_ENABLE_OPENBLAS}")
message(STATUS "  OpenMP:         ${NOVA_ENABLE_OPENMP}")
message(STATUS "  CUDA:           ${NOVA_ENABLE_CUDA}")
message(STATUS "  Install rules:  ${NOVA_INSTALL}")
if(WIN32)
    message(STATUS "  Platform:       Windows")
elseif(APPLE)
    message(STATUS "  Platform:       macOS")
else()
    message(STATUS "  Platform:       Unix")
endif()
message(STATUS "")
