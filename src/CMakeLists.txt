# src/CMakeLists.txt — FAHREN core library sources

set(FAHREN_CORE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/init.c
    ${CMAKE_CURRENT_SOURCE_DIR}/train.c
    ${CMAKE_CURRENT_SOURCE_DIR}/weights.c
    ${CMAKE_CURRENT_SOURCE_DIR}/blas.c
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizers.c
    ${CMAKE_CURRENT_SOURCE_DIR}/device.c
    ${CMAKE_CURRENT_SOURCE_DIR}/easy.c
    ${CMAKE_CURRENT_SOURCE_DIR}/mnist.c
    ${CMAKE_CURRENT_SOURCE_DIR}/errors.c
)

if(WIN32)
    message(STATUS "FAHREN: using Windows platform backend (win32.c)")
    list(APPEND FAHREN_CORE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/win32.c)
elseif(UNIX)
    message(STATUS "FAHREN: using POSIX platform backend (posix.c)")
    list(APPEND FAHREN_CORE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/posix.c)
else()
    message(WARNING "FAHREN: unknown platform; using POSIX backend as fallback")
    list(APPEND FAHREN_CORE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/posix.c)
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(STATUS "FAHREN: 64-bit target")
else()
    message(STATUS "FAHREN: 32-bit target")
endif()

set(FAHREN_SOURCES ${FAHREN_CORE_SOURCES} PARENT_SCOPE)
