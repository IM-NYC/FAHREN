set(NOVA_CORE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/init.c
    ${CMAKE_CURRENT_SOURCE_DIR}/train.c
    ${CMAKE_CURRENT_SOURCE_DIR}/weights.c
    ${CMAKE_CURRENT_SOURCE_DIR}/blas.c
    ${CMAKE_CURRENT_SOURCE_DIR}/device.c
    ${CMAKE_CURRENT_SOURCE_DIR}/easy.c
    ${CMAKE_CURRENT_SOURCE_DIR}/mnist.c
    ${CMAKE_CURRENT_SOURCE_DIR}/errors.c
    ${CMAKE_CURRENT_SOURCE_DIR}/hash.c
    ${CMAKE_CURRENT_SOURCE_DIR}/quantization.c
    ${CMAKE_CURRENT_SOURCE_DIR}/backend.c
    ${CMAKE_CURRENT_SOURCE_DIR}/storage.c
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizers.c
)

if(WIN32)
    message(STATUS "Novaflow: using Windows platform backend (win32.c)")
    list(APPEND NOVA_CORE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/win32.c)
elseif(UNIX)
    message(STATUS "Novaflow: using POSIX platform backend (posix.c)")
    list(APPEND NOVA_CORE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/posix.c)
else()
    message(WARNING "Novaflow: unknown platform; using POSIX backend as fallback")
    list(APPEND NOVA_CORE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/posix.c)
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(STATUS "Novaflow: 64-bit target")
else()
    message(STATUS "Novaflow: 32-bit target")
endif()

set(NOVA_SOURCES ${NOVA_CORE_SOURCES} PARENT_SCOPE)
