# cmake/FAHRENOptions.cmake — user-configurable build switches

option(FAHREN_BUILD_SHARED_LIBS
    "Build FAHREN as a shared library (DLL on Windows)"
    ON)

option(FAHREN_BUILD_TESTS
    "Build and register the fahren_test executable with CTest"
    ON)

option(FAHREN_BUILD_EXAMPLES
    "Build example programs (reserved for future examples/)"
    OFF)

option(FAHREN_BUILD_TOOLS
    "Build fahren_cli command-line runner"
    ON)

option(FAHREN_ENABLE_VERBOSE
    "Compile with FAHREN_VERBOSE=1 for extra library logging"
    OFF)

option(FAHREN_ENABLE_ADDONS
    "Include optional add-on modules from cmake/Addons.cmake"
    OFF)

option(FAHREN_ENABLE_CUDA
    "Build CUDA add-on (requires CUDA Toolkit and enables FAHREN_ENABLE_ADDONS)"
    OFF)

option(FAHREN_ENABLE_OPENBLAS
    "Link OpenBLAS for CPU GEMM (optional)"
    OFF)

option(FAHREN_ENABLE_OPENMP
    "Enable OpenMP parallelization in training loops"
    OFF)

option(FAHREN_INSTALL
    "Generate install() rules for the library and headers"
    ON)

option(FAHREN_WARNINGS_AS_ERRORS
    "Treat compiler warnings as errors"
    OFF)

# MSVC: static CRT vs DLL CRT (/MT vs /MD) — leave default unless user sets CMAKE_MSVC_RUNTIME_LIBRARY
if(WIN32 AND MSVC)
    option(FAHREN_MSVC_STATIC_RUNTIME
        "Link against the static MSVC runtime (/MT)"
        OFF)
endif()
