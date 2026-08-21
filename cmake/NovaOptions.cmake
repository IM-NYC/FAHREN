option(NOVA_BUILD_SHARED_LIBS
    "Build Novaflow as a shared library"
    ON)

option(NOVA_BUILD_TESTS
    "Build tests"
    ON)

option(NOVA_BUILD_EXAMPLES
    "Build example programs"
    OFF)

option(NOVA_BUILD_TOOLS
    "Build CLI tools"
    ON)

option(NOVA_ENABLE_VERBOSE
    "Enable verbose logging"
    OFF)

option(NOVA_ENABLE_OPENBLAS
    "Link OpenBLAS for CPU GEMM"
    OFF)

option(NOVA_ENABLE_OPENMP
    "Enable OpenMP parallelization"
    OFF)

option(NOVA_ENABLE_CUDA
    "Enable CUDA GPU acceleration (requires NVIDIA CUDA Toolkit)"
    OFF)

option(NOVA_INSTALL
    "Generate install rules"
    ON)

option(NOVA_WARNINGS_AS_ERRORS
    "Treat warnings as errors"
    OFF)

if(WIN32 AND MSVC)
    option(NOVA_MSVC_STATIC_RUNTIME
        "Use static MSVC runtime (/MT)"
        OFF)
endif()
