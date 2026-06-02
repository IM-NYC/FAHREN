find_package(CUDAToolkit REQUIRED)

target_sources(fahren PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cuda_train.cu)

target_compile_definitions(fahren PRIVATE FAHREN_ENABLE_CUDA)

target_link_libraries(fahren PRIVATE CUDA::cudart CUDA::cublas)

fahren_apply_cuda_flags(fahren)

message(STATUS "FAHREN: CUDA training enabled (cuBLAS forward, CPU backward)")
