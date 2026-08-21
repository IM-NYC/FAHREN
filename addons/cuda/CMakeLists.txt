find_package(CUDAToolkit REQUIRED)

target_sources(nova PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cuda_train.cu)

target_compile_definitions(nova PRIVATE NOVA_ENABLE_CUDA)

target_link_libraries(nova PRIVATE CUDA::cudart CUDA::cublas)

target_include_directories(nova PRIVATE ${CMAKE_SOURCE_DIR}/src)
