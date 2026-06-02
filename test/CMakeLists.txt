add_executable(fahren_test test.c)
add_executable(fahren_mnist_test mnist_test.c)

target_link_libraries(fahren_test PRIVATE Fahren::fahren)
target_link_libraries(fahren_mnist_test PRIVATE Fahren::fahren)
fahren_apply_compile_flags(fahren_test)
fahren_apply_compile_flags(fahren_mnist_test)

target_compile_definitions(fahren_mnist_test PRIVATE
    FAHREN_MNIST_DIR="${CMAKE_SOURCE_DIR}/mnist"
)

if(WIN32 AND BUILD_SHARED_LIBS)
    foreach(_t fahren_test fahren_mnist_test)
        add_custom_command(TARGET ${_t} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:fahren>
                $<TARGET_FILE_DIR:${_t}>
            COMMENT "Copy fahren.dll"
        )
    endforeach()
endif()

add_test(NAME fahren_smoke COMMAND $<TARGET_FILE:fahren_test>
         WORKING_DIRECTORY $<TARGET_FILE_DIR:fahren_test>)

add_test(NAME fahren_mnist COMMAND $<TARGET_FILE:fahren_mnist_test>
         WORKING_DIRECTORY $<TARGET_FILE_DIR:fahren_mnist_test>)

set_tests_properties(fahren_mnist PROPERTIES
    ENVIRONMENT "FAHREN_MNIST_DIR=${CMAKE_SOURCE_DIR}/mnist"
)

if(FAHREN_ENABLE_CUDA)
    add_executable(fahren_mnist_cuda_test mnist_cuda_test.c)
    target_link_libraries(fahren_mnist_cuda_test PRIVATE Fahren::fahren)
    fahren_apply_compile_flags(fahren_mnist_cuda_test)
    target_compile_definitions(fahren_mnist_cuda_test PRIVATE
        FAHREN_MNIST_DIR="${CMAKE_SOURCE_DIR}/mnist"
    )
    if(WIN32 AND BUILD_SHARED_LIBS)
        add_custom_command(TARGET fahren_mnist_cuda_test POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:fahren>
                $<TARGET_FILE_DIR:fahren_mnist_cuda_test>
        )
    endif()
    add_test(NAME fahren_mnist_cuda COMMAND $<TARGET_FILE:fahren_mnist_cuda_test>
             WORKING_DIRECTORY $<TARGET_FILE_DIR:fahren_mnist_cuda_test>)
    set_tests_properties(fahren_mnist_cuda PROPERTIES
        ENVIRONMENT "FAHREN_MNIST_DIR=${CMAKE_SOURCE_DIR}/mnist"
    )
endif()

if(CMAKE_CONFIGURATION_TYPES)
    add_custom_target(run_test
        COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --output-on-failure -R "fahren_"
        DEPENDS fahren_test fahren_mnist_test
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
else()
    add_custom_target(run_test
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R "fahren_"
        DEPENDS fahren_test fahren_mnist_test
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()
