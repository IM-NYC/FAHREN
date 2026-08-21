add_executable(nova_test test.c)
add_executable(nova_mnist_test mnist_test.c)

target_link_libraries(nova_test PRIVATE Novaflow::nova)
target_link_libraries(nova_mnist_test PRIVATE Novaflow::nova)
nova_apply_compile_flags(nova_test)
nova_apply_compile_flags(nova_mnist_test)

target_compile_definitions(nova_test PRIVATE
    NOVA_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
target_compile_definitions(nova_mnist_test PRIVATE
    NOVA_MNIST_DIR="${CMAKE_SOURCE_DIR}/mnist"
)

if(WIN32 AND BUILD_SHARED_LIBS)
    foreach(_t nova_test nova_mnist_test)
        add_custom_command(TARGET ${_t} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:nova>
                $<TARGET_FILE_DIR:${_t}>
            COMMENT "Copy nova.dll"
        )
    endforeach()
endif()

add_test(NAME nova_smoke COMMAND $<TARGET_FILE:nova_test>
         WORKING_DIRECTORY $<TARGET_FILE_DIR:nova_test>)

add_test(NAME nova_mnist COMMAND $<TARGET_FILE:nova_mnist_test>
         WORKING_DIRECTORY $<TARGET_FILE_DIR:nova_mnist_test>)

set_tests_properties(nova_mnist PROPERTIES
    ENVIRONMENT "NOVA_MNIST_DIR=${CMAKE_SOURCE_DIR}/mnist"
)

if(CMAKE_CONFIGURATION_TYPES)
    add_custom_target(run_test
        COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --output-on-failure -R "nova_"
        DEPENDS nova_test nova_mnist_test
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
else()
    add_custom_target(run_test
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R "nova_"
        DEPENDS nova_test nova_mnist_test
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()
