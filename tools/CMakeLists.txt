add_executable(nova_cli nova_cli.c)
target_link_libraries(nova_cli PRIVATE Novaflow::nova)
nova_apply_compile_flags(nova_cli)

if(WIN32 AND BUILD_SHARED_LIBS)
    add_custom_command(TARGET nova_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:nova>
            $<TARGET_FILE_DIR:nova_cli>
    )
endif()

install(TARGETS nova_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
