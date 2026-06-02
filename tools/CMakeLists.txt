add_executable(fahren_cli fahren_cli.c)
target_link_libraries(fahren_cli PRIVATE Fahren::fahren)
fahren_apply_compile_flags(fahren_cli)

if(WIN32 AND BUILD_SHARED_LIBS)
    add_custom_command(TARGET fahren_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:fahren>
            $<TARGET_FILE_DIR:fahren_cli>
    )
endif()

install(TARGETS fahren_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
