function(nova_apply_compile_flags target_name)
    if(NOT TARGET ${target_name})
        message(FATAL_ERROR "nova_apply_compile_flags: unknown target '${target_name}'")
    endif()

    set_target_properties(${target_name} PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)

    if(MSVC)
        target_compile_definitions(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:C>:_CRT_SECURE_NO_WARNINGS>
        )
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:C>:/W4>
            $<$<COMPILE_LANGUAGE:C>:/permissive->
        )
        if(NOVA_WARNINGS_AS_ERRORS)
            target_compile_options(${target_name} PRIVATE
                $<$<COMPILE_LANGUAGE:C>:/WX>
            )
        endif()
        if(NOVA_MSVC_STATIC_RUNTIME)
            set_property(TARGET ${target_name} PROPERTY
                MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
        endif()
    else()
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:C>:-Wall>
            $<$<COMPILE_LANGUAGE:C>:-Wextra>
            $<$<COMPILE_LANGUAGE:C>:-Wpedantic>
        )
        if(NOVA_WARNINGS_AS_ERRORS)
            target_compile_options(${target_name} PRIVATE
                $<$<COMPILE_LANGUAGE:C>:-Werror>
            )
        endif()
        if(NOT WIN32)
            target_compile_options(${target_name} PRIVATE
                $<$<COMPILE_LANGUAGE:C>:-fPIC>
            )
        endif()
    endif()

    target_compile_definitions(${target_name} PRIVATE
        $<$<BOOL:${NOVA_ENABLE_VERBOSE}>:NOVA_VERBOSE=1>
    )

    if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_CONFIGURATION_TYPES)
        target_compile_definitions(${target_name} PRIVATE NOVA_DEBUG=1)
    endif()
endfunction()
