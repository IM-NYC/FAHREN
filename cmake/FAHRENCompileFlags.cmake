# cmake/FAHRENCompileFlags.cmake — warning and optimization flags per toolchain

function(fahren_apply_compile_flags target_name)
    if(NOT TARGET ${target_name})
        message(FATAL_ERROR "fahren_apply_compile_flags: unknown target '${target_name}'")
    endif()

    set_target_properties(${target_name} PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)

    if(MSVC)
        target_compile_definitions(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:C>:_CRT_SECURE_NO_WARNINGS>
        )
        # MSVC /W4 and /permissive- must not be passed to nvcc (they look like input paths).
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:C>:/W4>
            $<$<COMPILE_LANGUAGE:C>:/permissive->
        )
        if(FAHREN_WARNINGS_AS_ERRORS)
            target_compile_options(${target_name} PRIVATE
                $<$<COMPILE_LANGUAGE:C>:/WX>
            )
        endif()
        if(FAHREN_MSVC_STATIC_RUNTIME)
            set_property(TARGET ${target_name} PROPERTY
                MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
        endif()
    else()
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:C>:-Wall>
            $<$<COMPILE_LANGUAGE:C>:-Wextra>
            $<$<COMPILE_LANGUAGE:C>:-Wpedantic>
        )
        if(FAHREN_WARNINGS_AS_ERRORS)
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
        $<$<BOOL:${FAHREN_ENABLE_VERBOSE}>:FAHREN_VERBOSE=1>
    )

    if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_CONFIGURATION_TYPES)
        target_compile_definitions(${target_name} PRIVATE FAHREN_DEBUG=1)
    endif()
endfunction()

function(fahren_apply_cuda_flags target_name)
    if(NOT TARGET ${target_name})
        message(FATAL_ERROR "fahren_apply_cuda_flags: unknown target '${target_name}'")
    endif()

    set_property(TARGET ${target_name} PROPERTY CUDA_SEPARABLE_COMPILATION OFF)

    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES native)
    endif()

    target_include_directories(${target_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/src
    )

    if(MSVC)
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc>
        )
    endif()
endfunction()
