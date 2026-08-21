#----------------------------------------------------------------
# Generated CMake target import file for configuration "RelWithDebInfo".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Novaflow::nova" for configuration "RelWithDebInfo"
set_property(TARGET Novaflow::nova APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(Novaflow::nova PROPERTIES
  IMPORTED_IMPLIB_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib/nova.lib"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/bin/nova.dll"
  )

list(APPEND _cmake_import_check_targets Novaflow::nova )
list(APPEND _cmake_import_check_files_for_Novaflow::nova "${_IMPORT_PREFIX}/lib/nova.lib" "${_IMPORT_PREFIX}/bin/nova.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
