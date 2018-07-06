cmake_minimum_required(VERSION 2.6)

if (CMAKE_MAJOR_VERSION GREATER 2)
	cmake_policy(SET CMP0026 OLD)
	if (CMAKE_MINOR_VERSION GREATER 0)
		cmake_policy(SET CMP0053 OLD)
	endif (CMAKE_MINOR_VERSION GREATER 0)
endif (CMAKE_MAJOR_VERSION GREATER 2)

project(GPGPU LANGUAGES CXX CUDA)

if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
   message(SEND_ERROR "In-source builds are not allowed.")
endif("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_MODULE_PATH 
  "${GPGPU_SOURCE_DIR}/cmake"
)

#change the default installation path
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${GPGPU_SOURCE_DIR}/Abgabe" CACHE PATH "GPGPU install prefix" FORCE)
endif (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

set(CPACK_PACKAGE_INSTALL_DIRECTORY "${CMAKE_INSTALL_PREFIX}")
set(CPACK_GENERATOR "ZIP")
include(Dependencies)
include(Install)
include(CreateLaunchers)

add_definitions(/DFREEIMAGE_LIB /DTW_STATIC)
include_directories(${GPGPU_SOURCE_DIR})
#disable win32 warning
add_definitions(/D_CRT_SECURE_NO_WARNINGS -DFREEGLUT_LIB_PRAGMAS=0 -DFREEGLUT_STATIC /DTW_STATIC /DTW_NO_LIB_PRAGMA)

add_subdirectory(ImageLoader)



# Die Projekte hinzufügen
add_subdirectory(A10_CUDA_OpenCL_InterOp)

install(DIRECTORY Texturen DESTINATION .)

include(CPack)