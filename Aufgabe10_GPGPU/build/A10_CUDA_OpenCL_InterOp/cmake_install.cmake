# Install script for directory: H:/gpgpu/GPGPU/A10_CUDA_OpenCL_InterOp

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "H:/gpgpu/GPGPU/Abgabe")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/A10_CUDA_OpenCL_InterOp" TYPE EXECUTABLE FILES "H:/gpgpu/GPGPU/build/A10_CUDA_OpenCL_InterOp/Debug/A10_CUDA_OpenCL_InterOp.exe")
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/A10_CUDA_OpenCL_InterOp" TYPE EXECUTABLE FILES "H:/gpgpu/GPGPU/build/A10_CUDA_OpenCL_InterOp/Release/A10_CUDA_OpenCL_InterOp.exe")
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "A10_CUDA_OpenCL_InterOp" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/A10_CUDA_OpenCL_InterOp" TYPE FILE FILES
    "H:/gpgpu/GPGPU/A10_CUDA_OpenCL_InterOp/main.cpp"
    "H:/gpgpu/GPGPU/A10_CUDA_OpenCL_InterOp/cudaOpenCV.cu"
    )
endif()

