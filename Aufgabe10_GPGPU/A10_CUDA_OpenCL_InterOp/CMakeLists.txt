
SET(PROJECT_NAME A10_CUDA_OpenCL_InterOp)

set(SOURCES main.cpp cudaOpenCV.cu)
set(LIBRARIES freeglut GLTools AntTweakBar opencv_core340d opencv_highgui340d opencv_imgproc340d opencv_imgcodecs340d opencv_video340d opencv_videoio340d)

project(${PROJECT_NAME})

INCLUDE_DIRECTORIES(${GPGPU_SOURCE_DIR} ${GPGPU_SOURCE_DIR}/Dependencies/OpenCV/include ${GPGPU_SOURCE_DIR}/Dependencies/CUDASamples/common/inc)
LINK_DIRECTORIES(${GPGPU_SOURCE_DIR}/Dependencies/OpenCV/lib/Debug)


add_executable(${PROJECT_NAME} ${SOURCES})
create_target_launcher(${PROJECT_NAME}  WORKING_DIRECTORY ${${PROJECT_NAME}_SOURCE_DIR})

target_link_libraries(${PROJECT_NAME} ${LIBRARIES})
SET_PROPERTY(TARGET ${PROJECT_NAME} PROPERTY FOLDER "Aufgaben")
install_sample(${PROJECT_NAME} ${SOURCES})
