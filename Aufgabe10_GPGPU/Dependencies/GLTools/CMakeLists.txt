project(GLTools)
ADD_DEFINITIONS(-D_CRT_SECURE_NO_WARNINGS)
include_directories(include ${GLEW_INCLUDE_DIR})
FILE(GLOB Sources "src/*.cpp" )
FILE(GLOB Headers "include/*.h")

add_library(GLTools ${Sources}  ${Headers} )
target_link_libraries(GLTools glew)
SET_PROPERTY(TARGET GLTools         PROPERTY FOLDER "Libraries")

set(GLTools_INCLUDE_DIR "${GLTools_SOURCE_DIR}/include" CACHE PATH "" FORCE)
set(GLTools_LIBRARY_DBG GLTools CACHE STRING "" FORCE)
set(GLTools_LIBRARY_REL GLTools CACHE STRING "" FORCE)
mark_as_advanced(GLTools_INCLUDE_DIR GLTools_LIBRARY_DBG GLTools_LIBRARY_REL)
