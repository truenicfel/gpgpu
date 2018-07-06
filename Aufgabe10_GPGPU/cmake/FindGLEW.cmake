#-------------------------------------------------------------------
# This file is part of the CMake build system for OGRE
#     (Object-oriented Graphics Rendering Engine)
# For the latest info, see http://www.ogre3d.org/
#
# The contents of this file are placed in the public domain. Feel
# free to make use of it in any way you like.
#-------------------------------------------------------------------

# - Try to find GLEW
# Once done, this will define
#
#  GLEW_FOUND - system has GLEW
#  GLEW_INCLUDE_DIRS - the GLEW include directories 
#  GLEW_LIBRARIES - link these to use GLEW

include(FindPkgMacros)
findpkg_begin(GLEW)

# Get path, convert backslashes as ${ENV_${var}}
getenv_path(GLEW_HOME)

# construct search paths
set(GLEW_PREFIX_PATH ${GLEW_HOME} ${ENV_GLEW_HOME} )
create_search_paths(GLEW)
# redo search if prefix path changed
clear_if_changed(GLEW_PREFIX_PATH
  GLEW_LIBRARY_FWK
  GLEW_LIBRARY_REL
  GLEW_LIBRARY_DBG
  GLEW_INCLUDE_DIR
)

set(GLEW_LIBRARY_NAMES glew)
get_debug_names(GLEW_LIBRARY_NAMES)

use_pkgconfig(GLEW_PKGC GLEW)

findpkg_framework(GLEW)

find_path(GLEW_INCLUDE_DIR NAMES GL/glew.h HINTS ${GLEW_INC_SEARCH_PATH} ${GLEW_PKGC_INCLUDE_DIRS})

if (SYMBIAN) 
set(ORIGINAL_CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH})
set(CMAKE_PREFIX_PATH ${CMAKE_SYSYEM_OUT_DIR})
message(STATUS "Lib will be searched in Symbian out dir: ${CMAKE_SYSYEM_OUT_DIR}")
endif (SYMBIAN)
find_library(GLEW_LIBRARY_REL NAMES ${GLEW_LIBRARY_NAMES} HINTS ${GLEW_LIB_SEARCH_PATH} ${GLEW_PKGC_LIBRARY_DIRS} PATH_SUFFIXES "" release relwithdebinfo minsizerel)
find_library(GLEW_LIBRARY_DBG NAMES ${GLEW_LIBRARY_NAMES_DBG} HINTS ${GLEW_LIB_SEARCH_PATH} ${GLEW_PKGC_LIBRARY_DIRS} PATH_SUFFIXES "" debug)
if (SYMBIAN) 
set(GLEW_LIBRARY_REL "${GLEW_LIBRARY_REL}")
set(CMAKE_PREFIX_PATH ${ORIGINAL_CMAKE_PREFIX_PATH})
endif (SYMBIAN)

make_library_set(GLEW_LIBRARY)
findpkg_finish(GLEW)

