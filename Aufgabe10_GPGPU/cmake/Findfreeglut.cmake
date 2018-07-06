#-------------------------------------------------------------------
# This file is part of the CMake build system for OGRE
#     (Object-oriented Graphics Rendering Engine)
# For the latest info, see http://www.ogre3d.org/
#
# The contents of this file are placed in the public domain. Feel
# free to make use of it in any way you like.
#-------------------------------------------------------------------

# - Try to find freeglut
# Once done, this will define
#
#  freeglut_FOUND - system has freeglut
#  freeglut_INCLUDE_DIRS - the freeglut include directories 
#  freeglut_LIBRARIES - link these to use freeglut

include(FindPkgMacros)
findpkg_begin(freeglut)

# Get path, convert backslashes as ${ENV_${var}}
getenv_path(freeglut_HOME)

# construct search paths
set(freeglut_PREFIX_PATH ${freeglut_HOME} ${ENV_freeglut_HOME} )
create_search_paths(freeglut)
# redo search if prefix path changed
clear_if_changed(freeglut_PREFIX_PATH
  freeglut_LIBRARY_FWK
  freeglut_LIBRARY_REL
  freeglut_LIBRARY_DBG
  freeglut_INCLUDE_DIR
)

set(freeglut_LIBRARY_NAMES freeglut)
get_debug_names(freeglut_LIBRARY_NAMES)

use_pkgconfig(freeglut_PKGC freeglut)

findpkg_framework(freeglut)

find_path(freeglut_INCLUDE_DIR NAMES GL/freeglut.h HINTS ${freeglut_INC_SEARCH_PATH} ${freeglut_PKGC_INCLUDE_DIRS})

if (SYMBIAN) 
set(ORIGINAL_CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH})
set(CMAKE_PREFIX_PATH ${CMAKE_SYSYEM_OUT_DIR})
message(STATUS "Lib will be searched in Symbian out dir: ${CMAKE_SYSYEM_OUT_DIR}")
endif (SYMBIAN)
find_library(freeglut_LIBRARY_REL NAMES ${freeglut_LIBRARY_NAMES} HINTS ${freeglut_LIB_SEARCH_PATH} ${freeglut_PKGC_LIBRARY_DIRS} PATH_SUFFIXES "" release relwithdebinfo minsizerel)
find_library(freeglut_LIBRARY_DBG NAMES ${freeglut_LIBRARY_NAMES_DBG} HINTS ${freeglut_LIB_SEARCH_PATH} ${freeglut_PKGC_LIBRARY_DIRS} PATH_SUFFIXES "" debug)
if (SYMBIAN) 
set(freeglut_LIBRARY_REL "${freeglut_LIBRARY_REL}")
set(CMAKE_PREFIX_PATH ${ORIGINAL_CMAKE_PREFIX_PATH})
endif (SYMBIAN)

make_library_set(freeglut_LIBRARY)
findpkg_finish(freeglut)

