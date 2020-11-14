INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_ARCTIC arctic)

FIND_PATH(
    ARCTIC_INCLUDE_DIRS
    NAMES arctic/api.h
    HINTS $ENV{ARCTIC_DIR}/include
        ${PC_ARCTIC_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    ARCTIC_LIBRARIES
    NAMES gnuradio-arctic
    HINTS $ENV{ARCTIC_DIR}/lib
        ${PC_ARCTIC_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/arcticTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(ARCTIC DEFAULT_MSG ARCTIC_LIBRARIES ARCTIC_INCLUDE_DIRS)
MARK_AS_ADVANCED(ARCTIC_LIBRARIES ARCTIC_INCLUDE_DIRS)
