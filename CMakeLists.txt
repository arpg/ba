cmake_minimum_required( VERSION 2.8 )
project("BA")
if(POLICY CMP0042)
  cmake_policy(SET CMP0042 NEW) # MACOSX_RPATH
endif()

set(BA_VERSION_MAJOR 0)
set(BA_VERSION_MINOR 1)
set(BA_VERSION ${BA_VERSION_MAJOR}.${BA_VERSION_MINOR})

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules/")
include( install_package )
include( SetPlatformVars )


if(ANDROID)
  add_definitions(-DREAL_TYPE=float)
else()
  add_definitions(-DREAL_TYPE=double)
endif()
add_definitions(-DSOPHUS_DISABLE_ENSURES)

#if(TEST_FLOAT)
#  add_definitions(-DREAL_TYPE=float)
#endif()

# Applications are not building correctly at the moment.
option( BUILD_BA_APPLICATIONS "Build BA Applications" OFF )

if( BUILD_BA_APPLICATIONS )
  add_definitions(-DBUILD_APPS)
endif()

option(BUILD_SHARED_LIBS "Build Shared Library" ON)
set( CMAKE_CXX_FLAGS "-std=c++0x -Wall ${CMAKE_CXX_FLAGS}" )
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

# Overide with cmake -DCMAKE_BUILD_TYPE=Debug {dir}
if( NOT CMAKE_BUILD_TYPE )
   set( CMAKE_BUILD_TYPE Release )
endif()

string( TOLOWER ${PROJECT_NAME} LIBRARY_NAME )

include_directories( include )

find_package( Eigen3 REQUIRED )
find_package( Calibu 0.1 REQUIRED )
find_package( Sophus REQUIRED )

# Basic includes / libraries
list( APPEND LIB_INC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include" )
list( APPEND USER_INC    ${EIGEN3_INCLUDE_DIR}
                         ${Calibu_INCLUDE_DIRS}
                         ${Sophus_INCLUDE_DIR} )

# this is done in baConfig.cmake.in
list( APPEND LINK_LIBS  ${Calibu_LIBRARIES} )

################################################################################
# Specify source files

set( INCDIR "include/ba" )

list(APPEND BA_SRCS
    src/BundleAdjuster.cpp
)

list(APPEND BA_HDRS
    ${INCDIR}/BlockAmbiVector.h
    ${INCDIR}/BlockAmbiVector.h
    ${INCDIR}/BlockCompressedStorage.h
    ${INCDIR}/BundleAdjuster.h
    ${INCDIR}/EigenCeresJetNumTraits.h
    ${INCDIR}/InterpolationBuffer.h
    ${INCDIR}/LocalParamSe3.h
    ${INCDIR}/SparseBlockMatrix.h
    ${INCDIR}/SparseBlockMatrixOps.h
    ${INCDIR}/Types.h
    ${INCDIR}/Utils.h
    ${INCDIR}/CeresCostFunctions.h
    ${INCDIR}/BundleAdjusterTest.h
)

include_directories( ${LIB_INC_DIR} )
include_directories( ${USER_INC} )
include_directories( ${CMAKE_CURRENT_BINARY_DIR}/include )

# build calibu library
add_library( ba ${BA_SRCS} ${BA_HDRS} )

# install everything
install_package(
    PKG_NAME ${PROJECT_NAME}
    LIB_NAME ${LIBRARY_NAME}
    VERSION ${BA_VERSION}
    DESCRIPTION "Bundle adjustment library."
#    INSTALL_HEADERS ${HEADERS}
    INSTALL_INCLUDE_DIR true
    DESTINATION ${CMAKE_INSTALL_PREFIX}
    INCLUDE_DIRS ${USER_INC}
    LINK_LIBS ${LINK_LIBS}
    )

if( BUILD_APPLICATIONS )
  set( BA_DIR ${CMAKE_CURRENT_BINARY_DIR} ) # so subdirs can call find_package( Calibu )
  add_subdirectory( applications )
endif()

if( BUILD_MATLAB )
      add_subdirectory( matlab )
endif()

message( STATUS "BA LIBRARIES: ${BA_LIBRARIES}")

###########################
 include(${CMAKE_MODULE_PATH}/cmake_uninstall.cmake.in)

 add_custom_target(uninstall
     COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
 ###########################
