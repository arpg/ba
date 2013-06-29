cmake_minimum_required( VERSION 2.8 )

################################################################################
# Find required dependencies

find_package( Eigen3 REQUIRED )
find_package( Calibu 0.1 REQUIRED )
find_package( Sophus REQUIRED )

# Basic includes / libraries
list( APPEND LIB_INC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include" )
list( APPEND USER_INC    ${EIGEN3_INCLUDE_DIR}
                         ${Calibu_INCLUDE_DIRS}
                         ${Sophus_INCLUDE_DIR} )

# this is done in baConfig.cmake.in
#list( APPEND LINK_LIBS  ${Calibu_LIBRARIES} )

################################################################################
# Specify source files

set( INCDIR "../include/ba" )

list(APPEND SOURCES    
)

list(APPEND HEADERS
    ${INCDIR}/BlockAmbiVector.h
    ${INCDIR}/BlockAmbiVector.h
    ${INCDIR}/BlockCompressedStorage.h
    ${INCDIR}/BundleAdjuster.h
    ${INCDIR}/EigenCeresJetNumTraits.h
    ${INCDIR}/InterpolationBuffer.h
    ${INCDIR}/LocalParamSe3.h
    ${INCDIR}/SparseBlockMatrix.h
    ${INCDIR}/SparseBlockMatrixOps.h
)



################################################################################
# Include collected includes / libraries

include_directories( ${LIB_INC_DIR} )
include_directories( ${USER_INC} )
#add_library( ${LIBRARY_NAME} ${SOURCES} ${HEADERS} )
#target_link_libraries( ${LIBRARY_NAME} ${LINK_LIBS} )

#######################################################
## Create configure file for inclusion in library

#CONFIGURE_FILE(
#  "${CMAKE_CURRENT_SOURCE_DIR}/config.h.in"
#  "${CMAKE_CURRENT_BINARY_DIR}/config.h"
#)

#######################################################

# This relative path allows installed files to be relocatable.
set( CMAKECONFIG_INSTALL_DIR "lib/cmake/${PROJECT_NAME}" )
file( RELATIVE_PATH REL_INCLUDE_DIR
    "${CMAKE_INSTALL_PREFIX}/${CMAKECONFIG_INSTALL_DIR}"
    "${CMAKE_INSTALL_PREFIX}/include" )

# Export library for easy inclusion from other cmake projects.
#file( REMOVE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake" )
#export( TARGETS ${LIBRARY_NAME}
#        APPEND FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake" )

# Version information
configure_file("${PROJECT_NAME}ConfigVersion.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake" @ONLY)

# Build tree config
set( EXPORT_LIB_INC_DIR "${LIB_INC_DIR}" )
CONFIGURE_FILE( "${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" @ONLY IMMEDIATE )

# Install tree config
set( EXPORT_LIB_INC_DIR "\${${PROJECT_NAME}_CMAKE_DIR}/${REL_INCLUDE_DIR}" )
configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}Config.cmake" @ONLY )

# Add package to CMake package registery for use from the build tree
export( PACKAGE ${PROJECT_NAME} )

#######################################################
## Install headers / targets

#install(FILES "${CMAKE_CURRENT_BINARY_DIR}/config.h"
#  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
#)
install(FILES ${HEADERS}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
)
#install(TARGETS ${LIBRARY_NAME}
#  EXPORT "${PROJECT_NAME}Targets"
#  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
#  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
#  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
#)

#######################################################
## Install CMake config

INSTALL(
    FILES "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}Config.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION ${CMAKECONFIG_INSTALL_DIR} )

#install( EXPORT "${PROJECT_NAME}Targets" DESTINATION ${CMAKECONFIG_INSTALL_DIR} )
