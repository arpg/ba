cmake_minimum_required( VERSION 3.0 )
include(def_library)
include(export_package)

################################################################################
# Find required dependencies

find_package( Eigen3 REQUIRED )
find_package( Calibu 0.1 REQUIRED )
find_package( Sophus REQUIRED )
find_package(TBB REQUIRED)

# Basic includes / libraries
list( APPEND LIB_INC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include" )
list( APPEND USER_INC    ${EIGEN3_INCLUDE_DIR}
                         ${Calibu_INCLUDE_DIRS}
                         ${Sophus_INCLUDE_DIR}
                         ${TBB_INCLUDE_DIRS})

# this is done in baConfig.cmake.in
list( APPEND LINK_LIBS  ${Calibu_LIBRARIES} ${TBB_LIBRARIES} )

################################################################################
# Specify source files

set( INCDIR "../include/ba" )

list(APPEND SOURCES    
    BundleAdjuster.cpp
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
    ${INCDIR}/Types.h
    ${INCDIR}/Utils.h
    ${INCDIR}/CeresCostFunctions.h
    ${INCDIR}/BundleAdjusterTest.h
)

add_custom_target(${PROJECT_NAME} SOURCES ${HEADERS})

include_directories( ${LIB_INC_DIR} )
include_directories( ${USER_INC} )

def_library(${LIBRARY_NAME}
  SOURCES ${SOURCES} ${HEADERS}
  LINK_LIBS ${LINK_LIBS}
  PACKAGE BA
  )

export_package(BA
  TARGETS ${LIBRARY_NAME}
  VERSION ${BA_VERSION}
  LIBRARY ${LIBRARY_NAME}
  LIBRARIES ${LIBRARY_NAME}
  INCLUDE_DIRS ${LIB_INC_DIR} ${USER_INC}
  )

install(FILES ${HEADERS}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
)
