cmake_minimum_required( VERSION 2.8 )
project("BA")
set(BA_VERSION_MAJOR 0)
set(BA_VERSION_MINOR 1)
set(BA_VERSION ${BA_VERSION_MAJOR}.${BA_VERSION_MINOR})

option(EXPORT_BA "Should BA be exported." ON)

if( ANDROID )
    add_definitions(-DREAL_TYPE=float)
else()
    add_definitions(-DREAL_TYPE=double)
endif()
add_definitions(-DSOPHUS_DISABLE_ENSURES)

if( TEST_FLOAT )
    add_definitions(-DREAL_TYPE=float)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake_modules/")

option( BUILD_APPLICATIONS "Build Applications" ON )

if(NOT ANDROID)
    option(BUILD_SHARED_LIBS "Build Shared Library" ON)
else()
    set(BUILD_SHARED_LIBS OFF)
    set( CMAKE_CXX_FLAGS "-llog ${CMAKE_CXX_FLAGS}" )
endif()

set( CMAKE_CXX_FLAGS "-std=c++0x -Wall ${CMAKE_CXX_FLAGS}" )
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

# Overide with cmake -DCMAKE_BUILD_TYPE=Debug {dir}
if( NOT CMAKE_BUILD_TYPE )
   set( CMAKE_BUILD_TYPE Release )
endif()

string( TOLOWER ${PROJECT_NAME} LIBRARY_NAME )

include_directories(include)
add_subdirectory( src )

if( BUILD_APPLICATIONS )
    add_subdirectory( applications )
endif()

