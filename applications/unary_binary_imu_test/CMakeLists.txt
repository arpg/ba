find_package( BA 0.1 )
include_directories( ${BA_INCLUDE_DIRS} )
link_libraries( ${BA_LIBRARIES} )

def_executable(unary_binary_imu_test
  SOURCES main.cpp
  DEPENDS
  ba
  LINK_LIBS
  ${BA_LIBRARIES}
  )
