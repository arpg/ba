find_package( BA 0.1 )
find_package( TBB )
include_directories( ${BA_INCLUDE_DIRS} )

def_executable(math_test
  SOURCES main.cpp
  DEPENDS ba
  LINK_LIBS ${BA_LIBRARIES} ${TBB_LIBRARIES}
  )
