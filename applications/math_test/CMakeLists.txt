find_package( BA 0.1 )
include_directories( ${BA_INCLUDE_DIRS} )
link_libraries( ${BA_LIBRARIES} )

add_executable( math_test main.cpp )


