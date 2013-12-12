include(def_executable)
if(NOT ANDROID)
    add_subdirectory(math_test)
    # add_subdirectory(unary_binary_imu_test)
endif()

