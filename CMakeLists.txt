cmake_minimum_required(VERSION 3.2)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

project(fastforest)
project(fastforest VERSION 0.0.1 DESCRIPTION "Minimal library code to deploy XGBoost models in C++")

include_directories(include)

add_subdirectory (src)
add_subdirectory (test)

enable_testing ()
add_test (NAME fastforestTest COMMAND Test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test)

file(GLOB_RECURSE SOURCE_FILES "src/*.cpp")

add_library (fastforest SHARED ${SOURCE_FILES})
target_link_libraries(fastforest stdc++fs)

set_target_properties(fastforest PROPERTIES VERSION ${PROJECT_VERSION})

set_target_properties(fastforest PROPERTIES SOVERSION 1)

set_target_properties(fastforest PROPERTIES PUBLIC_HEADER include/fastforest.h)

include(GNUInstallDirs)
install(TARGETS fastforest
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
