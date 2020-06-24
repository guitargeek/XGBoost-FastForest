cmake_minimum_required(VERSION 3.2)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

project(fastforest)
project(fastforest VERSION 0.2 LANGUAGES CXX)

include_directories(include)

add_subdirectory (src)
add_subdirectory (test)

enable_testing ()
add_test (NAME fastforestTest COMMAND Test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test)


option(EXPERIMENTAL_TMVA_SUPPORT "Build the experimental TMVA support" OFF)
if(EXPERIMENTAL_TMVA_SUPPORT)
    set(CMAKE_CXX_STANDARD 17)
    file(GLOB_RECURSE SOURCE_FILES "src/*.cpp")
    add_definitions(-DEXPERIMENTAL_TMVA_SUPPORT)
else()
    set(CMAKE_CXX_STANDARD 11)
    file(GLOB_RECURSE SOURCE_FILES src/common_details.cpp src/fastforest.cpp src/FastForest.cpp)
endif(EXPERIMENTAL_TMVA_SUPPORT)
unset(EXPERIMENTAL_TMVA_SUPPORT CACHE)


add_library (fastforest SHARED ${SOURCE_FILES})
target_link_libraries(fastforest stdc++fs)

set_target_properties(fastforest PROPERTIES VERSION ${PROJECT_VERSION})

set_target_properties(fastforest PROPERTIES SOVERSION 1)

set_target_properties(fastforest PROPERTIES PUBLIC_HEADER include/fastforest.h)

include(GNUInstallDirs)
install(TARGETS fastforest
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
