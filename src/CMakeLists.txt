set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -pedantic-errors")

option(EXPERIMENTAL_TMVA_SUPPORT "Build the experimental TMVA support" OFF)
if(EXPERIMENTAL_TMVA_SUPPORT)
    set(CMAKE_CXX_STANDARD 17)
    file(GLOB_RECURSE SOURCE_FILES "*.cpp")
else()
    set(CMAKE_CXX_STANDARD 98)
    file(GLOB_RECURSE SOURCE_FILES common_details.cpp fastforest_functions.cpp fastforest.cpp)
endif(EXPERIMENTAL_TMVA_SUPPORT)

add_library (fastforest SHARED ${SOURCE_FILES})

set_target_properties(fastforest PROPERTIES VERSION ${PROJECT_VERSION})

set_target_properties(fastforest PROPERTIES SOVERSION 1)

set_target_properties(fastforest PROPERTIES PUBLIC_HEADER include/fastforest.h)
