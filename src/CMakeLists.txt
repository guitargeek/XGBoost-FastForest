set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -pedantic-errors")

if(EXPERIMENTAL_TMVA_SUPPORT)
    file(GLOB_RECURSE SOURCE_FILES "*.cpp")
else()
    file(GLOB_RECURSE SOURCE_FILES common_details.cpp fastforest_functions.cpp fastforest.cpp)
endif(EXPERIMENTAL_TMVA_SUPPORT)

add_library (fastforest SHARED ${SOURCE_FILES})

set_target_properties(fastforest PROPERTIES VERSION ${PROJECT_VERSION})

set_target_properties(fastforest PROPERTIES SOVERSION 1)

set_target_properties(fastforest PROPERTIES PUBLIC_HEADER ../include/fastforest.h)

install(TARGETS fastforest
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
