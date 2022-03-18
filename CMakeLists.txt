cmake_minimum_required(VERSION 3.2)

project(fastforest)
project(fastforest VERSION 0.2 LANGUAGES CXX)

if(EXPERIMENTAL_TMVA_SUPPORT)
    add_definitions(-DEXPERIMENTAL_TMVA_SUPPORT)
endif(EXPERIMENTAL_TMVA_SUPPORT)

include_directories(include)

add_subdirectory (src)
add_subdirectory (test)

unset(EXPERIMENTAL_TMVA_SUPPORT CACHE)

include(GNUInstallDirs)
install(TARGETS fastforest
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
