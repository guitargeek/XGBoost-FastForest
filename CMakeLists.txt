cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 98)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(fastforest)
project(fastforest VERSION 0.2 LANGUAGES CXX)

option(EXPERIMENTAL_TMVA_SUPPORT "Build the experimental TMVA support" OFF)

if(EXPERIMENTAL_TMVA_SUPPORT)
    add_definitions(-DEXPERIMENTAL_TMVA_SUPPORT)
endif(EXPERIMENTAL_TMVA_SUPPORT)

include_directories(include)
include(GNUInstallDirs)

add_subdirectory (src)
add_subdirectory (test)
