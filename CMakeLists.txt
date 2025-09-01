cmake_minimum_required(VERSION 3.5)
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 98 CACHE STRING "Default C++ standard" FORCE)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

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
