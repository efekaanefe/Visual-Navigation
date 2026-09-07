cmake_minimum_required(VERSION 3.10)
project(GTSAM_Example)

# GTSAM requires at least C++14 (C++17 is recommended on Arch Linux)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Locate the GTSAM installation
find_package(GTSAM REQUIRED)

# Create the executable
add_executable(main main.cpp)

# Allow the compiler to find local headers like "scripts/utils.hpp"
target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link GTSAM to your executable
target_link_libraries(main PRIVATE gtsam)
