cmake_minimum_required(VERSION 3.14)
project(pacman_neat CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ale REQUIRED)

find_package(OpenMP)

add_executable(pacman_neat src/main.cpp)
target_link_libraries(pacman_neat PRIVATE ale::ale-lib)
target_include_directories(pacman_neat PRIVATE src ${CMAKE_PREFIX_PATH}/include)

if(OpenMP_CXX_FOUND)
    target_link_libraries(pacman_neat PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(pacman_neat PRIVATE USE_OPENMP)
endif()

target_compile_options(pacman_neat PRIVATE -O3 -march=native)
