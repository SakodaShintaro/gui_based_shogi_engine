cmake_minimum_required(VERSION 3.10)
project(mouse_control_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-march=native -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

find_package(X11 REQUIRED)

add_executable(a.out cpp/main.cpp)

target_link_libraries(a.out ${X11_LIBRARIES})
