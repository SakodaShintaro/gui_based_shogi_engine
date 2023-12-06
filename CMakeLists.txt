cmake_minimum_required(VERSION 3.10)
project(gui_based_shogi_engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-march=native -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

find_package(X11 REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(a.out cpp/main.cpp)

target_include_directories(a.out PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(a.out ${X11_LIBRARIES} ${OpenCV_LIBS})
