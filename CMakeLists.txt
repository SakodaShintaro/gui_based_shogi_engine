cmake_minimum_required(VERSION 3.10)
project(gui_based_shogi_engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-march=native -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

find_package(X11 REQUIRED)
find_package(OpenCV REQUIRED)
list(APPEND CMAKE_PREFIX_PATH ../libtorch-2.1.1)
find_package(Torch REQUIRED)

file(GLOB cpp_files cpp/*.cpp)

macro(define_app name)
  get_filename_component(binary_name ${name} NAME_WE)
  add_executable(${binary_name} ${name} ${cpp_files})
  target_include_directories(${binary_name} PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TORCH_INCLUDE})
  target_link_libraries(${binary_name} ${X11_LIBRARIES} ${OpenCV_LIBS} ${TORCH_LIBRARIES})
endmacro()

# main
define_app(cpp/main_functions/play_policy.cpp)
define_app(cpp/main_functions/train_policy.cpp)
define_app(cpp/main_functions/play_decision_transformer.cpp)
define_app(cpp/main_functions/train_decision_transformer.cpp)
