cmake_minimum_required(VERSION 3.10)
project(gui_based_shogi_engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-march=native -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

find_package(X11 REQUIRED)
find_package(OpenCV REQUIRED)
list(APPEND CMAKE_PREFIX_PATH ../libtorch-2.1.1)
find_package(Torch REQUIRED)

# main_for_random_agent
add_executable(main_for_random_agent
  cpp/main_for_random_agent.cpp
  cpp/neural_network.cpp)
target_include_directories(main_for_random_agent PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TORCH_INCLUDE})
target_link_libraries(main_for_random_agent ${X11_LIBRARIES} ${OpenCV_LIBS} ${TORCH_LIBRARIES})

# main_for_offline_training
add_executable(main_for_offline_training
  cpp/main_for_offline_training.cpp
  cpp/neural_network.cpp)
target_include_directories(main_for_offline_training PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TORCH_INCLUDE})
target_link_libraries(main_for_offline_training ${X11_LIBRARIES} ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TORCH_LIBRARIES})
