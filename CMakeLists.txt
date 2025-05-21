cmake_minimum_required(VERSION 3.8)
project(imitation_nav)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS})

ament_auto_find_build_dependencies()

# Set Libtorch directory manually (change as needed)
set(Torch_DIR "/opt/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

ament_auto_find_build_dependencies()

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${ament_index_cpp_INCLUDE_DIRS}
)

add_executable(imitation_nav
  src/imitation_nav_node.cpp
)

# link dependencies
target_link_libraries(imitation_nav
  ${OpenCV_LIBRARIES}
  ${TORCH_LIBRARIES}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_dependencies(
  ament_cmake_auto
)

target_link_libraries(${PROJECT_NAME}
  # 他の依存ライブラリ
)

ament_auto_package(
  INSTALL_TO_SHARE
)