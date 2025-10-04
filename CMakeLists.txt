cmake_minimum_required(VERSION 3.8)
project(imitation_nav)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

ament_auto_add_library(imitation_nav
  src/topo_localizer.cpp
  src/pointcloud_processor.cpp
)

target_link_libraries(imitation_nav
  ${TORCH_LIBRARIES}
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  yaml-cpp
)

target_include_directories(imitation_nav PRIVATE
  ${TORCH_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

ament_auto_add_executable(imitation_nav_node
  src/imitation_nav_node.cpp
)

target_link_libraries(imitation_nav_node
  imitation_nav
  ${TORCH_LIBRARIES}
  ${OpenCV_LIBS}
  yaml-cpp
)

target_include_directories(imitation_nav_node PRIVATE
  ${TORCH_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

install(DIRECTORY
  include/
  DESTINATION include/
)

install(DIRECTORY
  launch
  config
  weights
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS 
  imitation_nav
  imitation_nav_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package()
