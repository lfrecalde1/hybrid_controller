cmake_minimum_required(VERSION 3.8)
project(hybrid_controller)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

set(dependencies
  rclcpp
  geometry_msgs
  nav_msgs
  std_msgs
)

add_executable(cable_dynamics_node
  src/cable_dynamics_node.cpp
)
target_include_directories(cable_dynamics_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(cable_dynamics_node Eigen3::Eigen)
ament_target_dependencies(cable_dynamics_node ${dependencies})

add_executable(inverse_dynamics_controller_node
  src/inverse_dynamics_controller_node.cpp
)
target_include_directories(inverse_dynamics_controller_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(inverse_dynamics_controller_node Eigen3::Eigen)
ament_target_dependencies(inverse_dynamics_controller_node ${dependencies})

install(TARGETS
  cable_dynamics_node
  inverse_dynamics_controller_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()
