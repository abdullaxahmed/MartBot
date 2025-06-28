cmake_minimum_required(VERSION 3.0.2)
project(martbot_description)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  urdf
)

catkin_package(
)


include_directories(
  ${catkin_INCLUDE_DIRS}
)

