cmake_minimum_required(VERSION 3.0.2)
project(pointnet_s3dis)

## Find catkin and required components
find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  visualization_msgs
  geometry_msgs
)

## Declare a catkin package
catkin_package(
  CATKIN_DEPENDS 
    rospy 
    sensor_msgs 
    std_msgs
    visualization_msgs
    geometry_msgs
)

## Mark executable scripts (Python etc.) for installation
catkin_install_python(PROGRAMS
  scripts/run_pointnet_node.py
  src/ros/pointnet_ros_node.py
  src/ros/inference_engine.py
  src/ros/pointcloud_converter.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark other files for installation
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)

install(DIRECTORY rviz/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/rviz
  FILES_MATCHING PATTERN "*.rviz"
)

install(DIRECTORY configs/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/configs
  FILES_MATCHING PATTERN "*.yaml"
)

## Install Python modules
catkin_python_setup()
