#!/home/raigor/anaconda3/envs/point_net/bin/python
"""
Executable entry point for PointNet ROS Node
"""

import os
import sys

# Add project directory to Python path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)

from src.ros.pointnet_ros_node import main

if __name__ == '__main__':
    main()
