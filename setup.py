## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD
from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
d = generate_distutils_setup(
    packages=['pointnet_s3dis'],     # 这里填包含 .py 文件的文件夹名
    package_dir={'': ''}  # 表示在当前路径下寻找
)

setup(**d)