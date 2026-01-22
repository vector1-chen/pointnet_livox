# PointNet Livox Implementation

PyTorch implementation of PointNet semantic segmentation for Livox LiDAR. This project provides a complete pipeline for training, evaluation, and real-time ROS inference.

### 1 Installation
```bash
git clone https://github.com/vector1-chen/pointnet_livox.git
cd pointnet_livox
pip install -r requirements.txt
```
>Note: This project requires a ROS environment (e.g., Noetic, Melodic). 
>Required ROS packages: rospy, sensor_msgs, visualization_msgs.

### 2 Data Preparation
```bash
python src/data/preprocessing.py
```

### 3 Training
```bash
# Default training
python src/train.py

# Custom parameters
python src/train.py --batch_size 16 --num_points 4096 --epochs 100 --test_area 5
```
##  ROS Usage

### Launch ROS Node
```bash
# Using launch file (Recommended)
roslaunch pointnet_livox pointnet_segmentation.launch

# Or running python script directly
python scripts/run_pointnet_node.py
```

### ROS Topics
- **Subscribed**: `/cloud_registered` (sensor_msgs/PointCloud2)
- **Published**: 
    - `/pointnet/segmented_points` (Segmented result with labels)
    - `/pointnet/colored_cloud` (Visualized RGB pointcloud)
    - `/pointnet/legend` (Class markers)
    - other debug topics

##  Project Structure
```
pointnet_livox/
├── configs/                    # Configuration files
|—— checkpoints/                # Model
├── launch/                     # ROS launch files
├── rviz/                       # RViz configuration
├── scripts/                    # Executable scripts
├── src/                        # Source code
|   ├── ros/                    # ROS node implementation
│   ├── data/                   # Data processing
│   ├── models/                 # Model definitions
│   └── utils/                  # Utilities
├── notebooks/                  # Jupyter Notebooks
└── requirements.txt            # Python dependencies
```

##  Semantic Classes (S3DIS)

| ID | Class | Color |
|----|-------|-------|
| 0 | ceiling | Red |
| 1 | floor | Dark Green |
| 2 | wall | Bisque |
| 3 | beam | Brown |
| 4 | column | Purple |
| 5 | window | Light Blue |
| 6 | door | Orange |
| 7 | chair | Gray |
| 8 | table | Green |
| 9 | bookcase | Blue |
| 10 | sofa | Pink |
| 11 | board | Cyan |
| 12 | clutter | White |
| -1 | unclassified | Black |