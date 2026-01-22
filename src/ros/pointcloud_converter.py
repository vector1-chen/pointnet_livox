"""
PointCloud2 Conversion Utilities
Handles conversion between ROS PointCloud2 messages and numpy arrays
"""

import numpy as np
import struct

try:
    import rospy
    from sensor_msgs.msg import PointCloud2, PointField
    from visualization_msgs.msg import Marker, MarkerArray
    import sensor_msgs.point_cloud2 as pc2
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("Warning: ROS not available. PointCloud converter will not work.")


# S3DIS class names (13 classes)
CLASS_NAMES = [
    'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
    'chair', 'table', 'bookcase', 'sofa', 'board', 'clutter'
]

# RGB colors for each S3DIS class (0-255) - for RViz visualization
CLASS_COLORS_RGB = [
    (255, 0, 0),        # 0: ceiling - red
    (0, 100, 0),        # 1: floor - dark green
    (255, 228, 181),    # 2: wall - bisque/cream
    (139, 69, 19),      # 3: beam - brown
    (160, 32, 240),     # 4: column - purple
    (135, 206, 250),    # 5: window - light blue
    (255, 165, 0),      # 6: door - orange
    (120, 120, 120),    # 7: chair - gray
    (0, 255, 0),        # 8: table - green
    (0, 0, 255),        # 9: bookcase - blue
    (255, 192, 203),    # 10: sofa - pink
    (0, 255, 255),      # 11: board - cyan
    (255, 255, 255),    # 12: clutter - white
]


class PointCloudConverter:
    """Utility class for converting between ROS PointCloud2 and numpy arrays"""

    @staticmethod
    def pointcloud2_to_xyz_array(cloud_msg):
        """
        Convert ROS PointCloud2 message to numpy array of XYZ coordinates

        Args:
            cloud_msg: sensor_msgs/PointCloud2 message

        Returns:
            xyz_array: Numpy array of shape [N, 3] containing XYZ coordinates
        """
        if not ROS_AVAILABLE:
            raise RuntimeError("ROS is not available")

        # Read points from PointCloud2 message
        points_list = []

        # Use sensor_msgs.point_cloud2.read_points to extract data
        for point in pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            points_list.append([point[0], point[1], point[2]])

        # Convert to numpy array
        if len(points_list) == 0:
            return np.array([]).reshape(0, 3)

        xyz_array = np.array(points_list, dtype=np.float32)
        return xyz_array

    @staticmethod
    def create_segmented_pointcloud2(xyz, labels, confidences, header):
        """
        Create a PointCloud2 message with semantic labels and confidence scores

        Args:
            xyz: Numpy array of XYZ coordinates [N, 3]
            labels: Numpy array of semantic class labels [N] (uint8)
            confidences: Numpy array of confidence scores [N] (float32)
            header: Original message header (contains frame_id and timestamp)

        Returns:
            cloud_msg: sensor_msgs/PointCloud2 message with custom fields
        """
        if not ROS_AVAILABLE:
            raise RuntimeError("ROS is not available")

        n_points = len(xyz)

        # Define PointCloud2 fields
        # x, y, z (float32), label (uint8), confidence (float32)
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('label', 12, PointField.UINT8, 1),
            PointField('confidence', 13, PointField.FLOAT32, 1),
        ]

        # Pack data into binary format
        cloud_data = []
        for i in range(n_points):
            # Pack: x, y, z (float32), label (uint8), confidence (float32)
            # Total: 4 + 4 + 4 + 1 + 4 = 17 bytes per point
            point_data = struct.pack('fffBf',
                                    float(xyz[i, 0]),
                                    float(xyz[i, 1]),
                                    float(xyz[i, 2]),
                                    int(labels[i]),
                                    float(confidences[i]))
            cloud_data.append(point_data)

        # Concatenate all point data
        cloud_data_bytes = b''.join(cloud_data)

        # Create PointCloud2 message
        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1
        cloud_msg.width = n_points
        cloud_msg.fields = fields
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = 17  # bytes per point
        cloud_msg.row_step = cloud_msg.point_step * n_points
        cloud_msg.data = cloud_data_bytes
        cloud_msg.is_dense = True

        return cloud_msg

    @staticmethod
    def validate_pointcloud2(cloud_msg):
        """
        Validate PointCloud2 message format

        Args:
            cloud_msg: sensor_msgs/PointCloud2 message

        Returns:
            is_valid: Boolean indicating if message is valid
            error_msg: Error message if invalid, None otherwise
        """
        if not ROS_AVAILABLE:
            return False, "ROS is not available"

        # Check if message has required fields
        field_names = [field.name for field in cloud_msg.fields]

        if 'x' not in field_names or 'y' not in field_names or 'z' not in field_names:
            return False, "PointCloud2 message missing x, y, or z fields"

        # Check if message has points
        if cloud_msg.width == 0 or cloud_msg.height == 0:
            return False, "PointCloud2 message has no points"

        return True, None

    @staticmethod
    def create_xyz_pointcloud2(xyz, header, rgb=None):
        """
        Create a simple PointCloud2 message with XYZ (and optional RGB) for debugging

        Args:
            xyz: Numpy array of XYZ coordinates [N, 3]
            header: ROS message header
            rgb: Optional RGB colors [N, 3] as uint8 (0-255) or float (0-1)

        Returns:
            cloud_msg: sensor_msgs/PointCloud2 message
        """
        if not ROS_AVAILABLE:
            raise RuntimeError("ROS is not available")

        n_points = len(xyz)

        if rgb is not None:
            # XYZRGB format
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('rgb', 12, PointField.FLOAT32, 1),
            ]
            point_step = 16

            # Convert RGB to packed float
            if rgb.dtype != np.uint8:
                rgb = (rgb * 255).astype(np.uint8)

            cloud_data = []
            for i in range(n_points):
                # Pack RGB into a single float (ROS convention)
                r, g, b = int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2])
                rgb_packed = struct.pack('BBBB', b, g, r, 0)
                rgb_float = struct.unpack('f', rgb_packed)[0]

                point_data = struct.pack('ffff',
                                        float(xyz[i, 0]),
                                        float(xyz[i, 1]),
                                        float(xyz[i, 2]),
                                        rgb_float)
                cloud_data.append(point_data)
        else:
            # XYZ only format
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
            ]
            point_step = 12

            cloud_data = []
            for i in range(n_points):
                point_data = struct.pack('fff',
                                        float(xyz[i, 0]),
                                        float(xyz[i, 1]),
                                        float(xyz[i, 2]))
                cloud_data.append(point_data)

        # Concatenate all point data
        cloud_data_bytes = b''.join(cloud_data)

        # Create PointCloud2 message
        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1
        cloud_msg.width = n_points
        cloud_msg.fields = fields
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = point_step
        cloud_msg.row_step = point_step * n_points
        cloud_msg.data = cloud_data_bytes
        cloud_msg.is_dense = True

        return cloud_msg

    @staticmethod
    def create_colored_pointcloud2(xyz, labels, confidences, header):
        """
        Create a PointCloud2 message with RGB colors based on semantic labels

        Args:
            xyz: Numpy array of XYZ coordinates [N, 3]
            labels: Numpy array of semantic class labels [N]
            confidences: Numpy array of confidence scores [N]
            header: ROS message header

        Returns:
            PointCloud2 message with XYZRGB fields for RViz visualization
        """
        if not ROS_AVAILABLE:
            raise RuntimeError("ROS is not available")

        n_points = len(xyz)

        # Define PointCloud2 fields for XYZRGB
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1),
        ]

        # Pack data
        cloud_data = []
        for i in range(n_points):
            label = int(labels[i]) % len(CLASS_COLORS_RGB)
            r, g, b = CLASS_COLORS_RGB[label]
            if confidences[i] < 0.5:
                # black color for confidence < 0.5
                r, g, b = int(0), int(0), int(0)
            
            # Pack RGB into single uint32
            rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, 255))[0]

            point_data = struct.pack('fffI',
                                    float(xyz[i, 0]),
                                    float(xyz[i, 1]),
                                    float(xyz[i, 2]),
                                    rgb)
            cloud_data.append(point_data)

        cloud_data_bytes = b''.join(cloud_data)

        # Create message
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = n_points
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * n_points
        msg.data = cloud_data_bytes
        msg.is_dense = True

        return msg

    @staticmethod
    def create_legend_markers(frame_id):
        """
        Create MarkerArray for legend display in RViz

        Args:
            frame_id: Reference frame for markers

        Returns:
            MarkerArray message with colored text labels for each class
        """
        if not ROS_AVAILABLE:
            raise RuntimeError("ROS is not available")

        marker_array = MarkerArray()

        for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS_RGB)):
            # Text marker
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = "legend"
            marker.id = i
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD

            # Position in corner of view
            marker.pose.position.x = -2.0
            marker.pose.position.y = 8.0 - i * 1.8
            marker.pose.position.z = 5.0
            marker.pose.orientation.w = 1.0

            marker.scale.z = 0.8  # Text height

            r, g, b = color
            marker.color.r = r / 255.0
            marker.color.g = g / 255.0
            marker.color.b = b / 255.0
            marker.color.a = 1.0

            marker.text = f"[{i}] {name}"
            marker.lifetime = rospy.Duration(0)  # Persistent

            marker_array.markers.append(marker)

        return marker_array
    
    @staticmethod
    def calculate_class_confidence(labels, confidences):
        """
        Calculate and log class average confidence scores

        Args:
            labels: Numpy array of semantic class labels [N]
            confidences: Numpy array of confidence scores [N]
        """
        if not ROS_AVAILABLE:
            raise RuntimeError("ROS is not available")

        unique_labels, _ = np.unique(labels, return_counts=True)

        avg_confidences = {}
        for label in unique_labels:
            mask = (labels == label)
            avg_confidence = np.mean(confidences[mask])
            avg_confidences[CLASS_NAMES[label]] = avg_confidence
        rospy.loginfo(f"Average confidences: {avg_confidences}")