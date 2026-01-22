"""
PointNet ROS Node
Main ROS node for real-time semantic segmentation of point clouds
"""

import os
import sys
import numpy as np

try:
    import rospy
    from sensor_msgs.msg import PointCloud2
    from visualization_msgs.msg import MarkerArray
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("Error: ROS is not available. Please install ROS.")
    sys.exit(1)

# Import local modules
from src.ros.inference_engine import PointNetInferenceEngine, CLASS_NAMES
from src.ros.pointcloud_converter import PointCloudConverter


class PointNetROSNode:
    """
    ROS node for PointNet semantic segmentation

    Subscribes to point cloud topics, performs inference, and publishes
    segmented point clouds with semantic labels and confidence scores
    """

    def __init__(self):
        """Initialize the ROS node"""
        # Initialize ROS node
        rospy.init_node('pointnet_segmentation', anonymous=False)
        rospy.loginfo("Initializing PointNet ROS Node")

        # Load parameters
        self.load_parameters()

        # Initialize inference engine
        try:
            self.inference_engine = PointNetInferenceEngine(
                model_path=self.model_path,
                device=self.device,
                num_classes=13,
                feature_transform=self.feature_transform,
                chunk_size=self.chunk_size,
                overlap_ratio=self.overlap_ratio
            )
            rospy.loginfo("Inference engine initialized successfully")
        except Exception as e:
            rospy.logerr(f"Failed to initialize inference engine: {e}")
            sys.exit(1)

        # Setup main output publisher
        self.pub = rospy.Publisher(
            self.output_topic,
            PointCloud2,
            queue_size=self.queue_size
        )

        # Setup colored point cloud publisher for RViz visualization
        self.pub_colored = rospy.Publisher(
            self.colored_topic,
            PointCloud2,
            queue_size=self.queue_size
        )

        # Setup legend publisher for RViz visualization
        self.pub_legend = rospy.Publisher(
            self.legend_topic,
            MarkerArray,
            queue_size=1,
            latch=True  # Keep last message for new subscribers
        )

        # Publish legend once at startup
        self.publish_legend()

        # Setup debug publishers
        self.setup_debug_publishers()

        # Setup subscriber
        self.sub = rospy.Subscriber(
            self.input_topic,
            PointCloud2,
            self.pointcloud_callback,
            queue_size=self.queue_size
        )

        rospy.loginfo(f"Subscribed to: {self.input_topic}")
        rospy.loginfo(f"Publishing to: {self.output_topic}")
        rospy.loginfo(f"Publishing colored points to: {self.colored_topic}")
        rospy.loginfo(f"Publishing legend to: {self.legend_topic}")
        rospy.loginfo("PointNet ROS Node ready")

    def load_parameters(self):
        """Load parameters from ROS parameter server"""
        # Topic parameters
        self.input_topic = rospy.get_param('~input_topic', '/cloud_registered')
        self.output_topic = rospy.get_param('~output_topic', '/pointnet/segmented_points')
        self.colored_topic = rospy.get_param('~colored_topic', '/pointnet/colored_points')
        self.legend_topic = rospy.get_param('~legend_topic', '/pointnet/legend')
        self.frame_id = rospy.get_param('~frame_id', 'map')
        # Model parameters
        self.model_path = rospy.get_param('~model_path', 'checkpoints/best_pointnet_s3dis.pth')
        self.device = rospy.get_param('~device', 'cuda')
        self.feature_transform = rospy.get_param('~feature_transform', True)

        # Processing parameters
        self.chunk_size = rospy.get_param('~chunk_size', 4096)
        self.overlap_ratio = rospy.get_param('~overlap_ratio', 0.1)
        self.min_points = rospy.get_param('~min_points', 50)
        self.queue_size = rospy.get_param('~queue_size', 1)

        # Validate model path
        if not os.path.isabs(self.model_path):
            # Convert relative path to absolute
            package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.model_path = os.path.join(package_dir, self.model_path)

        rospy.loginfo("Parameters loaded:")
        rospy.loginfo(f"  Input topic: {self.input_topic}")
        rospy.loginfo(f"  Output topic: {self.output_topic}")
        rospy.loginfo(f"  Model path: {self.model_path}")
        rospy.loginfo(f"  Device: {self.device}")
        rospy.loginfo(f"  Chunk size: {self.chunk_size}")
        rospy.loginfo(f"  Overlap ratio: {self.overlap_ratio}")
        rospy.loginfo(f"  Min points: {self.min_points}")

        # Debug parameters
        self.debug_enabled = rospy.get_param('~debug/enabled', True)
        self.debug_topic_prefix = rospy.get_param('~debug/topic_prefix', '/pointnet/debug')
        self.publish_input_cloud = rospy.get_param('~debug/publish_input', True)
        self.publish_normalized_cloud = rospy.get_param('~debug/publish_normalized', True)
        self.publish_current_chunk = rospy.get_param('~debug/publish_chunk', True)

        if self.debug_enabled:
            rospy.loginfo("Debug mode enabled:")
            rospy.loginfo(f"  Topic prefix: {self.debug_topic_prefix}")
            rospy.loginfo(f"  Publish input cloud: {self.publish_input_cloud}")
            rospy.loginfo(f"  Publish normalized cloud: {self.publish_normalized_cloud}")
            rospy.loginfo(f"  Publish current chunk: {self.publish_current_chunk}")

    def publish_legend(self):
        """Publish the class color legend for RViz"""
        legend = PointCloudConverter.create_legend_markers(self.frame_id)
        self.pub_legend.publish(legend)
        rospy.loginfo("Published class color legend")

    def setup_debug_publishers(self):
        """Setup publishers for debugging intermediate point clouds"""
        self.debug_pubs = {}

        if not self.debug_enabled:
            return

        # Publisher for raw input point cloud
        if self.publish_input_cloud:
            self.debug_pubs['input'] = rospy.Publisher(
                f"{self.debug_topic_prefix}/input_cloud",
                PointCloud2,
                queue_size=1
            )
            rospy.loginfo(f"Debug publisher: {self.debug_topic_prefix}/input_cloud")

        # Publisher for normalized point cloud
        if self.publish_normalized_cloud:
            self.debug_pubs['normalized'] = rospy.Publisher(
                f"{self.debug_topic_prefix}/normalized_cloud",
                PointCloud2,
                queue_size=1
            )
            rospy.loginfo(f"Debug publisher: {self.debug_topic_prefix}/normalized_cloud")

        # Publisher for current processing chunk
        if self.publish_current_chunk:
            self.debug_pubs['chunk'] = rospy.Publisher(
                f"{self.debug_topic_prefix}/current_chunk",
                PointCloud2,
                queue_size=1
            )
            rospy.loginfo(f"Debug publisher: {self.debug_topic_prefix}/current_chunk")

            # Publisher for chunk with predictions (before merging)
            self.debug_pubs['chunk_segmented'] = rospy.Publisher(
                f"{self.debug_topic_prefix}/chunk_segmented",
                PointCloud2,
                queue_size=1
            )
            rospy.loginfo(f"Debug publisher: {self.debug_topic_prefix}/chunk_segmented")

    def publish_debug_cloud(self, cloud_type, xyz, header, labels=None, confidences=None):
        """
        Publish a debug point cloud

        Args:
            cloud_type: Type of cloud ('input', 'normalized', 'chunk', 'chunk_segmented')
            xyz: Point coordinates [N, 3]
            header: ROS message header
            labels: Optional semantic labels [N]
            confidences: Optional confidence scores [N]
        """
        if not self.debug_enabled or cloud_type not in self.debug_pubs:
            return

        try:
            if labels is not None and confidences is not None:
                # Create segmented point cloud
                msg = PointCloudConverter.create_segmented_pointcloud2(
                    xyz, labels, confidences, header
                )
            else:
                # Create simple XYZ point cloud
                msg = PointCloudConverter.create_xyz_pointcloud2(xyz, header)

            self.debug_pubs[cloud_type].publish(msg)
        except Exception as e:
            rospy.logwarn(f"Failed to publish debug cloud '{cloud_type}': {e}")

    def pointcloud_callback(self, cloud_msg):
        """
        Callback for incoming point cloud messages

        Args:
            cloud_msg: sensor_msgs/PointCloud2 message
        """
        try:
            # Validate message
            is_valid, error_msg = PointCloudConverter.validate_pointcloud2(cloud_msg)
            if not is_valid:
                rospy.logwarn(f"Invalid PointCloud2 message: {error_msg}")
                return

            # Convert PointCloud2 to numpy array
            start_time = rospy.Time.now()
            xyz = PointCloudConverter.pointcloud2_to_xyz_array(cloud_msg)

            # Check if point cloud has enough points
            n_points = len(xyz)
            if n_points < self.min_points:
                rospy.logwarn(f"Point cloud has only {n_points} points (min: {self.min_points}), skipping")
                return

            rospy.loginfo(f"Processing point cloud with {n_points} points")

            # Publish input cloud for debugging
            if self.debug_enabled and self.publish_input_cloud:
                self.publish_debug_cloud('input', xyz, cloud_msg.header)
            
            normalized_xyz, _, _ = self.inference_engine.normalize_coords(xyz)

            # Publish normalized cloud for debugging
            if self.debug_enabled and self.publish_normalized_cloud:
                self.publish_debug_cloud('normalized', normalized_xyz, cloud_msg.header)

            # Run inference with debug callback
            predictions, confidences = self.inference_engine.predict_full_pointcloud(
                xyz,
                verbose=True,
                chunk_callback=self.on_chunk_processed if self.debug_enabled else None,
                header=cloud_msg.header
            )

            # Create output PointCloud2 message
            output_msg = PointCloudConverter.create_segmented_pointcloud2(
                xyz,
                predictions,
                confidences,
                cloud_msg.header
            )

            # Publish result
            self.pub.publish(output_msg)

            # Publish colored point cloud for RViz visualization
            colored_msg = PointCloudConverter.create_colored_pointcloud2(xyz, predictions, confidences, cloud_msg.header)
            self.pub_colored.publish(colored_msg)

            # Log processing time
            processing_time = (rospy.Time.now() - start_time).to_sec()
            rospy.loginfo(f"Processed in {processing_time:.3f}s ({n_points/processing_time:.0f} pts/s)")

            # Log class distribution
            unique_labels, counts = np.unique(predictions, return_counts=True)
            class_dist = {CLASS_NAMES[label]: count for label, count in zip(unique_labels, counts)}
            rospy.logdebug(f"Class distribution: {class_dist}")

            #PointCloudConverter.calculate_class_confidence(predictions, confidences)



        except Exception as e:
            rospy.logerr(f"Error processing point cloud: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())

    def on_chunk_processed(self, chunk_info):
        """
        Callback for when a chunk is processed during inference

        Args:
            chunk_info: Dictionary containing:
                - 'chunk_idx': Current chunk index
                - 'total_chunks': Total number of chunks
                - 'chunk_points': Original chunk points [N, 3]
                - 'normalized_points': Normalized chunk points [N, 3]
                - 'predictions': Predicted labels [N]
                - 'confidences': Confidence scores [N]
                - 'header': ROS message header (if provided)
        """
        header = chunk_info.get('header')
        if header is None:
            return

        chunk_idx = chunk_info.get('chunk_idx', 0)
        total_chunks = chunk_info.get('total_chunks', 1)

        rospy.logdebug(f"Publishing debug for chunk {chunk_idx + 1}/{total_chunks}")

        # Publish current chunk (original coordinates)
        if self.publish_current_chunk and 'chunk_points' in chunk_info:
            self.publish_debug_cloud('chunk', chunk_info['chunk_points'], header)

        # Publish chunk with segmentation results
        if self.publish_current_chunk and 'chunk_points' in chunk_info and 'predictions' in chunk_info:
            self.publish_debug_cloud(
                'chunk_segmented',
                chunk_info['chunk_points'],
                header,
                labels=chunk_info['predictions'],
                confidences=chunk_info['confidences']
            )

    def run(self):
        """Keep the node running"""
        rospy.loginfo("PointNet ROS Node running...")
        rospy.spin()


def main():
    """Main entry point"""
    try:
        node = PointNetROSNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("PointNet ROS Node shutting down")
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
