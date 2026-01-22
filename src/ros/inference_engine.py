"""
PointNet Inference Engine for ROS
Handles model loading and inference for semantic segmentation
"""

import os
import sys
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.pointnet import PointNetSegmentation

# Class names for S3DIS dataset
CLASS_NAMES = [
    'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
    'chair', 'table', 'bookcase', 'sofa', 'board', 'clutter'
]

class PointNetInferenceEngine:
    """
    Inference engine for PointNet semantic segmentation

    Handles model loading, normalization, and chunked inference for large point clouds
    """

    def __init__(self, model_path, device='cuda', num_classes=13,
                 feature_transform=True, chunk_size=4096, overlap_ratio=0.1):
        """
        Initialize the inference engine

        Args:
            model_path: Path to model checkpoint (.pth file)
            device: Device to run inference on ('cuda' or 'cpu')
            num_classes: Number of semantic classes (default: 13 for S3DIS)
            feature_transform: Whether model uses feature transformation
            chunk_size: Number of points to process in each chunk
            overlap_ratio: Overlap between chunks (0.0 to 1.0)
        """
        self.model_path = model_path
        self.num_classes = num_classes
        self.feature_transform = feature_transform
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio

        # Set device with automatic fallback
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = 'cpu'
        self.device = torch.device(device)

        # Initialize model
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the trained PointNet model from checkpoint"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        print(f"Loading model from {self.model_path}")

        # Initialize model architecture
        self.model = PointNetSegmentation(
            num_classes=self.num_classes,
            feature_transform=self.feature_transform
        )

        # Load weights
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def normalize_coords(self, coords):
        """
        Normalize coordinates to unit sphere (must match training normalization)

        Args:
            coords: Point cloud coordinates [N, 3]

        Returns:
            normalized_coords: Normalized coordinates [N, 3]
            centroid: Original centroid
            scale: Normalization scale factor
        """
        # Center to origin
        centroid = np.mean(coords, axis=0)
        coords_centered = coords - centroid

        # Scale to unit sphere
        m = np.max(np.sum(coords_centered**2, axis=1))
        m = np.sqrt(m)
        if m > 0:
            coords_normalized = coords_centered / m
        else:
            coords_normalized = coords_centered

        return coords_normalized, centroid, m

    def predict_chunk(self, chunk_points):
        """
        Predict semantic labels for a single chunk

        Args:
            chunk_points: Normalized point cloud chunk [N, 3]

        Returns:
            pred_labels: Predicted class labels [N]
            pred_confidence: Prediction confidence scores [N]
        """
        with torch.no_grad():
            # Convert to tensor [1, 3, N]
            chunk_tensor = torch.FloatTensor(chunk_points).unsqueeze(0).to(self.device)
            chunk_tensor = chunk_tensor.transpose(2, 1)

            # Forward pass
            pred_logits, _, _ = self.model(chunk_tensor)

            # Get predictions and confidence
            pred_probs = torch.softmax(pred_logits, dim=-1)
            pred_labels = pred_logits.data.max(2)[1].cpu().numpy()[0]
            pred_confidence = pred_probs.data.max(2)[0].cpu().numpy()[0]

        return pred_labels, pred_confidence

    def predict_full_pointcloud(self, points, verbose=True, chunk_callback=None, header=None):
        """
        Predict semantic labels for a full point cloud using overlapping chunks

        Args:
            points: Full point cloud data [N, 3] (xyz coordinates)
            verbose: Whether to show progress bar
            chunk_callback: Optional callback function called after each chunk is processed.
                           Receives dict with chunk info for debugging/visualization.
            header: Optional ROS message header (passed to chunk_callback)

        Returns:
            predictions: Predicted labels for all points [N]
            confidence_scores: Prediction confidence scores [N]
        """
        n_points = len(points)

        # Initialize output arrays
        predictions = np.zeros(n_points, dtype=np.int32)
        confidence_scores = np.zeros(n_points, dtype=np.float32)
        vote_counts = np.zeros(n_points, dtype=np.int32)

        # Calculate step size for overlapping windows
        step_size = int(self.chunk_size * (1 - self.overlap_ratio))

        if verbose:
            print(f"Processing {n_points:,} points in chunks of {self.chunk_size} "
                  f"with {self.overlap_ratio*100:.0f}% overlap")

        # Process chunks
        chunk_indices = list(range(0, n_points, step_size))
        total_chunks = len(chunk_indices)
        iterator = tqdm(chunk_indices, desc="Processing chunks") if verbose else chunk_indices

        for chunk_num, start_idx in enumerate(iterator):
            end_idx = min(start_idx + self.chunk_size, n_points)

            # Extract chunk
            chunk_points = points[start_idx:end_idx].copy()
            actual_chunk_size = len(chunk_points)

            # Skip if chunk is too small
            if actual_chunk_size < self.chunk_size // 4:
                continue

            # Pad chunk if necessary
            if actual_chunk_size < self.chunk_size:
                indices = np.concatenate([
                    np.arange(actual_chunk_size),
                    np.random.choice(actual_chunk_size, self.chunk_size - actual_chunk_size, replace=True)
                ])
                chunk_points_padded = chunk_points[indices]
                chunk_indices_map = np.arange(start_idx, end_idx)[indices]
            else:
                chunk_points_padded = chunk_points
                chunk_indices_map = np.arange(start_idx, end_idx)

            # Normalize chunk
            normalized_chunk, _, _ = self.normalize_coords(chunk_points_padded)

            # Predict
            try:
                pred_labels, pred_confidence = self.predict_chunk(normalized_chunk)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"GPU OOM error, skipping chunk at {start_idx}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            # Handle padded points
            if actual_chunk_size < self.chunk_size:
                pred_labels = pred_labels[:actual_chunk_size]
                pred_confidence = pred_confidence[:actual_chunk_size]
                chunk_indices_map = chunk_indices_map[:actual_chunk_size]

            # Call chunk callback for debugging/visualization
            if chunk_callback is not None:
                chunk_callback({
                    'chunk_idx': chunk_num,
                    'total_chunks': total_chunks,
                    'chunk_points': chunk_points,
                    'normalized_points': normalized_chunk[:actual_chunk_size],
                    'predictions': pred_labels,
                    'confidences': pred_confidence,
                    'header': header
                })

            # Accumulate votes for overlapping regions
            for i, idx in enumerate(chunk_indices_map):
                if idx < n_points:
                    if vote_counts[idx] == 0:
                        # First prediction for this point
                        predictions[idx] = pred_labels[i]
                        confidence_scores[idx] = pred_confidence[i]
                    else:
                        # Multiple predictions - use highest confidence
                        if pred_confidence[i] > confidence_scores[idx]:
                            predictions[idx] = pred_labels[i]

                        # Average confidence scores
                        old_weight = vote_counts[idx]
                        new_weight = 1
                        total_weight = old_weight + new_weight
                        confidence_scores[idx] = (
                            confidence_scores[idx] * old_weight +
                            pred_confidence[i] * new_weight
                        ) / total_weight

                    vote_counts[idx] += 1

        if verbose:
            avg_votes = np.mean(vote_counts[vote_counts > 0])
            print(f"Prediction completed! Average votes per point: {avg_votes:.2f}")

        return predictions, confidence_scores

    def predict_full_pointcloud_spatial(self, points, verbose=True, chunk_callback=None, 
                                         header=None, voxel_size=None):
        """
        Predict semantic labels using 3D spatial voxel-based chunking.
        
        This method divides the point cloud into 3D cubic voxels, which preserves
        spatial locality better than linear chunking.

        Args:
            points: Full point cloud data [N, 3] (xyz coordinates)
            verbose: Whether to show progress bar
            chunk_callback: Optional callback function called after each chunk is processed.
            header: Optional ROS message header (passed to chunk_callback)
            voxel_size: Size of each voxel cube in meters. If None, auto-computed based on chunk_size.

        Returns:
            predictions: Predicted labels for all points [N]
            confidence_scores: Prediction confidence scores [N]
        """
        n_points = len(points)
        
        # Initialize output arrays
        predictions = np.zeros(n_points, dtype=np.int32)
        confidence_scores = np.zeros(n_points, dtype=np.float32)
        vote_counts = np.zeros(n_points, dtype=np.int32)

        # Calculate bounding box
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        extent = max_coords - min_coords

        # Auto-compute voxel size if not provided
        if voxel_size is None:
            # Estimate voxel size based on point density and chunk_size
            volume = np.prod(extent) if np.prod(extent) > 0 else 1.0
            point_density = n_points / volume
            # Target: each voxel should have roughly chunk_size points
            target_voxel_volume = self.chunk_size / point_density
            voxel_size = target_voxel_volume ** (1/3)
            # Clamp to reasonable range
            voxel_size = max(0.5, min(voxel_size, 10.0))

        # Calculate overlap size (similar to overlap_ratio but in 3D)
        overlap_size = voxel_size * self.overlap_ratio

        # Calculate effective step size (with overlap)
        step_size = voxel_size - overlap_size

        # Calculate grid dimensions
        grid_dims = np.ceil(extent / step_size).astype(int)
        grid_dims = np.maximum(grid_dims, 1)  # At least 1 voxel per dimension

        if verbose:
            print(f"Point cloud extent: {extent}")
            print(f"Voxel size: {voxel_size:.3f}m, Overlap: {overlap_size:.3f}m")
            print(f"Grid dimensions: {grid_dims} = {np.prod(grid_dims)} voxels")
            print(f"Processing {n_points:,} points with 3D spatial chunking")

        # Generate all voxel indices
        voxel_list = []
        for ix in range(grid_dims[0]):
            for iy in range(grid_dims[1]):
                for iz in range(grid_dims[2]):
                    voxel_list.append((ix, iy, iz))

        total_voxels = len(voxel_list)
        processed_voxels = 0
        skipped_voxels = 0

        iterator = tqdm(voxel_list, desc="Processing voxels") if verbose else voxel_list

        for voxel_idx, (ix, iy, iz) in enumerate(iterator):
            # Calculate voxel bounds (with overlap)
            voxel_min = min_coords + np.array([ix, iy, iz]) * step_size
            voxel_max = voxel_min + voxel_size

            # Find points within this voxel
            mask = np.all((points >= voxel_min) & (points < voxel_max), axis=1)
            point_indices = np.where(mask)[0]

            if len(point_indices) == 0:
                skipped_voxels += 1
                continue

            # Get points in this voxel
            voxel_points = points[point_indices]
            actual_chunk_size = len(voxel_points)

            # Handle large voxels by random sampling
            if actual_chunk_size > self.chunk_size:
                sample_indices = np.random.choice(
                    actual_chunk_size, self.chunk_size, replace=False
                )
                voxel_points_sampled = voxel_points[sample_indices]
                point_indices_sampled = point_indices[sample_indices]
            else:
                voxel_points_sampled = voxel_points
                point_indices_sampled = point_indices

            current_size = len(voxel_points_sampled)

            # Pad if necessary
            if current_size < self.chunk_size:
                if current_size < self.chunk_size // 4:
                    # Too few points, skip this voxel
                    skipped_voxels += 1
                    continue
                    
                # Pad with repeated points
                pad_indices = np.random.choice(current_size, self.chunk_size - current_size, replace=True)
                voxel_points_padded = np.vstack([voxel_points_sampled, voxel_points_sampled[pad_indices]])
            else:
                voxel_points_padded = voxel_points_sampled

            # Normalize the voxel points
            normalized_voxel, _, _ = self.normalize_coords(voxel_points_padded)

            # Predict
            try:
                pred_labels, pred_confidence = self.predict_chunk(normalized_voxel)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"GPU OOM error, skipping voxel ({ix}, {iy}, {iz})")
                    torch.cuda.empty_cache()
                    skipped_voxels += 1
                    continue
                else:
                    raise e

            # Only use predictions for original (non-padded) points
            pred_labels = pred_labels[:current_size]
            pred_confidence = pred_confidence[:current_size]

            processed_voxels += 1

            # Call chunk callback for debugging/visualization
            if chunk_callback is not None:
                chunk_callback({
                    'chunk_idx': voxel_idx,
                    'total_chunks': total_voxels,
                    'voxel_coords': (ix, iy, iz),
                    'voxel_bounds': (voxel_min, voxel_max),
                    'chunk_points': voxel_points_sampled,
                    'normalized_points': normalized_voxel[:current_size],
                    'predictions': pred_labels,
                    'confidences': pred_confidence,
                    'header': header
                })

            # Accumulate votes for overlapping regions
            for i, idx in enumerate(point_indices_sampled):
                if vote_counts[idx] == 0:
                    # First prediction for this point
                    predictions[idx] = pred_labels[i]
                    confidence_scores[idx] = pred_confidence[i]
                else:
                    # Multiple predictions - use highest confidence
                    if pred_confidence[i] > confidence_scores[idx]:
                        predictions[idx] = pred_labels[i]

                    # Weighted average of confidence scores
                    old_weight = vote_counts[idx]
                    new_weight = 1
                    total_weight = old_weight + new_weight
                    confidence_scores[idx] = (
                        confidence_scores[idx] * old_weight +
                        pred_confidence[i] * new_weight
                    ) / total_weight

                vote_counts[idx] += 1

        if verbose:
            covered_points = np.sum(vote_counts > 0)
            avg_votes = np.mean(vote_counts[vote_counts > 0]) if covered_points > 0 else 0
            print(f"\nSpatial prediction completed!")
            print(f"  Processed voxels: {processed_voxels}/{total_voxels}")
            print(f"  Skipped voxels (empty or too small): {skipped_voxels}")
            print(f"  Points covered: {covered_points:,}/{n_points:,} ({100*covered_points/n_points:.1f}%)")
            print(f"  Average votes per point: {avg_votes:.2f}")

        return predictions, confidence_scores


