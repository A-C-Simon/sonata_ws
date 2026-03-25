"""
SemanticKITTI Dataset Handler

Loads and preprocesses SemanticKITTI data for semantic scene completion.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import yaml

from utils.point_cloud import crop_lidar_radius, crop_lidar_radius_with_labels


class SemanticKITTI(Dataset):
    """
    SemanticKITTI dataset for semantic scene completion.
    
    Loads:
    - Input scans (partial LiDAR)
    - Ground truth complete scenes
    - Semantic labels
    - Poses for map generation
    """
    
    # SemanticKITTI class mapping
    LEARNING_MAP = {
        0: 0,      # "unlabeled"
        1: 0,      # "outlier" mapped to "unlabeled"
        10: 1,     # "car"
        11: 2,     # "bicycle"
        13: 5,     # "bus"
        15: 3,     # "motorcycle"
        16: 5,     # "on-rails"
        18: 4,     # "truck"
        20: 5,     # "other-vehicle"
        30: 6,     # "person"
        31: 7,     # "bicyclist"
        32: 8,     # "motorcyclist"
        40: 9,     # "road"
        44: 10,    # "parking"
        48: 11,    # "sidewalk"
        49: 12,    # "other-ground"
        50: 13,    # "building"
        51: 14,    # "fence"
        52: 0,     # "other-structure" to "unlabeled"
        60: 9,     # "lane-marking" to "road"
        70: 15,    # "vegetation"
        71: 16,    # "trunk"
        72: 17,    # "terrain"
        80: 18,    # "pole"
        81: 19,    # "traffic-sign"
        99: 0,     # "other-object" to "unlabeled"
        252: 1,    # "moving-car" to "car"
        253: 7,    # "moving-bicyclist" to "bicyclist"
        254: 6,    # "moving-person" to "person"
        255: 8,    # "moving-motorcyclist" to "motorcyclist"
        256: 5,    # "moving-on-rails" mapped to "other-vehicle"
        257: 5,    # "moving-bus" mapped to "other-vehicle"
        258: 4,    # "moving-truck" to "truck"
        259: 5,    # "moving-other-vehicle" to "other-vehicle"
    }
    
    NUM_CLASSES = 20

    # Split over sequences 00–10 only (matches GT from map_from_scans in README).
    # Not the official SemanticKITTI benchmark (that uses val=08, test=11–21).
    # train: 01–08, val: 09, test: 10. Add '00' to train if you want that scene in training.
    SPLITS = {
        'train': ['00', '01', '02', '03', '04', '05', '06', '07', '08'],
        'val': ['09'],
        'test': ['10'],
    }

    # Official SemanticKITTI: val=08, test=11–21 (no labels on test; need GT maps if you train/eval with GT).
    SPLITS_OFFICIAL = {
        'train': ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10'],
        'val': ['08'],
        'test': [f'{i:02d}' for i in range(11, 22)],
    }
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        voxel_size: float = 0.1,
        max_points: int = 20000,
        use_ground_truth_maps: bool = True,
        augmentation: bool = True,
        num_points_per_scan: Optional[int] = None,
        use_precomputed: bool = False,
        gt_subdir: str = "ground_truth",
        gt_name_suffix: str = "",
        split_preset: str = 'project',
        sequence_ids: Optional[List[str]] = None,
        intra_seq_val_fraction: float = 0.0,
        intra_seq_test_fraction: float = 0.0,
        scene_radius: float = 0.0,
        sequence_scan_start: Optional[int] = None,
        sequence_scan_end: Optional[int] = None,
    ):
        """
        Initialize SemanticKITTI dataset.
        
        Args:
            root: Path to SemanticKITTI dataset root
            split: Dataset split ('train', 'val', 'test')
            voxel_size: Voxel size for scene representation
            max_points: Maximum points per sample
            use_ground_truth_maps: Use pre-generated complete maps as GT
            augmentation: Apply data augmentation
            num_points_per_scan: Subsample scans to this number
            gt_subdir: Subfolder under root for GT NPZ (e.g. ground_truth_v2)
            gt_name_suffix: Filename suffix before .npz (e.g. _v2 -> 000000_v2.npz)
            split_preset: 'project' (SPLITS) or 'official' (SPLITS_OFFICIAL)
            sequence_ids: If set, use only these sequence ids (e.g. ['03'] or ['11','12']);
                overrides split_preset list for this split.
            intra_seq_val_fraction: If intra_seq_test_fraction==0 and this >0: two-way split —
                first (1-f) scans for train, last f for val. If intra_seq_test_fraction>0:
                three-way split — middle segment has fraction f (val), last segment has
                intra_seq_test_fraction (test), first segment is the rest (train).
            intra_seq_test_fraction: If >0, three-way temporal split (requires val fraction >0
                and exactly one sequence). split='train'|'val'|'test' selects the segment.
            scene_radius: If >0, crop raw LiDAR (with labels) and GT map points to this
                Euclidean radius (m) from sensor origin before voxelize (match inference).
            sequence_scan_start: If set (with a single sequence), keep scans from this
                0-based index onward (inclusive), after sorting by filename within the sequence.
            sequence_scan_end: If set, exclusive end index (Python slice). E.g. start=0,
                end=600 uses scans 0..599. Applied before intra-sequence train/val/test split.
        """
        super().__init__()
        
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.max_points = max_points
        self.use_ground_truth_maps = use_ground_truth_maps
        self.gt_subdir = gt_subdir.strip("/\\") if gt_subdir else "ground_truth"
        self.gt_name_suffix = gt_name_suffix or ""
        self.augmentation = augmentation
        self.num_points_per_scan = num_points_per_scan
        self.use_precomputed = use_precomputed
        self.split_preset = split_preset
        self.intra_seq_val_fraction = float(intra_seq_val_fraction)
        self.intra_seq_test_fraction = float(intra_seq_test_fraction)
        self.scene_radius = float(scene_radius)

        if sequence_ids is not None:
            self.sequences = [s.strip() for s in sequence_ids if str(s).strip()]
            if not self.sequences:
                raise ValueError("sequence_ids expanded to an empty list")
        elif split_preset == 'official':
            self.sequences = list(self.SPLITS_OFFICIAL[split])
        else:
            self.sequences = list(self.SPLITS[split])
        
        # Build file lists
        self.scan_files = []
        self.label_files = []
        self.pose_files = []
        self.gt_map_files = []
        
        self._build_file_lists()

        self._apply_sequence_scan_window(sequence_scan_start, sequence_scan_end)

        if self.intra_seq_test_fraction > 0:
            if self.intra_seq_val_fraction <= 0 or self.intra_seq_val_fraction >= 1.0:
                raise ValueError(
                    "intra_seq_test_fraction>0 requires intra_seq_val_fraction in (0,1) "
                    "(middle val segment)"
                )
            tr = 1.0 - self.intra_seq_val_fraction - self.intra_seq_test_fraction
            if tr <= 0:
                raise ValueError(
                    "Train fraction 1 - val - test must be positive; "
                    "reduce intra_seq_val_fraction or intra_seq_test_fraction"
                )
            if len(self.sequences) != 1:
                raise ValueError(
                    "Three-way intra-sequence split requires exactly one sequence in sequence_ids"
                )
            self._apply_intra_sequence_three_way()
        elif self.intra_seq_val_fraction > 0:
            if self.split == 'test':
                raise ValueError(
                    "intra_seq_val_fraction (two-way) is only for train/val; "
                    "use intra_seq_test_fraction>0 for a test segment"
                )
            if len(self.sequences) != 1:
                raise ValueError(
                    "intra_seq_val_fraction requires exactly one sequence "
                    "(pass sequence_ids='03' or similar)"
                )
            self._apply_intra_sequence_holdout()

        # Filter to only samples with existing precomputed features
        if self.use_precomputed:
            valid = [i for i, p in enumerate(self.precomputed_files) if os.path.exists(p)]
            self.scan_files = [self.scan_files[i] for i in valid]
            self.label_files = [self.label_files[i] for i in valid]
            self.pose_files = [self.pose_files[i] for i in valid]
            if self.use_ground_truth_maps:
                self.gt_map_files = [self.gt_map_files[i] for i in valid]
            self.precomputed_files = [self.precomputed_files[i] for i in valid]
        
        print(f"Loaded SemanticKITTI {split} split:")
        print(f"  Sequences: {self.sequences}")
        print(f"  Total scans: {len(self.scan_files)}")
        if self.use_ground_truth_maps and len(self.gt_map_files) > 0:
            print(
                f"  GT maps: .../{self.gt_subdir}/<seq>/*{self.gt_name_suffix}.npz"
            )
        if self.scene_radius > 0:
            print(f"  Scene crop radius: {self.scene_radius} m (LiDAR + GT, before voxelize)")
    
    def _build_file_lists(self):
        """Build lists of data files."""
        for seq in self.sequences:
            seq_path = os.path.join(self.root, 'sequences', seq)
            
            # Scan files
            scan_dir = os.path.join(seq_path, 'velodyne')
            if not os.path.exists(scan_dir):
                print(f"Warning: {scan_dir} does not exist")
                continue
            
            scan_files = sorted(os.listdir(scan_dir))
            
            for scan_file in scan_files:
                if not scan_file.endswith('.bin'):
                    continue
                
                scan_id = scan_file.replace('.bin', '')
                
                # Full paths
                scan_path = os.path.join(scan_dir, scan_file)
                label_path = os.path.join(
                    seq_path, 'labels', f'{scan_id}.label'
                )
                pose_path = os.path.join(seq_path, 'poses.txt')
                
                # Ground truth map (if using pre-generated)
                if self.use_ground_truth_maps:
                    gt_map_path = os.path.join(
                        self.root,
                        self.gt_subdir,
                        seq,
                        f'{scan_id}{self.gt_name_suffix}.npz',
                    )
                    self.gt_map_files.append(gt_map_path)
                
                if not hasattr(self, "precomputed_files"):
                    self.precomputed_files = []
                precomp_path = os.path.join(
                    self.root, "precomputed_v2", seq, f"{scan_id}.npz"
                )
                self.precomputed_files.append(precomp_path)

                self.scan_files.append(scan_path)
                self.label_files.append(label_path)
                self.pose_files.append(pose_path)

    def _apply_intra_sequence_holdout(self) -> None:
        """Time-ordered holdout within a single sequence (sorted scan ids)."""
        n = len(self.scan_files)
        if n == 0:
            return
        f = self.intra_seq_val_fraction
        if f <= 0.0 or f >= 1.0:
            raise ValueError("intra_seq_val_fraction must be in (0, 1)")
        cut = int(round(n * (1.0 - f)))
        cut = max(1, min(n - 1, cut))
        if self.split == 'train':
            idx_range = slice(0, cut)
        else:
            idx_range = slice(cut, n)
        self._slice_sample_lists(idx_range)

    def _apply_intra_sequence_three_way(self) -> None:
        """Time-ordered train | val | test within one sequence (sorted scan ids)."""
        n = len(self.scan_files)
        if n == 0:
            return
        train_f = 1.0 - self.intra_seq_val_fraction - self.intra_seq_test_fraction
        cut1 = int(round(n * train_f))
        cut2 = int(round(n * (train_f + self.intra_seq_val_fraction)))
        cut1 = max(1, min(cut1, n - 2))
        cut2 = max(cut1 + 1, min(cut2, n - 1))
        if self.split == 'train':
            idx_range = slice(0, cut1)
        elif self.split == 'val':
            idx_range = slice(cut1, cut2)
        else:
            idx_range = slice(cut2, n)
        self._slice_sample_lists(idx_range)

    def _apply_sequence_scan_window(
        self,
        start: Optional[int],
        end: Optional[int],
    ) -> None:
        """Restrict to ``[start, end)`` scan indices within the only sequence (after sort)."""
        if start is None and end is None:
            return
        if len(self.sequences) != 1:
            raise ValueError(
                "sequence_scan_start / sequence_scan_end require exactly one sequence "
                "(pass sequence_ids with a single id, e.g. ['02'])"
            )
        n = len(self.scan_files)
        s = 0 if start is None else int(start)
        e = n if end is None else int(end)
        if s < 0 or e < 0:
            raise ValueError("sequence_scan_start / sequence_scan_end must be non-negative")
        if s > n:
            raise ValueError(f"sequence_scan_start {s} is past num scans {n}")
        e = min(e, n)
        if e <= s:
            raise ValueError(
                f"Empty sequence scan window: start={s} end={e} (end is exclusive; need end > start)"
            )
        self._slice_sample_lists(slice(s, e))
        print(f"  Sequence scan window: indices [{s}, {e}) → {e - s} scans")

    def _slice_sample_lists(self, idx_range: slice) -> None:
        self.scan_files = self.scan_files[idx_range]
        self.label_files = self.label_files[idx_range]
        self.pose_files = self.pose_files[idx_range]
        if self.use_ground_truth_maps:
            self.gt_map_files = self.gt_map_files[idx_range]
        if hasattr(self, "precomputed_files"):
            self.precomputed_files = self.precomputed_files[idx_range]

    def __len__(self) -> int:
        return len(self.scan_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a single sample.
        
        Returns:
            Dictionary containing:
                - partial_scan: Incomplete LiDAR scan
                - complete_scene: Ground truth complete scene
                - semantic_labels: Semantic class labels
                - metadata: Additional information
        """
        # Fast path: load precomputed_v2 data
        if self.use_precomputed and hasattr(self, "precomputed_files"):
            precomp_path = self.precomputed_files[idx]
            if os.path.exists(precomp_path):
                precomp = np.load(precomp_path)
                cc = precomp["complete_coord"].astype(np.float32)
                cl = precomp["complete_labels"].astype(np.int64)
                if self.scene_radius > 0 and cc.shape[0] > 0:
                    d = np.linalg.norm(cc.astype(np.float64), axis=1)
                    m = d <= self.scene_radius + 1e-5
                    cc = cc[m]
                    cl = cl[m]
                cf = precomp["condition_features"]
                sc = precomp["scan_center"].astype(np.float32)
                return {
                    "partial_coord": torch.zeros(1, 3),
                    "partial_color": torch.zeros(1, 3),
                    "partial_normal": torch.zeros(1, 3),
                    "partial_labels": torch.zeros(1, dtype=torch.long),
                    "complete_coord": torch.from_numpy(cc),
                    "complete_color": torch.zeros(len(cc), 3),
                    "complete_labels": torch.from_numpy(cl),
                    "scan_center": torch.from_numpy(sc),
                    "condition_features": torch.from_numpy(cf.copy()),
                    "idx": idx,
                }

        # Load scan
        scan = self._load_scan(self.scan_files[idx])
        
        # Load labels
        if os.path.exists(self.label_files[idx]):
            labels = self._load_labels(self.label_files[idx])
        else:
            labels = np.zeros(scan.shape[0], dtype=np.int32)
        
        # Load or generate ground truth
        if self.use_ground_truth_maps and \
           os.path.exists(self.gt_map_files[idx]):
            gt_complete = self._load_gt_map(self.gt_map_files[idx])
        else:
            # Use scan itself as GT for testing
            gt_complete = scan.copy()

        if self.scene_radius > 0:
            scan, labels = crop_lidar_radius_with_labels(
                scan, labels, self.scene_radius
            )
            gt_complete = crop_lidar_radius(gt_complete, self.scene_radius)
        
        # Subsample scan if needed
        if self.num_points_per_scan is not None:
            scan, labels = self._subsample_scan(
                scan, labels, self.num_points_per_scan
            )
        
        # Apply augmentation
        if self.augmentation and self.split == 'train':
            scan, gt_complete, labels = self._augment(
                scan, gt_complete, labels
            )
        
        # LiDAR: use sensor origin (0,0,0), not mean — scene asymmetric, ego not at centroid.
        scan_center = np.zeros(3, dtype=np.float32)
        scan = scan - scan_center
        gt_complete = gt_complete - scan_center
        
        # Voxelize
        scan_voxel, scan_labels = self._voxelize(scan, labels)
        gt_labels_dummy = np.zeros(gt_complete.shape[0], dtype=np.int32)
        gt_voxel, gt_labels = self._voxelize(gt_complete, gt_labels_dummy)
        
        # Limit number of points
        if scan_voxel.shape[0] > self.max_points:
            indices = np.random.choice(
                scan_voxel.shape[0], 
                self.max_points, 
                replace=False
            )
            scan_voxel = scan_voxel[indices]
            scan_labels = scan_labels[indices]
        
        if gt_voxel.shape[0] > self.max_points:
            indices = np.random.choice(
                gt_voxel.shape[0], 
                self.max_points, 
                replace=False
            )
            gt_voxel = gt_voxel[indices]
            gt_labels = gt_labels[indices]
        
        # Convert to tensors
        data = {
            'partial_coord': torch.from_numpy(scan_voxel).float(),
            'partial_labels': torch.from_numpy(scan_labels).long(),
            'complete_coord': torch.from_numpy(gt_voxel).float(),
            'complete_labels': torch.from_numpy(gt_labels).long(),
            'scan_center': torch.from_numpy(scan_center).float(),
            'idx': idx,
        }
        
        # Add color if available (use height as color for now)
        partial_color = self._height_to_color(scan_voxel)
        complete_color = self._height_to_color(gt_voxel)
        
        data['partial_color'] = torch.from_numpy(partial_color).float()
        data['complete_color'] = torch.from_numpy(complete_color).float()
        
        # Add normals (computed from local neighbors)
        data['partial_normal'] = torch.zeros_like(data['partial_coord'])
        data['complete_normal'] = torch.zeros_like(data['complete_coord'])
        
        return data
    
    def _load_scan(self, scan_path: str) -> np.ndarray:
        """Load LiDAR scan from binary file."""
        scan = np.fromfile(scan_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))  # x, y, z, intensity
        return scan[:, :3]  # Return only xyz
    
    def _load_labels(self, label_path: str) -> np.ndarray:
        """Load semantic labels."""
        labels = np.fromfile(label_path, dtype=np.uint32)
        labels = labels.reshape((-1))
        labels = labels & 0xFFFF  # Remove instance info
        
        # Map to learning classes
        labels_mapped = np.vectorize(
            lambda x: self.LEARNING_MAP.get(x, 0)
        )(labels)
        
        return labels_mapped.astype(np.int32)
    
    def _load_gt_map(self, gt_path: str) -> np.ndarray:
        """Load pre-generated ground truth complete map."""
        data = np.load(gt_path)
        return data['points']
    
    def _subsample_scan(
        self,
        scan: np.ndarray,
        labels: np.ndarray,
        num_points: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Randomly subsample scan."""
        if scan.shape[0] <= num_points:
            return scan, labels
        
        indices = np.random.choice(
            scan.shape[0], num_points, replace=False
        )
        return scan[indices], labels[indices]
    
    def _voxelize(
        self,
        points: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Voxelize point cloud.
        
        Returns voxel centers and majority-vote labels.
        """
        # Compute voxel coordinates
        voxel_coords = np.floor(points / self.voxel_size).astype(np.int32)
        
        # Get unique voxels
        unique_voxels, inverse_indices = np.unique(
            voxel_coords, axis=0, return_inverse=True
        )
        
        # Compute voxel centers
        voxel_centers = unique_voxels * self.voxel_size + self.voxel_size / 2
        
        # Majority vote for labels (vectorized)
        voxel_labels = np.zeros(unique_voxels.shape[0], dtype=np.int32)
        if labels.any():
            # Use vectorized bincount per voxel
            num_voxels = unique_voxels.shape[0]
            num_classes = int(labels.max()) + 1
            counts = np.zeros((num_voxels, num_classes), dtype=np.int32)
            np.add.at(counts, (inverse_indices, labels), 1)
            voxel_labels = counts.argmax(axis=1).astype(np.int32)
        
        return voxel_centers, voxel_labels
    
    def _augment(
        self,
        partial: np.ndarray,
        complete: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply data augmentation."""
        # Random rotation around z-axis
        angle = np.random.uniform(-np.pi, np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        partial = partial @ rot_matrix.T
        complete = complete @ rot_matrix.T
        
        # Random flip
        if np.random.rand() > 0.5:
            partial[:, 1] = -partial[:, 1]
            complete[:, 1] = -complete[:, 1]
        
        # Random scaling
        scale = np.random.uniform(0.95, 1.05)
        partial = partial * scale
        complete = complete * scale
        
        # Random jittering
        partial += np.random.randn(*partial.shape) * 0.01
        
        return partial, complete, labels
    
    def _height_to_color(self, points: np.ndarray) -> np.ndarray:
        """Convert height to RGB color for visualization."""
        z = points[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
        
        # Create color gradient (blue -> green -> red)
        colors = np.zeros((points.shape[0], 3))
        colors[:, 0] = z_norm  # Red
        colors[:, 1] = 1 - np.abs(z_norm - 0.5) * 2  # Green
        colors[:, 2] = 1 - z_norm  # Blue
        
        return colors


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching samples.
    
    Creates batch indices for sparse tensor operations.
    """
    # Collect all data
    partial_coords = []
    partial_colors = []
    partial_normals = []
    partial_labels = []
    
    complete_coords = []
    complete_colors = []
    complete_labels = []
    
    batch_indices_partial = []
    batch_indices_complete = []
    
    for i, data in enumerate(batch):
        partial_coords.append(data['partial_coord'])
        partial_colors.append(data['partial_color'])
        partial_normals.append(data['partial_normal'])
        partial_labels.append(data['partial_labels'])
        
        complete_coords.append(data['complete_coord'])
        complete_colors.append(data['complete_color'])
        complete_labels.append(data['complete_labels'])
        
        # Batch indices
        batch_indices_partial.append(
            torch.full((data['partial_coord'].shape[0],), i, dtype=torch.long)
        )
        batch_indices_complete.append(
            torch.full((data['complete_coord'].shape[0],), i, dtype=torch.long)
        )
    
    # Concatenate
    batch_data = {
        'partial_coord': torch.cat(partial_coords, dim=0),
        'partial_color': torch.cat(partial_colors, dim=0),
        'partial_normal': torch.cat(partial_normals, dim=0),
        'partial_labels': torch.cat(partial_labels, dim=0),
        'partial_batch': torch.cat(batch_indices_partial, dim=0),
        
        'complete_coord': torch.cat(complete_coords, dim=0),
        'complete_color': torch.cat(complete_colors, dim=0),
        'complete_labels': torch.cat(complete_labels, dim=0),
        'complete_batch': torch.cat(batch_indices_complete, dim=0),
        
        'scan_center': torch.stack([d['scan_center'] for d in batch]),
        'idx': torch.tensor([d['idx'] for d in batch]),
    }

    # Add condition_features if present (from precomputed data)
    if 'condition_features' in batch[0]:
        batch_data['condition_features'] = torch.cat(
            [d['condition_features'] for d in batch], dim=0
        )
    
    return batch_data


if __name__ == "__main__":
    # Test dataset
    print("Testing SemanticKITTI dataset...")
    
    dataset = SemanticKITTI(
        root='Datasets/SemanticKITTI/dataset',
        split='train',
        voxel_size=0.1,
        use_ground_truth_maps=False,  # Test without GT maps first
        augmentation=True
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Load sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Partial scan shape: {sample['partial_coord'].shape}")
    print(f"Complete scene shape: {sample['complete_coord'].shape}")
    print(f"Labels shape: {sample['partial_labels'].shape}")
    print(f"Unique labels: {torch.unique(sample['partial_labels'])}")
    
    # Test collate
    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=collate_fn
    )
    
    batch = next(iter(loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Batch partial coords: {batch['partial_coord'].shape}")
    print(f"Batch indices: {torch.unique(batch['partial_batch'])}")
