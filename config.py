"""
Configuration module for InteriorGS Question Generation Pipeline

This module defines configuration dataclasses for object selection,
camera sampling, and the overall pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from pathlib import Path


@dataclass
class ObjectSelectionConfig:
    """Configuration for object selection/filtering from InteriorGS labels.json."""
    
    # Semantic blacklist for excluded object categories
    blacklist: Set[str] = field(default_factory=lambda: {
        # Structural elements
        "wall", "floor", "ceiling", "room", "door", "window",
        # Carpet variations
        "carpet", "rug",
        # Light fixtures
        "chandelier", "ceiling lamp", "spotlight", "lamp", "light",
        "downlights", "wall lamp", "table lamp", "strip light", "track light",
        "linear lamp", "decorative pendant",
        # Generic / unclear categories
        "other", "curtain", "unknown",
        # Small items that appear in large quantities
        "book", "boxed food", "bagged food", "medicine box",
        "vegetable", "fruit", "drinks", "canned food",
        "pen", "medicine bottle", "toiletries", "chocolate", "paper",
        "bread", "cigar", "wine", "fresh food",
    })
    
    # Geometric filter toggles (set to False to disable)
    enable_dim_filter: bool = False  # Enable min/max dimension filtering
    enable_volume_filter: bool = False  # Enable minimum volume filtering
    enable_aspect_ratio_filter: bool = False  # Enable aspect ratio filtering
    
    # Geometric constraints (only applied if corresponding filter is enabled)
    min_dim_component: float = 0.1  # Minimum dimension per axis (m)
    max_dim_component: float = 3.0  # Maximum dimension per axis (m)
    min_volume: float = 0.01  # Minimum volume (m^3)
    min_aspect_ratio: float = 0.05  # Min shortest/longest edge ratio
    
    # Pair constraints
    min_pair_dist: float = 0.3  # Minimum distance between paired objects
    max_pair_dist: float = 5.0  # Maximum distance between paired objects
    max_pair_dim_ratio: float = 3.0  # Max ratio of longest edges
    max_pair_dim_diff: float = 2.5  # Max absolute difference in longest edge
    
    # Dynamic thresholds
    dyn_min_mult: float = 0.2  # Multiplier for dynamic min distance
    dyn_max_mult: float = 2.0  # Multiplier for dynamic max distance


@dataclass
class CameraSamplingConfig:
    """Configuration for camera pose sampling in InteriorGS scenes.
    
    Move Patterns:
        - 'around': Horizontal circle - sample cameras around object on horizontal plane
        - 'spherical': Spherical sampling - sample on sphere surface (varied heights/angles)
        - 'rotation': Room rotation - stand at room center, rotate 360 degrees
        - 'linear': Linear trajectory - walk toward or past object in straight line
    """
    
    # Number of camera poses to sample per object/pair
    num_cameras_per_item: int = 3
    
    # Move pattern: 'around', 'spherical', 'rotation', or 'linear'
    # - 'around': Horizontal circle around object (default)
    # - 'spherical': Sample on sphere surface around object (varied heights)
    # - 'rotation': Stand at room center, rotate 360° (room-centric)
    # - 'linear': Walk toward or past object in straight line
    move_pattern: str = 'around'
    
    # Sampling parameters
    per_angle: int = 36  # Number of angles to try per radius
    max_tries: int = 300  # Maximum sampling attempts
    spherical_samples: int = 30  # Number of samples for spherical pattern
    
    # Camera height constraints (m)
    max_camera_height: float = 1.8  # Maximum camera height above ground
    min_camera_height: float = 0.8  # Minimum camera height above ground
    camera_height_offset: float = 0.2  # Height offset above object top
    
    # Distance from objects (m)
    min_distance: float = 0.5  # Minimum distance from camera to object
    max_distance: float = 4.0  # Maximum distance from camera to object
    
    # Visibility thresholds
    min_visibility_ratio: float = 0.05  # Minimum visible area ratio (5% of image)
    max_occlusion_ratio: float = 0.6  # Maximum allowed occlusion (60%)
    min_visible_corners: int = 2  # Minimum visible bbox corners
    skip_occlusion_check: bool = False  # Skip full occlusion check for better success rate
    check_wall_occlusion: bool = True  # Check wall occlusion
    
    # Camera intrinsics
    image_width: int = 512
    image_height: int = 512
    fov_deg: float = 60.0  # Field of view in degrees
    
    # ===== Rotation mode parameters (move_pattern='rotation') =====
    # Stand at room center, rotate 360° to look around
    rotation_interval: float = 5.0      # Degrees between each camera pose (360/5 = 72 images)
    rotation_camera_height: float = 1.5  # Fixed camera height (m)
    
    # ===== Linear mode parameters (move_pattern='linear') =====
    # Walk toward or past an object in a straight line
    # Camera orientation (yaw/pitch) remains FIXED throughout trajectory
    # Only camera POSITION changes along a linear path
    # Sub-patterns:
    #   - 'approach': Walk toward object in a straight line (forward walk)
    #   - 'pass_by': Walk along a line that passes by the object (side walk)
    linear_sub_pattern: str = 'approach'
    linear_num_steps: int = 5           # Number of poses along trajectory
    linear_fov_margin: float = 0.1      # FOV margin for visibility check
    linear_move_distance: float = 0.3   # Total movement distance along trajectory (meters)


@dataclass
class QuestionConfig:
    """Configuration for question generation."""
    
    # Question types to generate
    enabled_question_types: List[str] = field(default_factory=lambda: [
        # Single object questions
        "object_size",
        "object_distance_to_camera",
        # Pair object questions
        "object_size_comparison_relative",
        "object_size_comparison_absolute",
        "object_pair_distance_center",
        "object_pair_distance_center_w_size",
        # Multi-object questions
        "object_comparison_absolute_distance",
        "object_comparison_relative_distance",
        # Vector questions
        "object_pair_distance_vector",
    ])
    
    # Dimensions to use for size comparisons
    dimensions: List[str] = field(default_factory=lambda: ['length', 'width', 'height'])
    
    # Number of questions per type per camera pose
    max_questions_per_type: int = 5


@dataclass
class RenderConfig:
    """Configuration for rendering images from camera poses."""
    
    # Enable/disable rendering
    enable_rendering: bool = True
    
    # Gaussian Splatting scene root (can be same as scenes_root if PLY files are in scene folders)
    gs_root: str = ""
    
    # Rendering backend
    render_backend: str = "local"  # "local" or "client"
    client_url: str = None  # WebSocket URL for client mode
    
    # Image dimensions (should match camera_sampling config)
    image_width: int = 640
    image_height: int = 480
    fov_deg: float = 60.0
    
    # GPU device
    gpu_device: int = None  # None = auto


@dataclass 
class PipelineConfig:
    """Main configuration for the entire pipeline."""
    
    # Input/Output paths
    scenes_root: str = ""  # Root directory containing InteriorGS scene folders
    output_dir: str = ""  # Output directory for generated data
    experiment_name: str = "default"  # Experiment name for output directory structure
    
    # Scene selection
    scene_list: Optional[List[str]] = None  # List of scenes to process (None = all)
    
    # Sub-configs
    object_selection: ObjectSelectionConfig = field(default_factory=ObjectSelectionConfig)
    camera_sampling: CameraSamplingConfig = field(default_factory=CameraSamplingConfig)
    question_config: QuestionConfig = field(default_factory=QuestionConfig)
    render_config: RenderConfig = field(default_factory=RenderConfig)
    
    # Processing options
    save_intermediate: bool = True  # Save intermediate results per scene
    max_questions_per_scene: int = 1000  # Maximum questions per scene
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create config from dictionary."""
        obj_sel = ObjectSelectionConfig(**config_dict.pop('object_selection', {}))
        cam_samp = CameraSamplingConfig(**config_dict.pop('camera_sampling', {}))
        q_cfg = QuestionConfig(**config_dict.pop('question_config', {}))
        r_cfg = RenderConfig(**config_dict.pop('render_config', {}))
        
        return cls(
            object_selection=obj_sel,
            camera_sampling=cam_samp,
            question_config=q_cfg,
            render_config=r_cfg,
            **config_dict
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        import dataclasses
        result = dataclasses.asdict(self)
        # Convert sets to lists for JSON serialization
        if 'object_selection' in result and 'blacklist' in result['object_selection']:
            result['object_selection']['blacklist'] = list(result['object_selection']['blacklist'])
        return result
    
    def validate(self) -> bool:
        """Validate configuration."""
        if not self.scenes_root:
            raise ValueError("scenes_root must be specified")
        if not self.output_dir:
            raise ValueError("output_dir must be specified")
        if not Path(self.scenes_root).exists():
            raise ValueError(f"scenes_root does not exist: {self.scenes_root}")
        return True
