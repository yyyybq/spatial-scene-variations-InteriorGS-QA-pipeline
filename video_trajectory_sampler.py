"""
Video Trajectory Sampler for InteriorGS

Generates dense camera trajectories (30 frames each) for video rendering.
For each (object/pair, pattern) combination, generates N distinct trajectories.

Supported patterns:
  - around:    Arc trajectory around object on horizontal plane
  - spherical: Arc trajectory on sphere surface
  - linear:    Straight-line walk (approach or pass_by)
  - rotation:  In-place rotation at room center
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from scipy.interpolate import CubicSpline

from config import CameraSamplingConfig, VideoTrajectoryConfig
from object_selector import SceneObject, ObjectSelector
from camera_utils import (
    SceneBounds, AABB, CameraPose,
    is_target_in_fov, is_wall_occluded, count_visible_corners,
)
from camera_sampler import CameraSampler, scene_object_to_aabb


class VideoTrajectorySampler:
    """Generates multiple dense camera trajectories per (object, pattern) for video."""

    def __init__(self, cam_config: CameraSamplingConfig, video_config: VideoTrajectoryConfig):
        self.cam_cfg = cam_config
        self.vid_cfg = video_config
        # Reuse CameraSampler for scene loading and validation utilities
        self._sampler = CameraSampler(cam_config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_trajectories(
        self,
        scene_path: Path,
        objects: List[SceneObject],
        all_scene_objects: Optional[List[SceneObject]] = None,
        pattern: Optional[str] = None,
    ) -> List[List[CameraPose]]:
        """Generate *num_trajectories* distinct dense trajectories.

        Returns:
            List of trajectories, each trajectory is a list of CameraPose (len = num_frames).
        """
        pat = pattern or self.cam_cfg.move_pattern
        num_traj = self.vid_cfg.num_trajectories
        num_frames = self.vid_cfg.num_frames

        # Load scene constraints once
        scene_bounds = self._sampler.load_scene_bounds(scene_path)
        room_polys = self._sampler.load_room_polys(scene_path)
        wall_aabbs = self._sampler.load_wall_aabbs(scene_path)
        collision_objects = all_scene_objects if all_scene_objects else objects

        # Target center
        if len(objects) == 1:
            target = objects[0].center.copy()
        else:
            target = np.mean([obj.center for obj in objects], axis=0)

        # AABBs for visibility
        all_aabbs = [scene_object_to_aabb(obj) for obj in collision_objects]
        if wall_aabbs:
            all_aabbs.extend(wall_aabbs)
        target_ids = {str(obj.id) for obj in objects}

        ctx = dict(
            scene_bounds=scene_bounds,
            room_polys=room_polys,
            wall_aabbs=wall_aabbs,
            collision_objects=collision_objects,
            all_aabbs=all_aabbs,
            target_ids=target_ids,
            target=target,
            objects=objects,
        )

        trajectories: List[List[CameraPose]] = []
        base_attempts = self.vid_cfg.max_trajectory_attempts

        for traj_idx in range(num_traj):
            traj = None

            # Adaptive fallback stages for around: progressively easier settings.
            if pat == "around":
                phases = [
                    {"attempts": base_attempts, "kwargs": {}},
                    {"attempts": base_attempts, "kwargs": {"force_num_passes": 1}},
                    {"attempts": base_attempts, "kwargs": {"force_num_passes": 1, "arc_scale": 0.7}},
                    {
                        "attempts": base_attempts,
                        "kwargs": {"force_num_passes": 1, "arc_scale": 0.5, "skip_wall_occlusion": True},
                    },
                ]
            else:
                phases = [{"attempts": base_attempts, "kwargs": {}}]

            for phase_idx, phase in enumerate(phases):
                for _attempt in range(phase["attempts"]):
                    # Keep deterministic but diversified sampling across trajectory/phase/retry.
                    seed = 42 + traj_idx * 137 + phase_idx * 100003 + _attempt * 7919
                    rng = np.random.RandomState(seed=seed)

                    if pat == "around":
                        traj = self._gen_around(rng, ctx, num_frames, **phase["kwargs"])
                    elif pat == "spherical":
                        traj = self._gen_spherical(rng, ctx, num_frames)
                    elif pat == "linear":
                        traj = self._gen_linear(rng, ctx, num_frames)
                    elif pat == "rotation":
                        traj = self._gen_rotation(rng, ctx, num_frames, scene_path)
                    else:
                        raise ValueError(f"Unknown pattern: {pat}")

                    if traj is not None and len(traj) == num_frames:
                        break
                    traj = None
                if traj is not None:
                    break

            if traj is None:
                traj = self._build_static_fallback_trajectory(
                    scene_path=scene_path,
                    objects=objects,
                    all_scene_objects=all_scene_objects,
                    pattern=pat,
                    num_frames=num_frames,
                )

            if traj is None:
                traj = self._build_last_resort_trajectory(ctx=ctx, num_frames=num_frames)

            if traj is not None:
                trajectories.append(traj)
            else:
                print(f"  [WARN] Could not construct trajectory {traj_idx} for pattern={pat}")

        # Final safety net: always return exactly num_traj trajectories.
        if len(trajectories) < num_traj and trajectories:
            deficit = num_traj - len(trajectories)
            print(f"  [WARN] Backfilling {deficit} trajectories for pattern={pat}")
            base_trajs = [self._clone_trajectory(t) for t in trajectories]
            idx = 0
            while len(trajectories) < num_traj:
                trajectories.append(self._clone_trajectory(base_trajs[idx % len(base_trajs)]))
                idx += 1

        return trajectories[:num_traj]

    # ------------------------------------------------------------------
    # Pattern: around — arc on horizontal circle
    # ------------------------------------------------------------------

    def _gen_around(self, rng: np.random.RandomState, ctx: dict,
                    num_frames: int,
                    force_num_passes: Optional[int] = None,
                    arc_scale: float = 1.0,
                    skip_wall_occlusion: bool = False) -> Optional[List[CameraPose]]:
        target = ctx["target"]
        objects = ctx["objects"]
        min_dist = self.cam_cfg.min_distance
        max_dist = self.cam_cfg.max_distance
        camera_height = self._sampler.compute_camera_height(objects)

        # For multiple objects, use larger radius and smaller arc to keep all in view
        if len(objects) > 1:
            obj_spread = max(
                np.linalg.norm(o1.center[:2] - o2.center[:2])
                for o1 in objects for o2 in objects if o1 is not o2
            )
            # Ensure camera is far enough to see both objects
            min_dist = max(min_dist, obj_spread * 0.8)
            max_dist = max(max_dist, obj_spread * 2.0)
            # Smaller arc for wider spreads
            arc_range = min(self.vid_cfg.around_arc_range_deg, 60.0 + 60.0 / (1 + obj_spread))
        else:
            arc_range = self.vid_cfg.around_arc_range_deg

        # Random parameters
        radius = rng.uniform(min_dist, max_dist)
        # Discrete starting angles: {0, 30, 60, ..., 330}
        start_yaw_deg = float(rng.choice(np.arange(0, 360, 30)))
        # Randomly choose CW or CCW
        direction = rng.choice([-1, 1])
        # Randomly choose 1 or 2 passes unless forced by fallback stage.
        num_passes = int(force_num_passes) if force_num_passes is not None else int(rng.choice([1, 2]))
        effective_arc = max(15.0, arc_range * arc_scale)
        end_yaw_deg = start_yaw_deg + direction * effective_arc * num_passes

        yaw_sequence = np.linspace(start_yaw_deg, end_yaw_deg, num_frames)

        poses: List[CameraPose] = []
        for yaw_deg in yaw_sequence:
            yaw_rad = np.radians(yaw_deg)
            cam_x = target[0] + radius * np.cos(yaw_rad)
            cam_y = target[1] + radius * np.sin(yaw_rad)
            position = np.array([cam_x, cam_y, camera_height])

            if not self._sampler.is_position_valid(
                position, ctx["scene_bounds"], ctx["room_polys"], ctx["collision_objects"]
            ):
                return None

            dz = target[2] - position[2]
            hdist = np.sqrt((target[0] - position[0])**2 + (target[1] - position[1])**2)
            pitch = np.degrees(np.arctan2(-dz, hdist))

            pose = CameraPose(
                position=position,
                target=target.copy(),
                yaw=float(yaw_deg % 360),
                pitch=float(pitch),
                radius=float(radius),
                target_objects=[o.label for o in objects],
            )

            if not self._validate_visibility_relaxed(pose, ctx, skip_wall_occlusion=skip_wall_occlusion):
                return None
            poses.append(pose)

        return poses

    # ------------------------------------------------------------------
    # Pattern: spherical — arc on sphere surface
    # ------------------------------------------------------------------

    def _gen_spherical(self, rng: np.random.RandomState, ctx: dict,
                       num_frames: int) -> Optional[List[CameraPose]]:
        target = ctx["target"]
        objects = ctx["objects"]
        min_dist = self.cam_cfg.min_distance
        max_dist = self.cam_cfg.max_distance

        # Adaptive radius for multi-object
        if len(objects) > 1:
            obj_spread = max(
                np.linalg.norm(o1.center[:2] - o2.center[:2])
                for o1 in objects for o2 in objects if o1 is not o2
            )
            min_dist = max(min_dist, obj_spread * 0.8)
            max_dist = max(max_dist, obj_spread * 2.0)
            arc_deg = min(self.vid_cfg.spherical_arc_range_deg, 50.0 + 40.0 / (1 + obj_spread))
        else:
            arc_deg = self.vid_cfg.spherical_arc_range_deg

        radius = rng.uniform(min_dist, max_dist)

        # Sample two points on the sphere and interpolate great-circle-style
        theta_start = rng.uniform(0, 2 * np.pi)
        phi_start = np.arccos(rng.uniform(0.2, 0.9))  # avoid poles

        delta_theta = np.radians(rng.uniform(-arc_deg, arc_deg))
        delta_phi = np.radians(rng.uniform(-30, 30))

        theta_end = theta_start + delta_theta
        phi_end = np.clip(phi_start + delta_phi, 0.3, np.pi - 0.3)

        thetas = np.linspace(theta_start, theta_end, num_frames)
        phis = np.linspace(phi_start, phi_end, num_frames)

        poses: List[CameraPose] = []
        for theta, phi in zip(thetas, phis):
            cam_x = target[0] + radius * np.sin(phi) * np.cos(theta)
            cam_y = target[1] + radius * np.sin(phi) * np.sin(theta)
            cam_z = target[2] + radius * np.cos(phi)
            position = np.array([cam_x, cam_y, cam_z])

            # Height check
            if position[2] < self.cam_cfg.min_camera_height or position[2] > self.cam_cfg.max_camera_height:
                return None

            if not self._sampler.is_position_valid(
                position, ctx["scene_bounds"], ctx["room_polys"], ctx["collision_objects"]
            ):
                return None

            dz = target[2] - position[2]
            hdist = np.sqrt((target[0] - position[0])**2 + (target[1] - position[1])**2)
            pitch = np.degrees(np.arctan2(-dz, hdist))
            yaw = np.degrees(np.arctan2(
                target[1] - position[1], target[0] - position[0]
            ))

            pose = CameraPose(
                position=position,
                target=target.copy(),
                yaw=float(yaw % 360),
                pitch=float(pitch),
                radius=float(radius),
                target_objects=[o.label for o in objects],
            )

            if not self._validate_visibility_relaxed(pose, ctx):
                return None
            poses.append(pose)

        return poses

    # ------------------------------------------------------------------
    # Pattern: linear — straight-line walk
    # ------------------------------------------------------------------

    def _gen_linear(self, rng: np.random.RandomState, ctx: dict,
                    num_frames: int) -> Optional[List[CameraPose]]:
        target = ctx["target"]
        objects = ctx["objects"]
        min_dist = self.cam_cfg.min_distance
        max_dist = self.cam_cfg.max_distance
        camera_height = self._sampler.compute_camera_height(objects)
        total_dist = self.vid_cfg.linear_total_distance

        # Adaptive for multi-object
        if len(objects) > 1:
            obj_spread = max(
                np.linalg.norm(o1.center[:2] - o2.center[:2])
                for o1 in objects for o2 in objects if o1 is not o2
            )
            min_dist = max(min_dist, obj_spread * 0.8)
            max_dist = max(max_dist, obj_spread * 2.0)

        sub_pat = getattr(self.cam_cfg, "linear_sub_pattern", "approach")

        initial_yaw = rng.uniform(0, 360)
        initial_yaw_rad = np.radians(initial_yaw)
        dir_from_target = np.array([np.cos(initial_yaw_rad), np.sin(initial_yaw_rad), 0.0])

        if sub_pat == "approach":
            start_radius = rng.uniform(min_dist + total_dist, max_dist)
            end_radius = start_radius - total_dist

            start_pos = np.array([
                target[0] + start_radius * dir_from_target[0],
                target[1] + start_radius * dir_from_target[1],
                camera_height,
            ])
            end_pos = np.array([
                target[0] + end_radius * dir_from_target[0],
                target[1] + end_radius * dir_from_target[1],
                camera_height,
            ])
        else:  # pass_by
            base_radius = rng.uniform(min_dist, max_dist)
            perp_dir = np.array([-dir_from_target[1], dir_from_target[0], 0.0])
            base_pos = np.array([
                target[0] + base_radius * dir_from_target[0],
                target[1] + base_radius * dir_from_target[1],
                camera_height,
            ])
            start_pos = base_pos - (total_dist / 2) * perp_dir
            end_pos = base_pos + (total_dist / 2) * perp_dir

        # Fixed camera orientation
        look_dir = -dir_from_target
        fixed_yaw = np.degrees(np.arctan2(look_dir[1], look_dir[0]))
        mid_pos = (start_pos + end_pos) / 2
        mid_hdist = np.sqrt((target[0] - mid_pos[0])**2 + (target[1] - mid_pos[1])**2)
        dz = target[2] - camera_height
        fixed_pitch = np.degrees(np.arctan2(-dz, mid_hdist))

        look_distance = 10.0
        fixed_look_target = np.array([
            start_pos[0] + look_distance * look_dir[0],
            start_pos[1] + look_distance * look_dir[1],
            target[2],
        ])

        poses: List[CameraPose] = []
        for i in range(num_frames):
            t = i / max(1, num_frames - 1)
            position = start_pos + t * (end_pos - start_pos)
            current_dist = np.linalg.norm(position[:2] - target[:2])

            if not self._sampler.is_position_valid(
                position, ctx["scene_bounds"], ctx["room_polys"], ctx["collision_objects"]
            ):
                return None

            pose = CameraPose(
                position=position.copy(),
                target=fixed_look_target.copy(),
                yaw=float(fixed_yaw),
                pitch=float(fixed_pitch),
                radius=float(current_dist),
                target_objects=[o.label for o in objects],
            )
            if not self._validate_visibility_relaxed(pose, ctx):
                return None
            poses.append(pose)

        return poses

    # ------------------------------------------------------------------
    # Pattern: rotation — in-place rotation at room center
    # ------------------------------------------------------------------

    def _gen_rotation(self, rng: np.random.RandomState, ctx: dict,
                      num_frames: int, scene_path: Path) -> Optional[List[CameraPose]]:
        objects = ctx["objects"]
        collision_objects = ctx["collision_objects"]
        camera_height = self.cam_cfg.rotation_camera_height

        room_centers = self._sampler.compute_room_centers(scene_path)
        if not room_centers:
            return None

        # Pick a random room center
        room_info = room_centers[rng.randint(0, len(room_centers))]
        center_2d = room_info["center"]
        cam_x, cam_y = float(center_2d[0]), float(center_2d[1])
        position = np.array([cam_x, cam_y, camera_height])

        # Check collision
        for obj in collision_objects:
            margin = 0.1
            if (obj.aabb_min[0] - margin <= position[0] <= obj.aabb_max[0] + margin and
                obj.aabb_min[1] - margin <= position[1] <= obj.aabb_max[1] + margin and
                obj.aabb_min[2] - margin <= position[2] <= obj.aabb_max[2] + margin):
                return None

        sweep_deg = self.vid_cfg.rotation_sweep_deg
        # Discrete starting angles: {0, 30, 60, ..., 330}
        start_yaw = float(rng.choice(np.arange(0, 360, 30)))
        direction = rng.choice([-1, 1])
        # Randomly choose 1 or 2 passes
        num_passes = rng.choice([1, 2])
        end_yaw = start_yaw + direction * sweep_deg * num_passes

        yaw_sequence = np.linspace(start_yaw, end_yaw, num_frames)

        poses: List[CameraPose] = []
        for yaw_deg in yaw_sequence:
            yaw_rad = np.radians(yaw_deg)
            look_distance = 1.0
            target_pt = np.array([
                cam_x + look_distance * np.cos(yaw_rad),
                cam_y + look_distance * np.sin(yaw_rad),
                camera_height,
            ])

            pose = CameraPose(
                position=position.copy(),
                target=target_pt,
                yaw=float(yaw_deg % 360),
                pitch=0.0,
                radius=0.0,
                target_objects=[o.label for o in objects],
            )
            poses.append(pose)

        return poses

    # ------------------------------------------------------------------
    # Visibility helpers
    # ------------------------------------------------------------------

    def _validate_visibility_relaxed(self, pose: CameraPose, ctx: dict,
                                     skip_wall_occlusion: bool = False) -> bool:
        """Relaxed visibility check for video frames — object must be somewhere in FOV."""
        K = self._sampler.intrinsics
        width = self.cam_cfg.image_width
        height = self.cam_cfg.image_height
        objects = ctx["objects"]
        all_aabbs = ctx["all_aabbs"]

        for obj in objects:
            in_fov, _ = is_target_in_fov(
                K=K,
                cam_pos=pose.position,
                cam_target=pose.target,
                target_bmin=obj.aabb_min,
                target_bmax=obj.aabb_max,
                width=width,
                height=height,
                require_center=False,
                border=0,
            )
            if not in_fov:
                return False

            if not skip_wall_occlusion:
                # Quick wall occlusion check
                is_occluded, _ = is_wall_occluded(
                    cam_pos=pose.position,
                    target_bmin=obj.aabb_min,
                    target_bmax=obj.aabb_max,
                    occluders=all_aabbs,
                    target_id=str(obj.id),
                    sample_corners=True,
                    occlusion_threshold=0.7,
                )
                if is_occluded:
                    return False

        return True

    def _build_static_fallback_trajectory(
        self,
        scene_path: Path,
        objects: List[SceneObject],
        all_scene_objects: Optional[List[SceneObject]],
        pattern: str,
        num_frames: int,
    ) -> Optional[List[CameraPose]]:
        """Try building a trajectory from one valid static pose."""
        old_pattern = self.cam_cfg.move_pattern
        old_max_tries = self.cam_cfg.max_tries
        try:
            self.cam_cfg.move_pattern = pattern if pattern in {"around", "spherical", "linear"} else "around"
            self.cam_cfg.max_tries = max(200, old_max_tries)
            pose = self._sampler.sample_camera_pose(
                objects=objects,
                scene_path=scene_path,
                all_scene_objects=all_scene_objects,
            )
        finally:
            self.cam_cfg.move_pattern = old_pattern
            self.cam_cfg.max_tries = old_max_tries

        if pose is None:
            return None
        return [self._clone_pose(pose) for _ in range(num_frames)]

    def _build_last_resort_trajectory(self, ctx: dict, num_frames: int) -> List[CameraPose]:
        """Construct a synthetic mini-arc trajectory to guarantee output length."""
        target = ctx["target"]
        objects = ctx["objects"]
        camera_height = self._sampler.compute_camera_height(objects)
        radius = max(self.cam_cfg.min_distance, 1.0)

        poses: List[CameraPose] = []
        for yaw_deg in np.linspace(0.0, 45.0, num_frames):
            yaw_rad = np.radians(yaw_deg)
            position = np.array([
                target[0] + radius * np.cos(yaw_rad),
                target[1] + radius * np.sin(yaw_rad),
                camera_height,
            ])

            dz = target[2] - position[2]
            hdist = np.sqrt((target[0] - position[0]) ** 2 + (target[1] - position[1]) ** 2)
            pitch = np.degrees(np.arctan2(-dz, hdist))

            poses.append(
                CameraPose(
                    position=position,
                    target=target.copy(),
                    yaw=float(yaw_deg % 360),
                    pitch=float(pitch),
                    radius=float(radius),
                    target_objects=[o.label for o in objects],
                )
            )
        return poses

    def _clone_pose(self, pose: CameraPose) -> CameraPose:
        return CameraPose(
            position=np.array(pose.position, dtype=float),
            target=np.array(pose.target, dtype=float),
            yaw=float(pose.yaw),
            pitch=float(pose.pitch),
            radius=float(pose.radius),
            target_objects=list(pose.target_objects) if pose.target_objects else None,
        )

    def _clone_trajectory(self, traj: List[CameraPose]) -> List[CameraPose]:
        return [self._clone_pose(p) for p in traj]
