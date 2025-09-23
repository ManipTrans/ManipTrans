from __future__ import annotations

import os
import random
from enum import Enum
from itertools import cycle
from time import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from bps_torch.bps import bps_torch
from gym import spaces
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import normalize_angle, quat_conjugate, quat_mul
from copy import deepcopy
import math
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
from main.dataset.factory import ManipDataFactory
import time

# from main.dataset.favor_dataset_dexhand import FavorDatasetDexHand
from main.dataset.oakink2_dataset_dexhand_lh import OakInk2DatasetDexHandLH
from main.dataset.oakink2_dataset_dexhand_rh import OakInk2DatasetDexHandRH
from main.dataset.oakink2_dataset_utils import oakink2_obj_scale, oakink2_obj_mass
from main.dataset.transform import aa_to_quat, aa_to_rotmat, quat_to_rotmat, rotmat_to_aa, rotmat_to_quat, rot6d_to_aa
from torch import Tensor
from tqdm import tqdm


from maniptrans_envs.lib.envs.core.config import ROBOT_HEIGHT


def soft_clamp(x, lower, upper):
    return lower + torch.sigmoid(4 / (upper - lower) * (x - (lower + upper) / 2)) * (upper - lower)


class DexManipBiH:

    def __init__(
        self,
        args,
        rollout_seq,
    ):
    
        self.gym = gymapi.acquire_gym()
        self.sim_params = gymapi.SimParams()

        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        self.headless = args.headless
        if self.headless:
            self.graphics_device_id = -1

        self.rollout_seq = rollout_seq

        self.sim_params.substeps = 2
        self.sim_params.dt = 1.0 / 60.0
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 20
        self.sim_params.physx.num_velocity_iterations = 1
        self.sim_params.physx.num_threads = 4
        self.sim_params.physx.use_gpu = args.use_gpu
        self.sim_params.use_gpu_pipeline = args.use_gpu_pipeline
        self.sim_device = args.sim_device if args.use_gpu_pipeline else "cpu"

        self.device = torch.device("cuda:0") if args.use_gpu else torch.device("cpu")

        self.num_envs = 1  # ? only for visualization

        self.aggregate_mode = 3

        self.dexhand_rh = DexHandFactory.create_hand(rollout_seq["dexhand"], "right")
        self.dexhand_lh = DexHandFactory.create_hand(rollout_seq["dexhand"], "left")

        # Values to be filled in at runtime
        self.rh_states = {}
        self.lh_states = {}
        self.dexhand_rh_handles = {}  # will be dict mapping names to relevant sim handles
        self.dexhand_lh_handles = {}  # will be dict mapping names to relevant sim handles
        self.objs_handles = {}  # for obj handlers
        self.objs_assets = {}
        self.num_dofs = None  # Total number of DOFs per env
        self.actions = None  # Current actions to be deployed

        # Tensor placeholders
        self._root_state = None  # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None  # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self.net_cf = None  # contact force
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._dexhand_rh_effort_limits = None  # Actuator effort limits for dexhand_r
        self._dexhand_rh_dof_speed_limits = None  # Actuator speed limits for dexhand_r
        self._global_dexhand_rh_indices = None  # Unique indices corresponding to all envs in flattened array

        self.sim_device = torch.device("cuda:0") if args.use_gpu else torch.device("cpu")

        self.sim_initialized = False
        self.sim = self.gym.create_sim(
            args.compute_device_id, args.graphics_device_id, args.physics_engine, self.sim_params
        )
        self._create_ground_plane()
        self._create_envs()

        self.gym.prepare_sim(self.sim)
        self.sim_initialized = True
        self.set_viewer()

        self._refresh()

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def set_viewer(self):
        """Create the viewer."""
        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            self.enable_viewer_sync = True
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "record_frames")

            # set the camera position based on up axis
            num_per_row = int(np.sqrt(self.num_envs))
            cam_pos = gymapi.Vec3(num_per_row + 1.0, num_per_row + 1.0, 3.0)
            cam_target = gymapi.Vec3(num_per_row - 6.0, num_per_row - 6.0, 1.0)

            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _create_envs(self):
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # * >>> import table asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True

        table_width_offset = 0.2
        table_asset = self.gym.create_box(self.sim, 0.8 + table_width_offset, 1.6, 0.03, table_asset_options)

        table_pos = gymapi.Vec3(-table_width_offset / 2, 0, 0.4)
        self.dexhand_rh_pose = gymapi.Transform()
        table_half_height = 0.015
        table_half_width = 0.4
        self._table_surface_z = table_surface_z = table_pos.z + table_half_height
        self.dexhand_rh_pose.p = gymapi.Vec3(-table_half_width, 0, table_surface_z + ROBOT_HEIGHT)
        self.dexhand_rh_pose.r = gymapi.Quat.from_euler_zyx(0, -np.pi / 2, 0)
        self.dexhand_lh_pose = deepcopy(self.dexhand_rh_pose)

        mujoco2gym_transf = np.eye(4)
        mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(
            np.array([np.pi / 2, 0, 0])
        )
        mujoco2gym_transf[:3, 3] = np.array([0, 0, self._table_surface_z])
        self.mujoco2gym_transf = torch.tensor(mujoco2gym_transf, device=self.sim_device, dtype=torch.float32)

        dexhand_rh_asset_file = self.dexhand_rh.urdf_path
        dexhand_lh_asset_file = self.dexhand_lh.urdf_path
        asset_options = gymapi.AssetOptions()
        asset_options.thickness = 0.001
        asset_options.angular_damping = 20
        asset_options.linear_damping = 20
        asset_options.max_linear_velocity = 50
        asset_options.max_angular_velocity = 100
        asset_options.fix_base_link = False
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        dexhand_rh_asset = self.gym.load_asset(self.sim, *os.path.split(dexhand_rh_asset_file), asset_options)
        dexhand_lh_asset = self.gym.load_asset(self.sim, *os.path.split(dexhand_lh_asset_file), asset_options)
        dexhand_rh_dof_stiffness = torch.tensor(
            [500] * self.dexhand_rh.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        dexhand_rh_dof_damping = torch.tensor(
            [30] * self.dexhand_rh.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        dexhand_lh_dof_stiffness = torch.tensor(
            [500] * self.dexhand_lh.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        dexhand_lh_dof_damping = torch.tensor(
            [30] * self.dexhand_lh.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        self.limit_info = {}
        asset_rh_dof_props = self.gym.get_asset_dof_properties(dexhand_rh_asset)
        asset_lh_dof_props = self.gym.get_asset_dof_properties(dexhand_lh_asset)
        self.limit_info["rh"] = {
            "lower": np.asarray(asset_rh_dof_props["lower"]).copy().astype(np.float32),
            "upper": np.asarray(asset_rh_dof_props["upper"]).copy().astype(np.float32),
        }
        self.limit_info["lh"] = {
            "lower": np.asarray(asset_lh_dof_props["lower"]).copy().astype(np.float32),
            "upper": np.asarray(asset_lh_dof_props["upper"]).copy().astype(np.float32),
        }

        rigid_shape_rh_props_asset = self.gym.get_asset_rigid_shape_properties(dexhand_rh_asset)
        self.gym.set_asset_rigid_shape_properties(dexhand_rh_asset, rigid_shape_rh_props_asset)

        rigid_shape_lh_props_asset = self.gym.get_asset_rigid_shape_properties(dexhand_lh_asset)
        self.gym.set_asset_rigid_shape_properties(dexhand_lh_asset, rigid_shape_lh_props_asset)

        self.num_dexhand_rh_bodies = self.gym.get_asset_rigid_body_count(dexhand_rh_asset)
        self.num_dexhand_rh_dofs = self.gym.get_asset_dof_count(dexhand_rh_asset)
        self.num_dexhand_lh_bodies = self.gym.get_asset_rigid_body_count(dexhand_lh_asset)
        self.num_dexhand_lh_dofs = self.gym.get_asset_dof_count(dexhand_lh_asset)

        dexhand_rh_dof_props = self.gym.get_asset_dof_properties(dexhand_rh_asset)
        self.dexhand_rh_dof_lower_limits = []
        self.dexhand_rh_dof_upper_limits = []
        self._dexhand_rh_effort_limits = []
        self._dexhand_rh_dof_speed_limits = []
        for i in range(self.num_dexhand_rh_dofs):
            dexhand_rh_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dexhand_rh_dof_props["stiffness"][i] = dexhand_rh_dof_stiffness[i]
            dexhand_rh_dof_props["damping"][i] = dexhand_rh_dof_damping[i]

            self.dexhand_rh_dof_lower_limits.append(dexhand_rh_dof_props["lower"][i])
            self.dexhand_rh_dof_upper_limits.append(dexhand_rh_dof_props["upper"][i])
            self._dexhand_rh_effort_limits.append(dexhand_rh_dof_props["effort"][i])
            self._dexhand_rh_dof_speed_limits.append(dexhand_rh_dof_props["velocity"][i])

        self.dexhand_rh_dof_lower_limits = torch.tensor(self.dexhand_rh_dof_lower_limits, device=self.sim_device)
        self.dexhand_rh_dof_upper_limits = torch.tensor(self.dexhand_rh_dof_upper_limits, device=self.sim_device)
        self._dexhand_rh_effort_limits = torch.tensor(self._dexhand_rh_effort_limits, device=self.sim_device)
        self._dexhand_rh_dof_speed_limits = torch.tensor(self._dexhand_rh_dof_speed_limits, device=self.sim_device)

        # set dexhand_l dof properties
        dexhand_lh_dof_props = self.gym.get_asset_dof_properties(dexhand_lh_asset)
        self.dexhand_lh_dof_lower_limits = []
        self.dexhand_lh_dof_upper_limits = []
        self._dexhand_lh_effort_limits = []
        self._dexhand_lh_dof_speed_limits = []
        for i in range(self.num_dexhand_lh_dofs):
            dexhand_lh_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dexhand_lh_dof_props["stiffness"][i] = dexhand_lh_dof_stiffness[i]
            dexhand_lh_dof_props["damping"][i] = dexhand_lh_dof_damping[i]

            self.dexhand_lh_dof_lower_limits.append(dexhand_lh_dof_props["lower"][i])
            self.dexhand_lh_dof_upper_limits.append(dexhand_lh_dof_props["upper"][i])
            self._dexhand_lh_effort_limits.append(dexhand_lh_dof_props["effort"][i])
            self._dexhand_lh_dof_speed_limits.append(dexhand_lh_dof_props["velocity"][i])

        self.dexhand_lh_dof_lower_limits = torch.tensor(self.dexhand_lh_dof_lower_limits, device=self.sim_device)
        self.dexhand_lh_dof_upper_limits = torch.tensor(self.dexhand_lh_dof_upper_limits, device=self.sim_device)
        self._dexhand_lh_effort_limits = torch.tensor(self._dexhand_lh_effort_limits, device=self.sim_device)
        self._dexhand_lh_dof_speed_limits = torch.tensor(self._dexhand_lh_dof_speed_limits, device=self.sim_device)

        # compute aggregate size
        num_dexhand_rh_bodies = self.gym.get_asset_rigid_body_count(dexhand_rh_asset)
        num_dexhand_rh_shapes = self.gym.get_asset_rigid_shape_count(dexhand_rh_asset)
        num_dexhand_lh_bodies = self.gym.get_asset_rigid_body_count(dexhand_lh_asset)
        num_dexhand_lh_shapes = self.gym.get_asset_rigid_shape_count(dexhand_lh_asset)

        self.dexhand_rs = []
        self.dexhand_ls = []
        self.envs = []

        # Create environments
        self.manip_obj_rh_mass = []
        self.manip_obj_rh_com = []
        self.manip_obj_lh_mass = []
        self.manip_obj_lh_com = []
        num_per_row = int(np.sqrt(self.num_envs))
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            rh_current_asset, rh_sum_rigid_body_count, rh_sum_rigid_shape_count, rh_obj_scale, rh_obj_mass = (
                self._create_obj_assets(i, side="rh")
            )
            lh_current_asset, lh_sum_rigid_body_count, lh_sum_rigid_shape_count, lh_obj_scale, lh_obj_mass = (
                self._create_obj_assets(i, side="lh")
            )

            max_agg_bodies = (
                num_dexhand_rh_bodies
                + num_dexhand_lh_bodies
                + 1
                + rh_sum_rigid_body_count
                + lh_sum_rigid_body_count
                + (0 + (0 + self.dexhand_lh.n_bodies * 2 if not self.headless else 0))
            )  # 1 for table
            max_agg_shapes = (
                num_dexhand_rh_shapes
                + num_dexhand_lh_shapes
                + 1
                + rh_sum_rigid_shape_count
                + lh_sum_rigid_shape_count
                + (0 + (0 + self.dexhand_lh.n_bodies * 2 if not self.headless else 0))
            )
            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: dexhand_r should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create dexhand_r
            dexhand_rh_actor = self.gym.create_actor(
                env_ptr,
                dexhand_rh_asset,
                self.dexhand_rh_pose,
                "dexhand_r",
                i,
                1,  # * only for visualization
            )
            dexhand_lh_actor = self.gym.create_actor(
                env_ptr,
                dexhand_lh_asset,
                self.dexhand_lh_pose,
                "dexhand_l",
                i,
                1,  # * only for visualization
            )
            self.gym.enable_actor_dof_force_sensors(env_ptr, dexhand_rh_actor)
            self.gym.enable_actor_dof_force_sensors(env_ptr, dexhand_lh_actor)
            self.gym.set_actor_dof_properties(env_ptr, dexhand_rh_actor, dexhand_rh_dof_props)
            self.gym.set_actor_dof_properties(env_ptr, dexhand_lh_actor, dexhand_lh_dof_props)

            # Create table and obstacles
            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(table_pos.x, table_pos.y, table_pos.z)
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 1)
            table_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_actor)
            table_props[0].friction = 0.1  # ? only one table shape in each env
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_actor, table_props)
            # set table's color to be dark gray
            self.gym.set_rigid_body_color(env_ptr, table_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.1, 0.1))

            self.obj_rh_handle, _ = self._create_obj_actor(
                env_ptr, i, rh_current_asset, side="rh"
            )  # the handle is all the same for all envs
            self.obj_lh_handle, _ = self._create_obj_actor(env_ptr, i, lh_current_asset, side="lh")
            self.gym.set_actor_scale(env_ptr, self.obj_rh_handle, rh_obj_scale)
            self.gym.set_actor_scale(env_ptr, self.obj_lh_handle, lh_obj_scale)
            obj_props_rh = self.gym.get_actor_rigid_body_properties(env_ptr, self.obj_rh_handle)
            obj_props_lh = self.gym.get_actor_rigid_body_properties(env_ptr, self.obj_lh_handle)
            obj_props_rh[0].mass = min(0.5, obj_props_rh[0].mass)  # * we only consider the mass less than 500g
            obj_props_lh[0].mass = min(0.5, obj_props_lh[0].mass)  # * we only consider the mass less than 500g

            if rh_obj_mass is not None:
                obj_props_rh[0].mass = rh_obj_mass
            if lh_obj_mass is not None:
                obj_props_lh[0].mass = lh_obj_mass

            # ! Updating the mass and scale might slightly alter the inertia tensor;
            # ! however, because the magnitude of our modifications is minimal, we temporarily neglect this effect.
            self.gym.set_actor_rigid_body_properties(env_ptr, self.obj_rh_handle, obj_props_rh)
            self.gym.set_actor_rigid_body_properties(env_ptr, self.obj_lh_handle, obj_props_lh)
            self.manip_obj_rh_mass.append(obj_props_rh[0].mass)
            self.manip_obj_rh_com.append(
                torch.tensor([obj_props_rh[0].com.x, obj_props_rh[0].com.y, obj_props_rh[0].com.z])
            )
            self.manip_obj_lh_mass.append(obj_props_lh[0].mass)
            self.manip_obj_lh_com.append(
                torch.tensor([obj_props_lh[0].com.x, obj_props_lh[0].com.y, obj_props_lh[0].com.z])
            )

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.dexhand_rs.append(dexhand_rh_actor)
            self.dexhand_ls.append(dexhand_lh_actor)

        self.manip_obj_rh_mass = torch.tensor(self.manip_obj_rh_mass, device=self.device)
        self.manip_obj_rh_com = torch.stack(self.manip_obj_rh_com, dim=0).to(self.device)
        self.manip_obj_lh_mass = torch.tensor(self.manip_obj_lh_mass, device=self.device)
        self.manip_obj_lh_com = torch.stack(self.manip_obj_lh_com, dim=0).to(self.device)

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        dexhand_rh_handle = self.gym.find_actor_handle(env_ptr, "dexhand_r")
        dexhand_lh_handle = self.gym.find_actor_handle(env_ptr, "dexhand_l")
        self.dexhand_rh_handles = {
            k: self.gym.find_actor_rigid_body_handle(env_ptr, dexhand_rh_handle, k) for k in self.dexhand_rh.body_names
        }
        self.dexhand_lh_handles = {
            k: self.gym.find_actor_rigid_body_handle(env_ptr, dexhand_lh_handle, k) for k in self.dexhand_lh.body_names
        }
        self.dexhand_rh_cf_weights = {
            k: (1.0 if ("intermediate" in k or "distal" in k) else 0.0) for k in self.dexhand_rh.body_names
        }
        self.dexhand_lh_cf_weights = {
            k: (1.0 if ("intermediate" in k or "distal" in k) else 0.0) for k in self.dexhand_lh.body_names
        }
        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        _dof_force = self.gym.acquire_dof_force_tensor(self.sim)

        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._rh_base_state = self._root_state[:, 0, :]
        self._lh_base_state = self._root_state[:, 1, :]

        # ? >>> for visualization
        if not self.headless:

            self.mano_joint_rh_points = [
                self._root_state[:, self.gym.find_actor_handle(env_ptr, f"rh_mano_joint_{i}"), :]
                for i in range(self.dexhand_rh.n_bodies)
            ]
            self.mano_joint_lh_points = [
                self._root_state[:, self.gym.find_actor_handle(env_ptr, f"lh_mano_joint_{i}"), :]
                for i in range(self.dexhand_lh.n_bodies)
            ]
        # ? <<<

        self._manip_obj_rh_handle = self.gym.find_actor_handle(env_ptr, "manip_obj_rh")
        self._manip_obj_rh_root_state = self._root_state[:, self._manip_obj_rh_handle, :]
        self._manip_obj_lh_handle = self.gym.find_actor_handle(env_ptr, "manip_obj_lh")
        self._manip_obj_lh_root_state = self._root_state[:, self._manip_obj_lh_handle, :]
        self.net_cf = gymtorch.wrap_tensor(_net_cf).view(self.num_envs, -1, 3)
        self.dof_force = gymtorch.wrap_tensor(_dof_force).view(self.num_envs, -1)
        self._manip_obj_rh_rigid_body_handle = self.gym.find_actor_rigid_body_handle(
            env_ptr, self._manip_obj_rh_handle, "base"
        )
        self._manip_obj_lh_rigid_body_handle = self.gym.find_actor_rigid_body_handle(
            env_ptr, self._manip_obj_lh_handle, "base"
        )
        self._manip_obj_rh_cf = self.net_cf[:, self._manip_obj_rh_rigid_body_handle, :]
        self._manip_obj_lh_cf = self.net_cf[:, self._manip_obj_lh_rigid_body_handle, :]

        self.dexhand_rh_root_state = self._root_state[:, dexhand_rh_handle, :]
        self.dexhand_lh_root_state = self._root_state[:, dexhand_lh_handle, :]

        self.apply_forces = torch.zeros(
            (self.num_envs, self._rigid_body_state.shape[1], 3), device=self.device, dtype=torch.float
        )
        self.apply_torque = torch.zeros(
            (self.num_envs, self._rigid_body_state.shape[1], 3), device=self.device, dtype=torch.float
        )
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.curr_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize indices
        self._global_dexhand_rh_indices = torch.tensor(
            [self.gym.find_actor_index(env, "dexhand_r", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)
        self._global_dexhand_lh_indices = torch.tensor(
            [self.gym.find_actor_index(env, "dexhand_l", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)

        self._global_manip_obj_rh_indices = torch.tensor(
            [self.gym.find_actor_index(env, "manip_obj_rh", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)
        self._global_manip_obj_lh_indices = torch.tensor(
            [self.gym.find_actor_index(env, "manip_obj_lh", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)

    def _create_obj_assets(self, i, side="rh"):
        if side == "rh":
            obj_id = self.rollout_seq["oid_rh"]
        else:
            obj_id = self.rollout_seq["oid_lh"]

        asset_options = gymapi.AssetOptions()
        asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options.convex_decomposition_from_submeshes = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.thickness = 0.001
        asset_options.max_linear_velocity = 50
        asset_options.max_angular_velocity = 100
        asset_options.fix_base_link = False
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 200000
        asset_options.density = 200  # * the average density of low-fill-rate 3D-printed models
        if side == "rh":
            obj_urdf_path = self.rollout_seq["obj_rh_path"]
        else:
            obj_urdf_path = self.rollout_seq["obj_lh_path"]
        current_asset = self.gym.load_asset(self.sim, *os.path.split(obj_urdf_path), asset_options)

        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(current_asset)
        self.gym.set_asset_rigid_shape_properties(current_asset, rigid_shape_props_asset)

        # * load assigned scale and mass for the object if available
        if obj_id in oakink2_obj_scale:
            scale = oakink2_obj_scale[obj_id]
        else:
            scale = 1.0

        if obj_id in oakink2_obj_mass:
            mass = oakink2_obj_mass[obj_id]
        else:
            mass = None

        sum_rigid_body_count = self.gym.get_asset_rigid_body_count(current_asset)
        sum_rigid_shape_count = self.gym.get_asset_rigid_shape_count(current_asset)
        return current_asset, sum_rigid_body_count, sum_rigid_shape_count, scale, mass

    def _create_obj_actor(self, env_ptr, i, current_asset, side="rh"):

        if side == "rh":
            obj_transf = self.rollout_seq["state_manip_obj_rh"]
        else:
            obj_transf = self.rollout_seq["state_manip_obj_lh"]

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(obj_transf[0, 3], obj_transf[1, 3], obj_transf[2, 3])
        obj_aa = rotmat_to_aa(obj_transf[:3, :3])
        obj_aa_angle = torch.norm(obj_aa)
        obj_aa_axis = obj_aa / obj_aa_angle
        pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(obj_aa_axis[0], obj_aa_axis[1], obj_aa_axis[2]), obj_aa_angle)

        if side == "rh":
            obj_actor = self.gym.create_actor(env_ptr, current_asset, pose, "manip_obj_rh", i, 1)
        else:
            obj_actor = self.gym.create_actor(env_ptr, current_asset, pose, "manip_obj_lh", i, 1)
        obj_index = self.gym.get_actor_index(env_ptr, obj_actor, gymapi.DOMAIN_SIM)

        return obj_actor, obj_index

    def _refresh(self):

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def play(self):
        iter = 0

        while True:
            if iter >= len(self.rollout_seq["dq_rh"]):
                iter = 0
            self._root_state[:, 0] = torch.tensor(self.rollout_seq["state_rh"][iter][None], device=self.sim_device)
            self._root_state[:, 1] = torch.tensor(self.rollout_seq["state_lh"][iter][None], device=self.sim_device)
            self._manip_obj_rh_root_state[:] = torch.tensor(
                self.rollout_seq["state_manip_obj_rh"][iter][None], device=self.sim_device
            )
            self._manip_obj_lh_root_state[:] = torch.tensor(
                self.rollout_seq["state_manip_obj_lh"][iter][None], device=self.sim_device
            )
            q_state_rh = torch.stack(
                [
                    self.rollout_seq["q_rh"][[iter]],
                    self.rollout_seq["dq_rh"][[iter]],
                ],
                dim=-1,
            ).to(self.sim_device)
            q_state_lh = torch.stack(
                [
                    self.rollout_seq["q_lh"][[iter]],
                    self.rollout_seq["dq_lh"][[iter]],
                ],
                dim=-1,
            ).to(self.sim_device)
            self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(torch.cat([q_state_rh, q_state_lh], dim=1)))
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_state))

            # Step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            if not self.headless:
                self.gym.step_graphics(self.sim)

            # Update jacobian and mass matrix
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            # Step rendering
            if not self.headless:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

            time.sleep(self.sim_params.dt)

            iter += 1
