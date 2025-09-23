import os
import pickle
from functools import lru_cache

import numpy as np
import smplx
import torch
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from smplx.lbs import batch_rigid_transform, batch_rodrigues
from torch.utils.data import Dataset

from main.dataset.transform import aa_to_rotmat, caculate_align_mat, rotmat_to_aa, quat_to_aa
from scipy.ndimage import gaussian_filter1d
from termcolor import cprint
from manotorch.manolayer import ManoLayer, MANOOutput
import trimesh
from .base import ManipData
from .decorators import register_manipdata

@register_manipdata("gigahands_rh")
class GigaHandsDexHandRH(ManipData):
    def __init__(
        self,
        *,
        data_dir: str = "data/gigadual_demo",
        split: str = "all",
        skip: int = 1,
        device="cuda:0",
        mujoco2gym_transf=None,
        max_seq_len=int(1e10),
        dexhand=None,
        **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            split=split,
            skip=skip,
            device=device,
            mujoco2gym_transf=mujoco2gym_transf,
            max_seq_len=max_seq_len,
            dexhand=dexhand,
            **kwargs,
        )
        self.manolayer = ManoLayer(
            rot_mode="axisang",
            side="left",
            center_idx=None,
            mano_assets_root="data/mano_v1_2",
            use_pca=False,
            flat_hand_mean=True,
        ).to(device)

        pathes = os.listdir(data_dir)
        self.data_pathes = [os.path.join(self.data_dir, p) for p in pathes]

        self.device = device

        transf_offset = np.eye(4)
        transf_offset[:3, :3] = aa_to_rotmat(np.array([0, -np.pi / 2, 0]))
        transf_offset[:3, 3] = np.array([0.00, -0.34, -0.20])

        self.transf_offset = torch.tensor(transf_offset, dtype=torch.float32, device=mujoco2gym_transf.device)

        self.mujoco2gym_transf = mujoco2gym_transf @ self.transf_offset
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data_pathes)

    @lru_cache(maxsize=None)
    def __getitem__(self, idx):
        # TODO: adapt to gigahands
        idx = int(idx[2:])
        assert self.mujoco2gym_transf is not None

        data = np.load(os.path.join(self.data_pathes[idx], "001_sv_dict.npy"), allow_pickle=True).item()
        # TODO: adapt to gigahands
        obj_mesh = trimesh.load(os.path.join(self.data_pathes[idx], "urdf", "scan.obj"), process=False)

        length = len(data["rhand_global_orient_gt"])
        obj_pose = np.eye(4)[None].repeat(length, axis=0)
        obj_pose[:, :3, :3] = data["object_global_orient"].transpose(0, 2, 1)  # ?
        obj_pose[:, :3, 3] = data["object_transl"]
        obj_pose = torch.tensor(obj_pose, device=self.device)
        hand_rot = torch.tensor(data["rhand_global_orient_gt"], device=self.device)
        hand_tsl = torch.tensor(data["rhand_transl"], device=self.device)
        mano_out_verts = torch.tensor(data["rhand_verts"], device=self.device)
        mano_out_joints = torch.matmul(self.manolayer.th_J_regressor, mano_out_verts)

        wrist_pos = mano_out_joints.detach()[:, 0]
        middle_pos = mano_out_joints.detach()[:, 4]

        wrist_pos = wrist_pos - (middle_pos - wrist_pos) * 0.25  # TODO hack for wrist position

        mano_joints = {
            "index_proximal": mano_out_joints.detach()[:, 1],
            "index_intermediate": mano_out_joints.detach()[:, 2],
            "index_distal": mano_out_joints.detach()[:, 3],
            "index_tip": mano_out_verts[:, 353].detach(),  # reselect tip
            "middle_proximal": mano_out_joints.detach()[:, 4],
            "middle_intermediate": mano_out_joints.detach()[:, 5],
            "middle_distal": mano_out_joints.detach()[:, 6],
            "middle_tip": mano_out_verts[:, 467].detach(),  # reselect tip
            "pinky_proximal": mano_out_joints.detach()[:, 7],
            "pinky_intermediate": mano_out_joints.detach()[:, 8],
            "pinky_distal": mano_out_joints.detach()[:, 9],
            "pinky_tip": mano_out_verts[:, 695].detach(),  # reselect tip
            "ring_proximal": mano_out_joints.detach()[:, 10],
            "ring_intermediate": mano_out_joints.detach()[:, 11],
            "ring_distal": mano_out_joints.detach()[:, 12],
            "ring_tip": mano_out_verts[:, 576].detach(),  # reselect tip
            "thumb_proximal": mano_out_joints.detach()[:, 13],
            "thumb_intermediate": mano_out_joints.detach()[:, 14],
            "thumb_distal": mano_out_joints.detach()[:, 15],
            "thumb_tip": mano_out_verts[:, 766].detach(),  # reselect tip
        }

        inspire_rot_offset = self.dexhand.relative_rotation
        wrist_rot = aa_to_rotmat(hand_rot).float() @ torch.tensor(
            np.repeat(inspire_rot_offset[None], length, axis=0), device=self.device
        ).float()

        mesh = Meshes(
            verts=torch.from_numpy(obj_mesh.vertices[None, ...]).float(),
            faces=torch.from_numpy(obj_mesh.faces[None, ...]).float(),
        )
        rs_verts_obj = self.random_sampling_pc(mesh)

        data = {
            "data_path": os.path.join(self.data_pathes[idx], "001_sv_dict.npy"),
            "obj_id": "-1",  # ? placeholder
            "obj_verts": rs_verts_obj,
            "obj_urdf_path": os.path.join(self.data_pathes[idx], "urdf", "scan.urdf"), # TODO: adapt to gigahands
            "obj_trajectory": torch.tensor(
                np.stack(obj_pose[:: self.skip].cpu()), device=self.device, dtype=torch.float
            ),
            "scene_objs": [],  # ? placeholder
            "wrist_pos": wrist_pos,
            "wrist_rot": wrist_rot,
            "mano_joints": mano_joints,
        }

        self.process_data(data, idx, rs_verts_obj)
        opt_path = f"data/retargeting/gigadual_demo/mano2{str(self.dexhand)}/001_sv_dict.pkl"

        self.load_retargeted_data(data, opt_path)

        return data


if __name__ == "__main__":
    fdata = GigaHandsDexHandRH(mujoco2gym_transf=torch.eye(4))
    print(fdata[0])
