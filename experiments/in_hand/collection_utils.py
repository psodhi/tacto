import logging

import os
import numpy as np
import pandas as pd

import imageio
from attrdict import AttrDict

import pybullet as p

log = logging.getLogger(__name__)
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def get_wps_posn_back_forth(center_pos, end_pos, nsteps, noise_scale=0):

    pos_step = (end_pos - center_pos) / (nsteps / 4.0)

    waypoints = []
    waypoints.append(center_pos)
    for step in range(0, nsteps):
        step_frac = step / nsteps

        noise = noise_scale * (np.random.rand() - 0.5)
        pos_step = pos_step + noise

        if (step_frac <= 0.25):
            pos = waypoints[-1] + pos_step
        elif (step_frac > 0.25) & (step_frac <= 0.75):
            pos = waypoints[-1] - pos_step
        else:
            pos = waypoints[-1] + pos_step

        waypoints.append(pos)

    return waypoints


def get_wps_pose(start_pose, end_pose, nsteps, noise_scale=0):

    t_range = np.linspace(0., 1., nsteps)
    start_xyz, start_rpy = start_pose
    end_xyz, end_rpy = end_pose

    waypoints = []
    for t in t_range:
        curr_pos = (1 - t) * start_xyz + t * end_xyz
        curr_rpy = (1 - t) * start_rpy + t * end_rpy
        curr_ori = p.getQuaternionFromEuler(curr_rpy)

        pose = (curr_pos, curr_ori)
        waypoints.append(pose)

    return waypoints


def set_gui_params(cam_params=None):

    enable_preview = False
    p.configureDebugVisualizer(
        p.COV_ENABLE_RGB_BUFFER_PREVIEW, enable_preview)
    p.configureDebugVisualizer(
        p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, enable_preview)
    p.configureDebugVisualizer(
        p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, enable_preview)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, enable_preview)

    if cam_params is not None:
        cam_tgt_pos = cam_params['cam_tgt_pos']
        cam_dist = cam_params['cam_dist']
        cam_yaw = cam_params['cam_yaw']
        cam_pitch = cam_params['cam_pitch']
        p.resetDebugVisualizerCamera(cam_dist, cam_yaw, cam_pitch, cam_tgt_pos)


class WaypointSetter():
    def __init__(self, robot, max_force=100, slider_params={}):
        super().__init__()
        self.robot = robot
        self.max_force = max_force

        pos, ori = self.robot.get_base_pose()

        self.cid = p.createConstraint(
            self.robot.id,
            -1,
            -1,
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            childFramePosition=pos,
            childFrameOrientation=ori,
        )

    def update(self, pos=None, ori=None):

        pos = pos if pos is not None else self.robot.get_base_pose()[0]
        ori = ori if ori is not None else self.robot.get_base_pose()[1]

        p.changeConstraint(self.cid, pos, ori, maxForce=self.max_force)


class DataLogger:
    def __init__(self, cfg):
        self.cfg = cfg

        self.dataset_dstdir = self.cfg.dataset.dstdir
        self.dataset_name = self.cfg.dataset.name

        self.data_list = []
        self.data_csvname = "poses_imgs"

    def start_new_episode(self, obj, step_idx, eps_idx):
        obj.reset(rand_mag_pos=0.1)
        step_idx = 0
        eps_idx = eps_idx + 1

        os.makedirs(f"{BASE_PATH}/{self.dataset_dstdir}/{self.dataset_name}/{eps_idx:04d}/top/color/", exist_ok=True)
        os.makedirs(f"{BASE_PATH}/{self.dataset_dstdir}/{self.dataset_name}/{eps_idx:04d}/top/depth/", exist_ok=True)
        os.makedirs(f"{BASE_PATH}/{self.dataset_dstdir}/{self.dataset_name}/{eps_idx:04d}/top/normal/", exist_ok=True)

        os.makedirs(f"{BASE_PATH}/{self.dataset_dstdir}/{self.dataset_name}/{eps_idx:04d}/bot/color/", exist_ok=True)
        os.makedirs(f"{BASE_PATH}/{self.dataset_dstdir}/{self.dataset_name}/{eps_idx:04d}/bot/depth/", exist_ok=True)
        os.makedirs(f"{BASE_PATH}/{self.dataset_dstdir}/{self.dataset_name}/{eps_idx:04d}/bot/normal/", exist_ok=True)

        return obj, step_idx, eps_idx

    def get_pose_from_renderer(self, renderer):

        obj_pose = None

        obj_names = list(renderer.object_nodes.keys())
        for obj_name in obj_names:
            node = renderer.object_nodes[obj_name]
            obj_pose = renderer.scene.get_pose(node)

        obj_pos = list(obj_pose[0:3, -1].reshape(-1))
        obj_ori = list(obj_pose[0:3, 0:3].reshape(-1))

        return obj_pos, obj_ori

    def save_episode_step(self, eps_idx, step_idx, imgs_top, imgs_bot, obj, digit_top, digit_bottom, renderer=None):

        # save digit top img frames
        img_top_color_loc = f"{eps_idx:04d}/top/color/{step_idx:04d}.png"
        img_top_depth_loc = f"{eps_idx:04d}/top/depth/{step_idx:04d}.tiff"
        img_top_normal_loc = f"{eps_idx:04d}/top/normal/{step_idx:04d}.png"

        imageio.imwrite(f"{BASE_PATH}/{self.dataset_dstdir}/{self.dataset_name}/{img_top_color_loc}", imgs_top[0])
        imageio.imwrite(f"{BASE_PATH}/{self.dataset_dstdir}/{self.dataset_name}/{img_top_depth_loc}", imgs_top[1])
        imageio.imwrite(f"{BASE_PATH}/{self.dataset_dstdir}/{self.dataset_name}/{img_top_normal_loc}", imgs_top[2])

        # save digit bottom img frames
        img_bot_color_loc = f"{eps_idx:04d}/bot/color/{step_idx:04d}.png"
        img_bot_depth_loc = f"{eps_idx:04d}/bot/depth/{step_idx:04d}.tiff"
        img_bot_normal_loc = f"{eps_idx:04d}/bot/normal/{step_idx:04d}.png"

        imageio.imwrite(f"{BASE_PATH}/{self.dataset_dstdir}/{self.dataset_name}/{img_bot_color_loc}", imgs_bot[0])
        imageio.imwrite(f"{BASE_PATH}/{self.dataset_dstdir}/{self.dataset_name}/{img_bot_depth_loc}", imgs_bot[1])
        imageio.imwrite(f"{BASE_PATH}/{self.dataset_dstdir}/{self.dataset_name}/{img_bot_normal_loc}", imgs_bot[2])

        # object, digit poses
        # obj_pos, obj_ori = obj.get_base_pose()[0], p.getMatrixFromQuaternion(obj.get_base_pose()[1])
        obj_pos, obj_ori = self.get_pose_from_renderer(renderer)
        digit_top_pos, digit_top_ori = digit_top.get_base_pose(
        )[0], p.getMatrixFromQuaternion(digit_top.get_base_pose()[1])
        digit_bot_pos, digit_bot_ori = digit_bottom.get_base_pose(
        )[0], p.getMatrixFromQuaternion(digit_bottom.get_base_pose()[1])

        data_row = {'obj_pos': list(obj_pos),
                    'obj_ori': list(obj_ori),
                    'digit_top_pos': list(digit_top_pos),
                    'digit_top_ori': list(digit_top_ori),
                    'digit_bot_pos': list(digit_bot_pos),
                    'digit_bot_ori': list(digit_bot_ori),
                    'img_top_color_loc': img_top_color_loc,
                    'img_top_depth_loc': img_top_depth_loc,
                    'img_top_normal_loc': img_top_normal_loc,
                    'img_bot_color_loc': img_bot_color_loc,
                    'img_bot_depth_loc': img_bot_depth_loc,
                    'img_bot_normal_loc': img_bot_normal_loc,
                    }

        self.data_list.append(data_row)

    def save_episode_dataset(self, eps_idx):
        csvfile = f"{BASE_PATH}/{self.dataset_dstdir}/{self.dataset_name}/{eps_idx:04d}/{self.data_csvname}.csv"

        self.data_frame = pd.DataFrame(self.data_list)
        self.data_frame.to_csv(csvfile)

        log.info(f"Saving episode {eps_idx} to {csvfile}")
        self.data_list = []
