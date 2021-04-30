import logging

import os
import numpy as np
import csv
import pandas as pd

import imageio
import cv2
from skimage import img_as_ubyte

import cv2
import hydra
import pybullet as p
import tacto  # Import TACTO

import pybulletX as px

log = logging.getLogger(__name__)

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

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
        p.resetDebugVisualizerCamera(
            cam_dist, cam_yaw, cam_pitch, cam_tgt_pos)


def save_dataset_frame(cfg, eps_idx, step_idx, imgs_top, imgs_bot, obj, digit_top, digit_bottom):

    # save digit top img frames
    # img_top_color_loc = f"{cfg.dataset.dstdir}/{cfg.dataset.type}/{eps_idx:04d}/top/color/{step_idx:04d}.png"
    img_top_color_loc = f"{cfg.dataset.type}/{eps_idx:04d}/top/color/{step_idx:04d}.png"
    img_top_depth_loc = f"{cfg.dataset.type}/{eps_idx:04d}/top/depth/{step_idx:04d}.png"
    img_top_normal_loc = f"{cfg.dataset.type}/{eps_idx:04d}/top/normal/{step_idx:04d}.png"
    img_top_silhouette_loc = f"{cfg.dataset.type}/{eps_idx:04d}/top/silhouette/{step_idx:04d}.png"

    # .astype(np.uint8)
    # imgs_top[2] = (255. * (imgs_top[2] + 1) / 2)
    # img_as_ubyte

    imageio.imwrite(f"{BASE_PATH}/{cfg.dataset.dstdir}/{img_top_color_loc}", imgs_top[0])
    imageio.imwrite(f"{BASE_PATH}/{cfg.dataset.dstdir}/{img_top_depth_loc}", imgs_top[1])
    imageio.imwrite(f"{BASE_PATH}/{cfg.dataset.dstdir}/{img_top_normal_loc}", imgs_top[2])
    imageio.imwrite(f"{BASE_PATH}/{cfg.dataset.dstdir}/{img_top_silhouette_loc}", imgs_top[3])

    # save digit bottom img frames
    img_bot_color_loc = f"{cfg.dataset.type}/{eps_idx:04d}/bot/color/{step_idx:04d}.png"
    img_bot_depth_loc = f"{cfg.dataset.type}/{eps_idx:04d}/bot/depth/{step_idx:04d}.png"
    img_bot_normal_loc = f"{cfg.dataset.type}/{eps_idx:04d}/bot/normal/{step_idx:04d}.png"
    img_bot_silhouette_loc = f"{cfg.dataset.type}/{eps_idx:04d}/bot/silhouette/{step_idx:04d}.png"

    imageio.imwrite(f"{BASE_PATH}/{cfg.dataset.dstdir}/{img_bot_color_loc}", imgs_bot[0])
    imageio.imwrite(f"{BASE_PATH}/{cfg.dataset.dstdir}/{img_bot_depth_loc}", imgs_bot[1])
    imageio.imwrite(f"{BASE_PATH}/{cfg.dataset.dstdir}/{img_bot_normal_loc}", imgs_bot[2])
    imageio.imwrite(f"{BASE_PATH}/{cfg.dataset.dstdir}/{img_bot_silhouette_loc}", imgs_bot[3])
    
    # object, digit poses
    obj_pos, obj_ori = obj.get_base_pose()[0], p.getEulerFromQuaternion(obj.get_base_pose()[1])
    digit_top_pos, digit_top_ori = digit_top.get_base_pose()[0], p.getEulerFromQuaternion(digit_top.get_base_pose()[1])
    digit_bot_pos, digit_bot_ori = digit_bottom.get_base_pose()[0], p.getEulerFromQuaternion(digit_bottom.get_base_pose()[1])

    data_row = {'obj_pose': [np.hstack((np.array(obj_pos), np.array(obj_ori)))],
            'digit_top_pose': [np.hstack((np.array(digit_top_pos), np.array(digit_top_ori)))],
            'digit_bot_pose': [np.hstack((np.array(digit_bot_pos), np.array(digit_bot_ori)))],
            'img_top_color_loc': img_top_color_loc,
            'img_top_depth_loc': img_top_depth_loc,
            'img_top_normal_loc': img_top_normal_loc,
            'img_top_silhouette_loc': img_top_silhouette_loc,
            'img_bot_color_loc': img_bot_color_loc,
            'img_bot_depth_loc': img_bot_depth_loc,
            'img_bot_normal_loc': img_bot_normal_loc,
            'img_bot_silhouette_loc': img_bot_silhouette_loc}

    csvfile = f"{BASE_PATH}/{cfg.dataset.dstdir}/{cfg.dataset.type}/{eps_idx:04d}/poses_imgs.csv"
    header_flag = False if os.path.exists(csvfile) else True
    df = pd.DataFrame(data=data_row)
    df.to_csv(csvfile, mode='a', header=header_flag)

    # header_flag = False if os.path.exists(csvfile) else True
    # with open(csvfile, 'a') as csvfile:
    #   writer = csv.writer(csvfile)
    #   if (header_flag):
    #     writer.writerow(fields)
    #   writer.writerow(data)

def compute_wps_traj(cfg):

    center_pos = np.asarray(cfg.digits.top.base_position)
    end_pos = np.asarray(cfg.waypoints.end_pos)

    nsteps = cfg.waypoints.nsteps
    pos_step = (end_pos - center_pos) / (nsteps / 4.0)

    waypoints = []
    waypoints.append(center_pos)
    for step in range(0, nsteps):
        step_frac = step / nsteps

        noise = 0e-6 * (np.random.rand() - 0.5)
        pos_step = pos_step + noise

        if (step_frac <= 0.25):
            pos = waypoints[-1] + pos_step
        elif (step_frac > 0.25) & (step_frac <= 0.75):
            pos = waypoints[-1] - pos_step
        else:
            pos = waypoints[-1] + pos_step

        waypoints.append(pos)

    return waypoints

def reset_episode(cfg, obj, step_idx, eps_idx):
    obj.reset()
    step_idx = 0
    eps_idx = eps_idx + 1

    os.makedirs(f"{BASE_PATH}/{cfg.dataset.dstdir}/{cfg.dataset.type}/{eps_idx:04d}/top/color/", exist_ok=True)
    os.makedirs(f"{BASE_PATH}/{cfg.dataset.dstdir}/{cfg.dataset.type}/{eps_idx:04d}/top/depth/", exist_ok=True)
    os.makedirs(f"{BASE_PATH}/{cfg.dataset.dstdir}/{cfg.dataset.type}/{eps_idx:04d}/top/normal/", exist_ok=True)
    os.makedirs(f"{BASE_PATH}/{cfg.dataset.dstdir}/{cfg.dataset.type}/{eps_idx:04d}/top/silhouette/", exist_ok=True)

    os.makedirs(f"{BASE_PATH}/{cfg.dataset.dstdir}/{cfg.dataset.type}/{eps_idx:04d}/bot/color/", exist_ok=True)
    os.makedirs(f"{BASE_PATH}/{cfg.dataset.dstdir}/{cfg.dataset.type}/{eps_idx:04d}/bot/depth/", exist_ok=True)
    os.makedirs(f"{BASE_PATH}/{cfg.dataset.dstdir}/{cfg.dataset.type}/{eps_idx:04d}/bot/normal/", exist_ok=True)
    os.makedirs(f"{BASE_PATH}/{cfg.dataset.dstdir}/{cfg.dataset.type}/{eps_idx:04d}/bot/silhouette/", exist_ok=True)

    return obj, step_idx, eps_idx

@hydra.main(config_path="conf/", config_name="rolling_data_collection")
def main(cfg):

    # bg = cv2.imread("conf/bg_digit_240_320.jpg")
    bg = cv2.imread("conf/bg_digit_320_240.png")
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    bg = cv2.transpose(bg)

    # Initialize digits
    digits = tacto.Sensor(**cfg.tacto, background=bg)

    # Initialize World
    log.info("Initializing world")
    px.init()

    p.resetDebugVisualizerCamera(**cfg.pybullet_camera)

    # Create and initialize DIGIT URDF (top & bottom)
    digit_top = px.Body(**cfg.digits.top)
    digit_bottom = px.Body(**cfg.digits.bottom)

    digits.add_camera(digit_top.id, [-1])
    digits.add_camera(digit_bottom.id, [-1])

    # Add object to pybullet and tacto simulator
    obj = px.Body(**cfg.object)
    digits.add_body(obj)

    # Get waypoints
    wps_setter = WaypointSetter(digit_top)
    wps = compute_wps_traj(cfg)

    # Start p.stepSimulation in another thread
    set_gui_params()
    t = px.utils.SimulationThread(real_time_factor=1.0)
    t.start()

    # Run simulation
    step_idx, eps_idx = -1, -1
    obj, step_idx, eps_idx = reset_episode(cfg, obj, step_idx, eps_idx)
    nsteps = cfg.waypoints.nsteps
    contact_flag, no_contact_count = 1, 0
    pause = False

    while True:
        # reset new episode
        reset_flag = (obj.get_base_pose()[0][2] <= 0.01) | (step_idx == nsteps) | (contact_flag & (no_contact_count > 10))
        if reset_flag:
            obj, step_idx, eps_idx = reset_episode(cfg, obj, step_idx, eps_idx)
            no_contact_count = 0

            logging.info(f'Starting new episode {eps_idx:04d}.')
                
        # set waypoint
        wps_setter.update(pos=wps[step_idx])

        # render frame
        color, depth, normal, silhouette = digits.render()
        digits.updateGUI(color, depth, normal, silhouette)

        # save contact frame
        contact_flag = (np.linalg.norm(depth[0]) > 0.025)
        imgs_top = (color[0], depth[0], normal[0], silhouette[0])
        imgs_bot = (color[1], depth[1], normal[1], silhouette[1])
        if contact_flag:
            tid = p.addUserDebugText(f"episode_{eps_idx:04d}", textPosition=[
                                     0.02, 0, 0.1], textColorRGB=[1, 0, 0], textSize=2, lifeTime=1e-1)
            save_dataset_frame(cfg, eps_idx, step_idx, imgs_top, imgs_bot, obj, digit_top, digit_bottom)

        # check for lost contact
        no_contact_count = no_contact_count + 1 if (contact_flag == 0) else 0

        step_idx = step_idx + 1

        if pause:
            input("Press Enter to continue...")
            pause = False

    t.stop()

if __name__ == "__main__":
    main()
