import logging

import os
import numpy as np
import csv
import pandas as pd

import imageio
import cv2
from attrdict import AttrDict

import cv2
import hydra
import pybullet as p
import tacto  # Import TACTO

import pybulletX as px

from collection_utils import *

log = logging.getLogger(__name__)

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        
@hydra.main(config_path="conf/", config_name="cube")
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
    
    # Get (position, orientation) waypoints
    wps_setter = WaypointSetter(obj)
    pose_wps = get_wps_pose(start_xyz_list=np.asarray(cfg.waypoints.start_xyz), start_rpy_list=np.asarray(cfg.waypoints.start_rpy),
                            end_xyz_list=np.asarray(cfg.waypoints.end_xyz), end_rpy_list=np.asarray(cfg.waypoints.end_rpy), nsteps=cfg.waypoints.nsteps)

    # Init data logger object
    data_logger = DataLogger(cfg)

    # Create control panel to control the 6DoF pose of the object
    panel = px.gui.PoseControlPanel(obj, **cfg.object_control_panel)
    panel.start()
    log.info("Use the slides to move the object until in contact with the DIGIT")

    # Start p.stepSimulation in another thread
    # set_gui_params()
    t = px.utils.SimulationThread(real_time_factor=1.0)
    t.start()

    # Run simulation
    step_idx, eps_idx = -1, -1
    obj, step_idx, eps_idx = data_logger.start_new_episode(obj, step_idx, eps_idx)
    nsteps = cfg.waypoints.nsteps
    contact_flag, no_contact_count = 1, 0
    
    while True:
        # reset new episode
        reset_flag = (obj.get_base_pose()[0][2] <= 0.01) | (step_idx == nsteps) | (contact_flag & (no_contact_count > 10))
        if reset_flag:
            data_logger.save_episode_dataset(eps_idx)
            obj, step_idx, eps_idx = data_logger.start_new_episode(obj, step_idx, eps_idx)
            no_contact_count = 0

            logging.info(f'Starting new episode {eps_idx:04d}.')
                
        # set waypoint
        pos = pose_wps[step_idx][0] if (cfg.waypoints.pos_ctrl == True) else None
        ori = pose_wps[step_idx][1] if (cfg.waypoints.ori_ctrl == True) else None
        wps_setter.update(pos=pos, ori=ori)

        # render frame
        color, depth, normal, silhouette = digits.render()
        digits.updateGUI(color, depth, normal, silhouette)

        # save contact frame
        contact_depth_thresh = 0.025
        contact_flag = (np.linalg.norm(depth[0]) > contact_depth_thresh) | (np.linalg.norm(depth[1]) > contact_depth_thresh)
        imgs_top = (color[0], depth[0], normal[0], silhouette[0])
        imgs_bot = (color[1], depth[1], normal[1], silhouette[1])
        if contact_flag:
            tid = p.addUserDebugText(f"episode_{eps_idx:04d}", textPosition=[
                                     0.02, 0, 0.1], textColorRGB=[1, 0, 0], textSize=2, lifeTime=1e-1)
            data_logger.save_episode_step(eps_idx, step_idx, imgs_top, imgs_bot, obj, digit_top, digit_bottom, renderer=digits.renderer)

        # check for lost contact
        no_contact_count = no_contact_count + 1 if (contact_flag == 0) else 0

        step_idx = step_idx + 1
        
        if (eps_idx > cfg.dataset.max_episodes):
            break

    t.stop()

if __name__ == "__main__":
    main()
