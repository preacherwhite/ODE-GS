#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, normalize_camera_fids, sample_camera_infos
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON


def _resolve_max_eval_image_limit(args):
    """
    Determine the max_eval_images value from either the provided args or a nested stage1 config.
    Returns None when no positive limit is defined.
    """
    def _extract(candidate):
        if candidate is None:
            return None
        try:
            value = int(candidate)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    limit = _extract(getattr(args, 'max_eval_images', None))
    if limit is not None:
        return limit

    stage1_cfg = getattr(args, 'stage1', None)
    if stage1_cfg is not None:
        limit = _extract(getattr(stage1_cfg, 'max_eval_images', None))
        if limit is not None:
            return limit
    return None


class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        load_iteration=None,
        shuffle=True,
        resolution_scales=[1.0],
        skip_pointcloud_init: bool = False,
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        print("source_path: ", args.source_path)
        if os.path.exists(os.path.join(args.source_path, "sparse")) or os.path.exists(os.path.join(args.source_path, "colmap_sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "cameras_sphere.npz")):
            print("Found cameras_sphere.npz file, assuming DTU data set!")
            scene_info = sceneLoadTypeCallbacks["DTU"](args.source_path, "cameras_sphere.npz", "cameras_sphere.npz")
        elif os.path.exists(os.path.join(args.source_path, "dataset.json")):
            print("Found dataset.json file, assuming Nerfies data set!")
            skip_train_images = getattr(args, 'skip_train_images', False)
            scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, args.eval, skip_train_images=skip_train_images)
        # elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")) and os.path.exists(os.path.join(args.source_path,"points3D_downsample2.ply")):
        #     print("Found poses_bounds.npy and points3D_downsample2.ply, assuming Neu3D data set cook spinach!")
        #     # Use fewer frames for cook_spinach to speed up training
        #     scene_info = sceneLoadTypeCallbacks["cookSpinach"](args.source_path, args.eval, -1)
        elif os.path.exists(os.path.join(args.source_path, "train_meta.json")) and os.path.exists(os.path.join(args.source_path, "ims")):
            print("Found train_meta.json and ims/, assuming PanopticSports data set!")
            scene_info = sceneLoadTypeCallbacks["PanopticSports"](args.source_path, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            print("Found calibration_full.json, assuming plenopticVideo data set!")
            skip_train_images = getattr(args, 'skip_train_images', False)
            scene_info = sceneLoadTypeCallbacks["plenopticVideo"](args.source_path, args.eval, -1, skip_train_images=skip_train_images)
        elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
            print("Found calibration_full.json, assuming Dynamic-360 data set!")
            scene_info = sceneLoadTypeCallbacks["dynamic360"](args.source_path)
        else:
            assert False, "Could not recognize scene type!"

        max_eval_images = _resolve_max_eval_image_limit(args)
        self.max_eval_images_limit = max_eval_images
        if max_eval_images is not None and scene_info.test_cameras:
            original_test_count = len(scene_info.test_cameras)
            limited_test_cameras = sample_camera_infos(scene_info.test_cameras, max_eval_images)
            if len(limited_test_cameras) < original_test_count:
                print(f"Limiting evaluation cameras to {len(limited_test_cameras)} / {original_test_count} (max_eval_images={max_eval_images})")
                scene_info = scene_info._replace(test_cameras=limited_test_cameras)

        # Normalize all camera fid values to [0, 1] range
        # Normalize train and test cameras together to ensure consistent range
        print("Normalizing camera fids to [0, 1] range...")
        all_cameras = scene_info.train_cameras + scene_info.test_cameras
        if all_cameras:
            all_cameras_normalized = normalize_camera_fids(all_cameras)
            train_count = len(scene_info.train_cameras)
            scene_info = scene_info._replace(
                train_cameras=all_cameras_normalized[:train_count],
                test_cameras=all_cameras_normalized[train_count:]
            )
        else:
            scene_info = scene_info._replace(
                train_cameras=normalize_camera_fids(scene_info.train_cameras),
                test_cameras=normalize_camera_fids(scene_info.test_cameras)
            )

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
                                                                   'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                            args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                           args)

        if skip_pointcloud_init:
            pass
        elif self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"),
                                    og_number_points=len(scene_info.point_cloud.points))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def setTrainCameras(self, train_cameras, scale=1.0):
        self.train_cameras[scale] = train_cameras

    def setTestCameras(self, test_cameras, scale=1.0):
        self.test_cameras[scale] = test_cameras
    def apply_time_split(self, split_ratio=0.8):
        """
        Split cameras into training and validation sets based on time index (fid).
        
        Args:
            split_ratio: Float between 0 and 1 indicating the ratio of frames to use for training
                        (default: 0.8 means 80% for training, 20% for validation)
        
        Returns:
            train_cameras: List of cameras for training
            test_cameras: List of cameras for validation
        """
        # Get all cameras from both train and test sets (for the default resolution scale)
        all_cameras = []
        for cam in self.getTrainCameras():
            all_cameras.append(cam)
        for cam in self.getTestCameras():
            all_cameras.append(cam)
        
        # Sort by frame ID (time index)
        all_cameras.sort(key=lambda x: x.fid)
        
        # Calculate split index based on time
        total_frames = len(all_cameras)
        split_idx = int(total_frames * split_ratio)
        
        # Create new train and test camera lists
        train_cameras = all_cameras[:split_idx]
        test_cameras = all_cameras[split_idx:]
        
        # Update the camera dictionaries for all resolution scales
        for scale in self.train_cameras.keys():
            self.train_cameras[scale] = train_cameras
            self.test_cameras[scale] = test_cameras
        
        print(f"Time-based split: {len(train_cameras)} training frames, {len(test_cameras)} validation frames")
        
        return train_cameras, test_cameras