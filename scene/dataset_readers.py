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
import sys
from PIL import Image
from typing import NamedTuple, Optional
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
import imageio
from glob import glob
import cv2 as cv
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.camera_utils import camera_nerfies_from_JSON
from scene.neural_3D_dataset_NDC import Neural3D_NDC_Dataset
from tqdm import tqdm

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float
    depth: Optional[np.array] = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]]
                 for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return K, pose


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def normalize_camera_fids(cam_infos):
    """
    Normalize all camera fid values to [0, 1] range.
    
    Args:
        cam_infos: List of CameraInfo objects
    
    Returns:
        List of CameraInfo objects with normalized fid values
    """
    if not cam_infos or len(cam_infos) == 0:
        return cam_infos
    
    fids = [cam.fid for cam in cam_infos]
    min_fid = min(fids)
    max_fid = max(fids)
    
    if max_fid == min_fid:
        normalized_fids = [0.0] * len(cam_infos)
    else:
        normalized_fids = [(fid - min_fid) / (max_fid - min_fid) for fid in fids]
    
    normalized_cams = []
    for cam, norm_fid in zip(cam_infos, normalized_fids):
        normalized_cams.append(cam._replace(fid=norm_fid))
    
    return normalized_cams


def sample_camera_infos(cam_infos, max_count):
    """
    Return an evenly spaced subset of CameraInfo objects limited to max_count.

    Args:
        cam_infos: List of CameraInfo objects.
        max_count: Maximum number of entries to keep. Non-positive or None disables limiting.

    Returns:
        Either the original list (if no limiting is applied) or a new list containing
        a deterministic, evenly spaced subset of the input cameras.
    """
    if not cam_infos or max_count is None:
        return cam_infos
    try:
        max_count = int(max_count)
    except (TypeError, ValueError):
        return cam_infos
    if max_count <= 0 or max_count >= len(cam_infos):
        return cam_infos

    indices = np.linspace(0, len(cam_infos) - 1, num=max_count, dtype=int)
    seen = set()
    ordered_indices = []
    for idx in indices:
        if idx not in seen:
            ordered_indices.append(idx)
            seen.add(idx)

    if len(ordered_indices) < max_count:
        for idx in range(len(cam_infos)):
            if idx not in seen:
                ordered_indices.append(idx)
                seen.add(idx)
            if len(ordered_indices) == max_count:
                break

    ordered_indices.sort()
    return [cam_infos[idx] for idx in ordered_indices]


def translate_cam_info(cam_infos, translate):
    """Apply translation vector to all camera centers."""
    if translate is None:
        return
    translate = np.array(translate, dtype=np.float32).reshape(3)
    for idx, cam in enumerate(cam_infos):
        if cam is None:
            continue
        new_T = np.array(cam.T, dtype=np.float32) + translate
        cam_infos[idx] = cam._replace(T=new_T)


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, msk_folder=None):
    cam_infos = []
    num_frames = len(cam_extrinsics)
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write(
            "Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE" or intr.model == "OPENCV" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        if '_' in image_name:
            image_idx = int(image_name.split('_')[-1])
            fid = image_idx / (num_frames / 2 - 1)
        else:
            fid = int(image_name) / (num_frames - 1)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, fid=fid)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'],
                       vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=4, apply_cam_norm=False, recenter_by_pcl=False):
    #sparse_name = "sparse" if os.path.exists(os.path.join(path, "sparse")) else "colmap_sparse"
    sparse_name = "colmap_sparse"
    try:
        cameras_extrinsic_file = os.path.join(path, f"{sparse_name}/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, f"{sparse_name}/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, f"{sparse_name}/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, f"{sparse_name}/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    #reading_dir = "images" if images == None else images
    reading_dir = "images"
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        print(f"Eval mode detecting, splitting train and test cameras")
        with open(f'{path}/dataset.json', 'r') as f:
            dataset_json = json.load(f)
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in dataset_json['train_ids']]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in dataset_json['val_ids']]
    else:
        print(f"No eval mode, using all cameras")
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos + test_cam_infos)

    if recenter_by_pcl:
        ply_path = os.path.join(path, f"{sparse_name}/0/points3d_recentered.ply")
    elif apply_cam_norm:
        ply_path = os.path.join(path, f"{sparse_name}/0/points3d_normalized.ply")
    else:
        ply_path = os.path.join(path, f"{sparse_name}/0/points3d.ply")
    bin_path = os.path.join(path, f"{sparse_name}/0/points3D.bin")
    txt_path = os.path.join(path, f"{sparse_name}/0/points3D.txt")
    adj_path = os.path.join(path, f"{sparse_name}/0/camera_adjustment")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        if apply_cam_norm:
            xyz += nerf_normalization["apply_translate"]
            xyz /= nerf_normalization["apply_radius"]
        if recenter_by_pcl:
            pcl_center = xyz.mean(axis=0)
            translate_cam_info(train_cam_infos, - pcl_center)
            translate_cam_info(test_cam_infos, - pcl_center)
            xyz -= pcl_center
            np.savez(adj_path, translate=-pcl_center)
        storePly(ply_path, xyz, rgb)
    elif recenter_by_pcl:
        translate = np.load(adj_path + '.npz')['translate']
        translate_cam_info(train_cam_infos, translate=translate)
        translate_cam_info(test_cam_infos, translate=translate)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            file_path = frame["file_path"]
            # Check if file_path already has the extension
            if not file_path.endswith(extension):
                file_path = file_path + extension
            cam_name = os.path.join(path, file_path)
            frame_time = frame['time']

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]

            image_path = cam_name
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array(
                [1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            mask = norm_data[..., 3:4]

            arr = norm_data[:, :, :3] * norm_data[:, :,
                                                  3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(
                np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovx
            FovX = fovy

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, width=image.size[
                                            0],
                                        height=image.size[1], fid=frame_time))
    print(f"Loaded {len(cam_infos)} cameras")
    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = []
    if os.path.exists(os.path.join(path, "transforms_test.json")):
        test_cam_infos = readCamerasFromTransforms(
            path, "transforms_test.json", white_background, extension)

    if not eval:
        print(f"No eval mode, using all cameras WARNING: for our purpose we alsays enable eval by default")
        # train_cam_infos.extend(test_cam_infos)
        # test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 10000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readDTUCameras(path, render_camera, object_camera):
    camera_dict = np.load(os.path.join(path, render_camera))
    images_lis = sorted(glob(os.path.join(path, 'image/*.png')))
    masks_lis = sorted(glob(os.path.join(path, 'mask/*.png')))
    n_images = len(images_lis)
    cam_infos = []
    cam_idx = 0
    for idx in range(0, n_images):
        image_path = images_lis[idx]
        image = np.array(Image.open(image_path))
        mask = np.array(imageio.imread(masks_lis[idx])) / 255.0
        image = Image.fromarray((image * mask).astype(np.uint8))
        world_mat = camera_dict['world_mat_%d' % idx].astype(np.float32)
        fid = camera_dict['fid_%d' % idx] / (n_images / 12 - 1)
        image_name = Path(image_path).stem
        scale_mat = camera_dict['scale_mat_%d' % idx].astype(np.float32)
        P = world_mat @ scale_mat
        P = P[:3, :4]

        K, pose = load_K_Rt_from_P(None, P)
        a = pose[0:1, :]
        b = pose[1:2, :]
        c = pose[2:3, :]

        pose = np.concatenate([a, -c, -b, pose[3:, :]], 0)

        S = np.eye(3)
        S[1, 1] = -1
        S[2, 2] = -1
        pose[1, 3] = -pose[1, 3]
        pose[2, 3] = -pose[2, 3]
        pose[:3, :3] = S @ pose[:3, :3] @ S

        a = pose[0:1, :]
        b = pose[1:2, :]
        c = pose[2:3, :]

        pose = np.concatenate([a, c, b, pose[3:, :]], 0)

        pose[:, 3] *= 0.5

        matrix = np.linalg.inv(pose)
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        FovY = focal2fov(K[0, 0], image.size[1])
        FovX = focal2fov(K[0, 0], image.size[0])
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=image.size[
                                  0], height=image.size[1],
                              fid=fid)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readNeuSDTUInfo(path, render_camera, object_camera):
    print("Reading DTU Info")
    train_cam_infos = readDTUCameras(path, render_camera, object_camera)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=[],
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readNerfiesCameras(path, skip_train_images=False):
    with open(f'{path}/scene.json', 'r') as f:
        scene_json = json.load(f)
    with open(f'{path}/metadata.json', 'r') as f:
        meta_json = json.load(f)
    with open(f'{path}/dataset.json', 'r') as f:
        dataset_json = json.load(f)

    coord_scale = scene_json['scale']
    scene_center = scene_json['center']

    name = path.split('/')[-2]
    if name.startswith('vrig'):
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 0.25
    elif name.startswith('NeRF'):
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 1.0
    elif name.startswith('interp'):
        all_id = dataset_json['ids']
        train_img = all_id[::4]
        val_img = all_id[2::4]
        all_img = train_img + val_img
        ratio = 0.5
    else:  # for hypernerf
        train_img = dataset_json['ids'][::4]
        train_img = dataset_json['ids']
        all_img = train_img
        ratio = 0.5

    train_num = len(train_img)
    train_img_set = set(train_img)

    all_cam = [meta_json[i]['camera_id'] for i in all_img]
    try:
        all_time = [meta_json[i]['time_id'] for i in all_img]
    except:
        all_time = [meta_json[i]['warp_id'] for i in all_img]
    max_time = max(all_time)
    try:
        all_time = [meta_json[i]['time_id'] / max_time for i in all_img]
    except:
        all_time = [meta_json[i]['warp_id'] / max_time for i in all_img]
    selected_time = set(all_time)

    # all poses
    all_cam_params = []
    for im in all_img:
        camera = camera_nerfies_from_JSON(f'{path}/camera/{im}.json', ratio)
        camera['position'] = camera['position'] - scene_center
        camera['position'] = camera['position'] * coord_scale
        all_cam_params.append(camera)

    all_img = [f'{path}/rgb/{int(1 / ratio)}x/{i}.png' for i in all_img]

    cam_infos = []
    for idx in range(len(all_img)):
        image_path = all_img[idx]
        image_name = Path(image_path).stem
        
        # Check if this is a training image and we should skip loading it
        is_train_image = idx < train_num or (all_img[idx].split('/')[-1].replace('.png', '') in train_img_set)
        
        if skip_train_images and is_train_image:
            # Don't load image, use None and get dimensions from metadata or default
            image = None
            # Try to get dimensions from a sample image or use defaults
            try:
                sample_img = Image.open(image_path)
                width, height = sample_img.size
                sample_img.close()
            except:
                # Fallback: use default dimensions if we can't read
                width, height = 800, 600
        else:
            image = np.array(Image.open(image_path))
            image = Image.fromarray((image).astype(np.uint8))
            width, height = image.size

        orientation = all_cam_params[idx]['orientation'].T
        position = -all_cam_params[idx]['position'] @ orientation
        focal = all_cam_params[idx]['focal_length']
        fid = all_time[idx]
        T = position
        R = orientation

        FovY = focal2fov(focal, height)
        FovX = focal2fov(focal, width)
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width,
                              height=height, fid=fid)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos, train_num, scene_center, coord_scale


def readNerfiesInfo(path, eval, skip_train_images=False):
    print("Reading Nerfies Info")
    if skip_train_images:
        print("[HyperNeRF] Skipping loading of training view images")
    cam_infos, train_num, scene_center, scene_scale = readNerfiesCameras(path, skip_train_images=skip_train_images)

    if eval:
        print(f"Eval mode detecting, splitting train and test cameras")
        train_cam_infos = cam_infos[:train_num]
        test_cam_infos = cam_infos[train_num:]
        if skip_train_images:
            print(f"[HyperNeRF] Skipped loading images for {len(train_cam_infos)} training cameras")
    else:
        print(f"No eval mode, using all cameras")
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        print(f"Generating point cloud from nerfies...")

        xyz = np.load(os.path.join(path, "points.npy"))
        xyz = (xyz - scene_center) * scene_scale
        num_pts = xyz.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromNpy(path, npy_file, split, hold_id, num_images):
    cam_infos = []
    video_paths = sorted(glob(os.path.join(path, 'frames/*')))
    poses_bounds = np.load(os.path.join(path, npy_file))

    poses = poses_bounds[:, :15].reshape(-1, 3, 5)
    H, W, focal = poses[0, :, -1]

    n_cameras = poses.shape[0]
    poses = np.concatenate(
        [poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    bottoms = np.array([0, 0, 0, 1]).reshape(
        1, -1, 4).repeat(poses.shape[0], axis=0)
    poses = np.concatenate([poses, bottoms], axis=1)
    poses = poses @ np.diag([1, -1, -1, 1])

    i_test = np.array(hold_id)
    video_list = i_test if split != 'train' else list(
        set(np.arange(n_cameras)) - set(i_test))

    for i in video_list:
        video_path = video_paths[i]
        c2w = poses[i]
        images_names = sorted(os.listdir(video_path))
        n_frames = num_images

        matrix = np.linalg.inv(np.array(c2w))
        R = np.transpose(matrix[:3, :3])
        T = matrix[:3, 3]

        for idx, image_name in enumerate(images_names[:num_images]):
            image_path = os.path.join(video_path, image_name)
            image = Image.open(image_path)
            frame_time = idx / (n_frames - 1)

            FovX = focal2fov(focal, image.size[0])
            FovY = focal2fov(focal, image.size[1])

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovX=FovX, FovY=FovY,
                                        image=image,
                                        image_path=image_path, image_name=image_name,
                                        width=image.size[0], height=image.size[1], fid=frame_time))

            idx += 1
    return cam_infos


def _neural3d_dataset_to_cam_infos(dataset, max_frames_per_camera, skip_train_images=False, is_train_split=False):
    """Convert Neural3D dataset samples to CameraInfo entries.
    
    Args:
        dataset: Neural3D dataset instance
        max_frames_per_camera: Maximum frames per camera to load
        skip_train_images: If True and is_train_split is True, skip loading images
        is_train_split: Whether this is the training split
    """
    cam_infos = []
    width, height = dataset.img_wh
    width = int(width)
    height = int(height)
    focal = float(dataset.focal[0])
    fov_x = focal2fov(focal, width)
    fov_y = focal2fov(focal, height)
    frames_per_camera = dataset.time_number if dataset.time_number > 0 else len(dataset.image_paths)
    total_images = len(dataset.image_paths)
    total_cameras = dataset.cam_number if dataset.cam_number > 0 else max(total_images // max(1, frames_per_camera), 1)
    target_frames = max_frames_per_camera if max_frames_per_camera and max_frames_per_camera > 0 else frames_per_camera
    target_frames = min(target_frames, frames_per_camera)
    target_size = (width, height)

    image_index = 0
    
    for cam_idx in tqdm(range(total_cameras), desc="Loading cameras"):
        camera_indices = []
        while image_index < total_images and len(camera_indices) < frames_per_camera:
            camera_indices.append(image_index)
            image_index += 1
        if not camera_indices:
            break
        camera_indices = camera_indices[:target_frames]

        for frame_idx in camera_indices:
            image_path = dataset.image_paths[frame_idx]
            
            # Skip loading image if this is training split and flag is set
            if skip_train_images and is_train_split:
                image_copy = None
            else:
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    if img.size != target_size:
                        img = img.resize(target_size, Image.LANCZOS)
                    image_copy = img.copy()
            
            image_name = Path(image_path).stem
            fid = float(dataset.image_times[frame_idx])
            R, T = dataset.image_poses[frame_idx]
            R = np.array(R, dtype=np.float32)
            T = np.array(T, dtype=np.float32)

            cam_infos.append(CameraInfo(
                uid=len(cam_infos),
                R=R,
                T=T,
                FovY=fov_y,
                FovX=fov_x,
                image=image_copy,
                image_path=image_path,
                image_name=image_name,
                width=width,
                height=height,
                fid=fid
            ))
    return cam_infos


def _load_plenoptic_point_cloud(path):
    candidates = [
        "points3D_downsample2.ply",
        "points3D_dense.ply",
        "points3d.ply",
        "points3D.ply",
    ]
    for candidate in candidates:
        ply_path = os.path.join(path, candidate)
        if os.path.exists(ply_path):
            try:
                return fetchPly(ply_path), ply_path
            except Exception:
                continue
    return None, os.path.join(path, candidates[0])


def readPlenopticVideoDataset(path, eval, num_images, hold_id=[0], skip_train_images=False):
    if len(hold_id) > 1:
        print(f"[PlenopticVideo] Multiple hold IDs provided ({hold_id}); using the first one.")
    eval_index = hold_id[0] if hold_id else 0
    max_frames = num_images if num_images and num_images > 0 else None

    print("Initializing Neural3D dataset (train split)")
    train_dataset = Neural3D_NDC_Dataset(
        path,
        split="train",
        downsample=2.0,
        time_scale=1.0,
        scene_bbox_min=[-2.5, -2.0, -1.0],
        scene_bbox_max=[2.5, 2.0, 1.0],
        eval_index=eval_index,
    )
    train_cam_infos = _neural3d_dataset_to_cam_infos(train_dataset, max_frames, skip_train_images=skip_train_images, is_train_split=True)
    print(f"train_cam_infos: {len(train_cam_infos)}")
    if skip_train_images:
        print(f"[PlenopticVideo] Skipped loading images for {len(train_cam_infos)} training cameras")
    print("Initializing Neural3D dataset (eval split)")
    test_dataset = Neural3D_NDC_Dataset(
        path,
        split="test",
        downsample=2.0,
        time_scale=1.0,
        scene_bbox_min=[-2.5, -2.0, -1.0],
        scene_bbox_max=[2.5, 2.0, 1.0],
        eval_index=eval_index,
    )
    test_cam_infos = _neural3d_dataset_to_cam_infos(test_dataset, max_frames, skip_train_images=False, is_train_split=False)
    print(f"test_cam_infos: {len(test_cam_infos)}")

    if not eval:
        train_cam_infos = train_cam_infos + test_cam_infos
        test_cam_infos = []
    if not train_cam_infos:
        raise RuntimeError("No plenoptic cameras were loaded. Check dataset path and preprocessing.")

    nerf_normalization = getNerfppNorm(train_cam_infos)
    pcd, ply_path = _load_plenoptic_point_cloud(path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readPanopticSports(path, eval=True):
    """Read Panoptic Sports dataset.

    Assumes a metadata JSON at path/train_meta.json with keys:
      - w, h: image dimensions
      - fn[t][c]: image filename relative to path/ims/
      - k[t][c]: 3x3 intrinsics
      - w2c[t][c]: 4x4 world-to-camera matrix
    Images are stored under path/ims/.
    Optionally uses path/init_pt_cld.npz with ["data"] Nx6 (x,y,z,r,g,b) to seed a point cloud.
    """
    print("Reading Panoptic Sports Dataset")

    meta_path = os.path.join(path, "train_meta.json")
    with open(meta_path, 'r') as f:
        md = json.load(f)

    width = int(md['w'])
    height = int(md['h'])
    filenames = md['fn']
    intrinsics = md['k']
    w2c_mats = md['w2c']

    num_timesteps = len(filenames)
    train_cam_infos = []
    test_cam_infos = []

    for t in range(num_timesteps):
        num_cams_t = len(filenames[t])
        for c in range(num_cams_t):
            fn = filenames[t][c]
            image_path = os.path.join(path, "ims", fn)
            image = Image.open(image_path)

            K = np.array(intrinsics[t][c], dtype=np.float32)
            W2C = np.array(w2c_mats[t][c], dtype=np.float32)

            # Convert world-to-camera to the (R, T) convention expected by getWorld2View2
            R = W2C[:3, :3].T
            T = W2C[:3, 3]

            fx, fy = float(K[0, 0]), float(K[1, 1])
            FovX = focal2fov(fx, width)
            FovY = focal2fov(fy, height)

            if num_timesteps > 1:
                fid = t / (num_timesteps - 1)
            else:
                fid = 0.0

            cam_info = CameraInfo(
                uid=len(train_cam_infos),
                R=R, T=T, FovY=FovY, FovX=FovX,
                image=image, image_path=image_path,
                image_name=os.path.splitext(os.path.basename(fn))[0],
                width=width, height=height,
                fid=fid
            )
            train_cam_infos.append(cam_info)

    if eval and len(train_cam_infos) > 0:
        split_idx = int(0.9 * len(train_cam_infos))
        test_cam_infos = train_cam_infos[split_idx:]
        train_cam_infos = train_cam_infos[:split_idx]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        init_pt_cld_path = os.path.join(path, "init_pt_cld.npz")
        if os.path.exists(init_pt_cld_path):
            init_pt_cld = np.load(init_pt_cld_path)["data"]
            points = init_pt_cld[:, :3].astype(np.float32)
            colors = init_pt_cld[:, 3:6]
            # Ensure colors are uint8 in 0..255
            if colors.max() <= 1.0:
                colors = (colors * 255.0).clip(0, 255)
            colors = colors.astype(np.uint8)
            storePly(ply_path, points, colors)
        else:
            # Fallback: create a small random cloud if no seed exists
            num_pts = 10000
            print(f"Generating random point cloud ({num_pts})...")
            xyz = np.random.random((num_pts, 3)).astype(np.float32) * 2.6 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
            storePly(ply_path, xyz, (SH2RGB(shs) * 255).astype(np.uint8))

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path
    )

    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,  # colmap dataset reader from official 3D Gaussian [https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/]
    "Blender": readNerfSyntheticInfo,  # D-NeRF dataset [https://drive.google.com/file/d/1uHVyApwqugXTFuIRRlE4abTW8_rrVeIK/view?usp=sharing]
    "DTU": readNeuSDTUInfo,  # DTU dataset used in Tensor4D [https://github.com/DSaurus/Tensor4D]
    "nerfies": readNerfiesInfo,  # NeRFies & HyperNeRF dataset proposed by [https://github.com/google/hypernerf/releases/tag/v0.1]
    "plenopticVideo": readPlenopticVideoDataset,  # Neural 3D dataset in [https://github.com/facebookresearch/Neural_3D_Video]
    "PanopticSports": readPanopticSports,
}