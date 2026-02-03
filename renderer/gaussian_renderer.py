import copy
import os
import traceback
from typing import List
import imageio
import numpy as np
import torch
import torch.nn
from tqdm import tqdm
from pathlib import Path

from compression.compression_exp import run_single_decompression
from gaussian_renderer import render_simple,convert_pointclouds
from scene import GaussianModel
from scene.cameras import CustomCam
from renderer.base_renderer import Renderer
from splatviz_utils.dict_utils import EasyDict
from loguru import logger
import time
from datetime import datetime


def sort_polygon_points_on_plane(points, plane_normal):
    """
    输入：points 是 (N, 3) 的 numpy array，表示平面上的点
         plane_normal 是平面法向量
    输出：按照极角顺序排序后的 (N, 3) numpy array
    """
    points = np.array(points)
    center = points.mean(axis=0)

    # 1. 归一化法向量
    n = plane_normal / np.linalg.norm(plane_normal)

    # 2. 构造平面局部坐标轴 (u, v)
    # 任取一个不平行的向量与法向量叉乘
    arbitrary = np.array([1, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(n, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)  # 保证 uvn 构成右手坐标系

    # 3. 投影点到 u-v 平面，并计算极角
    rel = points - center
    x = rel @ u  # shape (N,)
    y = rel @ v
    angles = np.arctan2(y, x)  # 极角 [-π, π]

    # 4. 按极角排序
    indices = np.argsort(angles)
    sorted_points = points[indices]

    return sorted_points

def compute_aabb(xyz: torch.Tensor,scale=0.1):
    """
    input: xyz ->(N, 3)  torch.Tensor
    output : bbox_min, bbox_max，(3,)  torch.Tensor
    """
    assert xyz.ndim == 2 and xyz.shape[1] == 3, "input mush be  (N, 3) point cloud"
    xyz=xyz*scale # have one larger box
    bbox_min = xyz.min(dim=0).values  # min of each column
    bbox_max = xyz.max(dim=0).values  # max of each column

    # 如果某个维度范围为0，则人为设置一个默认范围
    bbox_range = bbox_max - bbox_min
    default_range = 3.0  # 你可以改为 0.1 或其他值

    for i in range(3):
        if xyz.shape[0]<10:
            bbox_min[i] -= default_range / 2
            bbox_max[i] += default_range / 2
            
            bbox_min[i]*=scale
            bbox_max[i]*=scale
    
    return bbox_min, bbox_max

def intersect_plane_aabb(plane_normal, plane_d, bbox_min, bbox_max, eps=1e-6):
    """compute intersection with AABB"""
    # 8 corners
    corners = np.array([
        [bbox_min[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_min[1], bbox_min[2]],
        [bbox_min[0], bbox_max[1], bbox_min[2]],
        [bbox_max[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_min[1], bbox_max[2]],
        [bbox_min[0], bbox_max[1], bbox_max[2]],
        [bbox_max[0], bbox_max[1], bbox_max[2]],
    ])

    # 12 edges
    edges = [
        (0,1), (1,3), (3,2), (2,0),  # bottom
        (4,5), (5,7), (7,6), (6,4),  # top
        (0,4), (1,5), (2,6), (3,7)   # sides
    ]

    points = []

    for i0, i1 in edges:
        p0, p1 = corners[i0], corners[i1]
        f0 = np.dot(plane_normal, p0) + plane_d
        f1 = np.dot(plane_normal, p1) + plane_d

        # whether across plane
        if f0 * f1 < -eps:
            t = -f0 / (f1 - f0)
            intersection = p0 + t * (p1 - p0)
            points.append(intersection)

    return points
class GaussianRenderer(Renderer):
    def __init__(self, num_parallel_scenes=16):
        super().__init__()
        self.num_parallel_scenes = num_parallel_scenes
        self.gaussian_models: List[GaussianModel | None] = [None] * num_parallel_scenes
        self._current_ply_file_paths: List[str | None] = [None] * num_parallel_scenes
        self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32).to("cuda")
        self._last_num_scenes = 0

    def _render_impl(
        self,
        res,
        fov,
        edit_text,
        eval_text,
        resolution,
        ply_file_paths,
        cam_params,
        current_ply_names,
        background_color,
        video_cams=[],
        render_depth=False,
        render_alpha=False,
        img_normalize=False,
        use_splitscreen=False,
        highlight_border=False,
        save_ply_path=None,
        slider={},
        **other_args,
    ):
        cam_params = cam_params.to("cuda")
        slider = EasyDict(slider)
        if len(ply_file_paths) == 0:
            res.error = "Select a .ply file"
            return

        # Remove old scenes
        if len(ply_file_paths) < self._last_num_scenes:
            for i in range(ply_file_paths, self.num_parallel_scenes):
                self.gaussian_models[i] = None
            self._last_num_scenes = len(ply_file_paths)
        
        sub_names = [
            os.path.basename(os.path.dirname(p))
            for p in ply_file_paths
        ]
        other_args['sub_name']=sub_names[0]# only take one
        # logger.info(sub_names)
        images = []
        for scene_index, ply_file_path in enumerate(ply_file_paths):
            # Load
            if ply_file_path != self._current_ply_file_paths[scene_index]:
                self.gaussian_models[scene_index] = self._load_model(ply_file_path)
                self._current_ply_file_paths[scene_index] = ply_file_path

            # Edit
            gs: GaussianModel = copy.deepcopy(self.gaussian_models[scene_index])
            try:
                exec(self.sanitize_command(edit_text))
            except Exception as e:
                error = traceback.format_exc()
                error += str(e)
                res.error = error

            # Render video
            if len(video_cams) > 0:
                self.render_video("./_videos", video_cams, gs)


            
            
            # Render current view
            fov_rad = fov / 360 * 2 * np.pi
            render_cam = CustomCam(resolution, resolution, fovy=fov_rad, fovx=fov_rad, extr=cam_params)

            if "vizPlane" in other_args.keys() and other_args['vizPlane']:
                # If I need to viz the plane, I need to have some thing different
                if "scale" in other_args.keys():
                    bbox_min, bbox_max = compute_aabb(gs.get_xyz.detach().cpu(), scale=other_args['scale'])
                else:
                    bbox_min, bbox_max = compute_aabb(gs.get_xyz.detach().cpu(), scale=1.0)

                if "clipping_planes" in other_args.keys():
                    if len(other_args['clipping_planes']) == 1:
                        clipper_n = other_args['clipping_planes'][0]
                        intersections = intersect_plane_aabb(clipper_n['normal'], clipper_n['d'], bbox_min.numpy(), bbox_max.numpy())

                        if len(intersections) > 3:
                            #vis the plane when we have more than three intersections
                            other_args['vizPlane']=True
                            other_args['n_inters']=len(intersections)
                            sorted_pts = sort_polygon_points_on_plane(intersections, clipper_n['normal'])
                            intersections_tensor = torch.from_numpy(np.stack(sorted_pts, axis=0)).cuda().float()
                            other_args['intersections_tensor'] = intersections_tensor
                        else:
                            other_args['vizPlane']=False
                    else:
                        other_args['vizPlane'] = False
                else:
                    other_args['vizPlane']=False

            # Render current view with EVS splitting
            render = render_simple(viewpoint_camera=render_cam, pc=gs, bg_color=background_color.to("cuda"),**other_args)


            # logger.info(f"Now FPS:{1/render['ellapsed_time']}")
            if render_alpha:
                images.append(render["alpha"])
            elif render_depth:
                images.append(render["depth"] / render["depth"].max())
            else:
                images.append(render["render"])

            # Pass EVS stats to result for UI display
            if "evs_stats" in render:
                res.evs_stats = render["evs_stats"]

            # Save ply
            if save_ply_path is not None:
                self.save_ply(gs, save_ply_path)

        self._return_image(
            images,
            res,
            normalize=img_normalize,
            use_splitscreen=use_splitscreen,
            highlight_border=highlight_border,
        )

        res.mean_xyz = torch.mean(gs.get_xyz, dim=0)
        res.std_xyz = torch.std(gs.get_xyz)
        if len(eval_text) > 0:
            res.eval = eval(eval_text)

    def _load_model(self, ply_file_path):
        if ply_file_path.endswith(".ply"):
            model = GaussianModel(sh_degree=0, disable_xyz_log_activation=True)
            model.load_ply(ply_file_path)
        elif ply_file_path.endswith("compression_config.yml"):
            model = run_single_decompression(Path(ply_file_path).parent.absolute())
        else:
            raise NotImplementedError("Only .ply or .yml files are supported.")
        return model

    def render_video(self, save_path, video_cams, gaussian):
        os.makedirs(save_path, exist_ok=True)
        filename = f"{save_path}/rotate_{len(os.listdir(save_path))}.mp4"
        video = imageio.get_writer(filename, mode="I", fps=30, codec="libx264", bitrate="16M", quality=10)
        for render_cam in tqdm(video_cams):
            img =  render_simple(viewpoint_camera=video_cams, pc=gaussian, bg_color=self.bg_color)["render"]
            img = (img * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            video.append_data(img)
        video.close()
        print(f"Video saved in {filename}.")

    @staticmethod
    def save_ply(gaussian, save_ply_path):
        os.makedirs(save_ply_path, exist_ok=True)
        save_path = os.path.join(save_ply_path, f"model_{len(os.listdir(save_ply_path))}.ply")
        print("Model saved in", save_path)
        gaussian.save_ply(save_path)
