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
import open3d as o3d
import torch
import math
from evsplitting import EVSSettings, EVSplitter
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import time
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
# from ray_tracer import tracer
from loguru import logger
import numpy as np
from PIL import Image
import os
from .evs_scene_graph import EVSSceneGraph, EVSSplitter as EVSSceneGraphSplitter, EVSSceneAssembler, EVSMemoryManager
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    evs_raster_settings = EVSSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    
    evs_rasterizer=EVSplitter(raster_settings=evs_raster_settings)
    

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = evs_rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}


def render_simple(viewpoint_camera, pc: GaussianModel, bg_color: torch.Tensor, scaling_modifier=1.0,
           override_color=None, debug=False,**other_args):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    ellapsed_time=0.0
    start = time.perf_counter()
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    # check clipping info  DL: precheck the params, only the widget is click we can get the default params
    n_clips=0
    clippers= torch.tensor([1.0, 0.0,0.0, 1], dtype=pc.get_xyz.dtype,device=pc.get_xyz.device)  # (normal, distance)
    if "clipping_planes" in other_args.keys():
        # logger.info(f"clipping_planes:{other_args['clipping_planes']}")
        n_clips=len(other_args['clipping_planes'])
        clippers= torch.zeros(n_clips, 4, device=pc.get_xyz.device, dtype=pc.get_xyz.dtype)
        for n in range(n_clips):# fill all clippers
            clipper_n=other_args['clipping_planes'][n]
            clippers[n][0]=clipper_n['normal'][0]
            clippers[n][1]=clipper_n['normal'][1]
            clippers[n][2]=clipper_n['normal'][2]
            clippers[n][3]=clipper_n['d']
        # logger.info(f"{n_clips} clippers : \n {clippers}")
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # check evs splitting info
    enable_evs=False
    if "enable_evs" in other_args.keys():
        enable_evs=other_args['enable_evs']
        # logger.info(f"enable_evs-{enable_evs}")
    evs_debug=False
    if "evs_debug" in other_args.keys():
        evs_debug=other_args['evs_debug']
        # logger.info(f"evs_debug-{evs_debug}")

    # check clip_model flag - when enabled, cull Gaussians on the wrong side of clipping plane
    clip_model=False
    if "clip_model" in other_args.keys():
        clip_model=other_args['clip_model']
        # logger.info(f"clip_model-{clip_model}")

    # Benefit-cost split control parameters
    evs_split_mode = other_args.get('evs_split_mode', 0)    # 0=naive, 1=proxy_control
    evs_cost_mode = other_args.get('evs_cost_mode', 0)      # 0=1-min(Cl,Cr), 1=|Cl-Cr|
    evs_lambda = other_args.get('evs_lambda', 1.0)          # benefit-cost threshold

    # RaRa Clipper parameters (kept for C++ backend compatibility, but not used)
    rr_clipping = False
    rr_strategy = False
    oenD_gs_strategy = False
    vizPlane = other_args.get('vizPlane', False)  # Read from UI
    n_inters = other_args.get('n_inters', 0)  # Read computed intersection count
    # Read computed intersection points, or use default
    if 'intersections_tensor' in other_args:
        intersections_tensor = other_args['intersections_tensor']
    else:
        intersections_tensor = torch.zeros((1, 3), dtype=pc.get_xyz.dtype, device=pc.get_xyz.device)
    evs_raster_settings = EVSSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=evs_debug,
        rr_clipping=rr_clipping,
        rr_strategy=rr_strategy,
        oenD_gs_strategy=oenD_gs_strategy,
        n_clips=n_clips,
        clipprs=clippers,
        vizPlane=vizPlane,
        n_inters=n_inters,
        intersections_tensor=intersections_tensor,
        clip_model=clip_model,
        evs_split_mode=evs_split_mode,
        evs_cost_mode=evs_cost_mode,
        evs_lambda=evs_lambda
    )
    
    # rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    evs_rasterizer=EVSplitter(raster_settings=evs_raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    scales = pc.get_scaling
    rotations = pc.get_rotation
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    convert_shs_python = False
    colors_precomp = None
    if override_color is None:
        if convert_shs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if not enable_evs:
        rendered_image, rendered_depth, rendered_alpha, radii = evs_rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.

        # torch.cuda.synchronize()
        end_time = time.perf_counter()

        ellapsed_time=end_time-start

        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "alpha": rendered_alpha,
            "depth": rendered_depth,
            "ellapsed_time":ellapsed_time
        }

    # -----------------------------------------
    # Adaptive Multi-Pass EVS Splitting
    # -----------------------------------------
    evs_max_passes = other_args.get('evs_max_passes', 2)
    evs_min_split_threshold = other_args.get('evs_min_split_threshold', 0)
    evs_mode = other_args.get('evs_mode', 'naive')  # 'naive' | 'scenegraph'
    evs_measure_memory = other_args.get('evs_measure_memory', False)

    # Helper function to calculate tensor memory in bytes
    def tensor_memory_bytes(t):
        return t.numel() * t.element_size()

    # Reset CUDA peak memory stats for accurate measurement
    cuda_mem_before = 0
    if evs_measure_memory:
        torch.cuda.reset_peak_memory_stats()
        cuda_mem_before = torch.cuda.memory_allocated()

    # -----------------------------------------
    # Scene-Graph based EVS (memory-efficient)
    # -----------------------------------------
    if evs_mode == 'scenegraph':
        # Create scene graph with references (no clone!)
        graph = EVSSceneGraph.create(means3D, scales, rotations, opacity, shs)
        splitter = EVSSceneGraphSplitter(evs_rasterizer)

        total_splits = 0
        for pass_idx in range(evs_max_passes):
            graph, num_gs_split = splitter.split_generation(
                graph,
                generation_idx=pass_idx,
                means2D=means2D,
                colors_precomp=colors_precomp,
                cov3D_precomp=cov3D_precomp
            )

            total_splits += num_gs_split

            if num_gs_split <= evs_min_split_threshold:
                logger.info(f"[EVS-SG] Early termination: only {num_gs_split} splits (threshold: {evs_min_split_threshold})")
                break

        # Calculate bytes per Gaussian dynamically
        P = means3D.shape[0]
        original_gs_bytes = (
            tensor_memory_bytes(means3D) +
            tensor_memory_bytes(scales) +
            tensor_memory_bytes(rotations) +
            tensor_memory_bytes(opacity) +
            tensor_memory_bytes(shs)
        )
        bytes_per_gs = original_gs_bytes // P if P > 0 else 236

        # Optimization: if no splits occurred, use original data directly (no assembly needed)
        if total_splits == 0:
            # No splits - use original data, no extra memory needed!
            rendered_image, rendered_depth, rendered_alpha, radii = evs_rasterizer(
                means3D=means3D,
                means2D=None,
                shs=shs,
                colors_precomp=None,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=None
            )

            torch.cuda.synchronize()
            end_time = time.perf_counter()
            ellapsed_time = end_time - start

            # Only keep_mask was allocated (minimal overhead)
            sg_memory_bytes = graph.keep_mask.numel() * 1  # bool = 1 byte

            # Get CUDA peak memory (real system measurement)
            cuda_peak_bytes = torch.cuda.max_memory_allocated() - cuda_mem_before if evs_measure_memory else 0

            evs_stats = {
                'original_gs_count': P,
                'final_gs_count': P,
                'total_splits': 0,
                'kept_original': P,
                'total_children': 0,
                'num_generations': 0,
                'bytes_per_gs': bytes_per_gs,
                'memory_overhead_mb': 0,
                'evs_mode': 'scenegraph',
                'gs_memory_mb': sg_memory_bytes / (1024 * 1024),  # only keep_mask
                'peak_memory_mb': cuda_peak_bytes / (1024 * 1024),  # CUDA measured peak
                'fps': 1.0 / ellapsed_time if ellapsed_time > 0 else 0,
            }
        else:
            # Splits occurred - need assembly
            assembled = EVSSceneAssembler.assemble(graph, evs_debug)

            rendered_image, rendered_depth, rendered_alpha, radii = evs_rasterizer(
                means3D=assembled['means3D'],
                means2D=None,
                shs=assembled['shs'],
                colors_precomp=None,
                opacities=assembled['opacity'],
                scales=assembled['scales'],
                rotations=assembled['rotations'],
                cov3D_precomp=None
            )

            torch.cuda.synchronize()
            end_time = time.perf_counter()
            ellapsed_time = end_time - start

            stats = EVSSceneAssembler.get_stats(graph)

            # Calculate memory for SceneGraph approach:
            #
            # SceneGraph's TRUE advantage over Naive:
            # - Naive: must clone ALL original data at start = P × bytes_per_gs
            # - SceneGraph: only needs keep_mask (P bytes) + assembly (final × bytes_per_gs)
            #
            # Note: generations data is temporary and gets copied into assembly,
            # so we only count assembly (the final result) for fair comparison.
            #
            # The REAL savings: SceneGraph doesn't need the initial clone!
            sg_memory_bytes = 0
            sg_memory_bytes += graph.keep_mask.numel() * 1  # bool = 1 byte
            # Final assembly tensors only (generations data is copied here)
            assembly_bytes = (
                tensor_memory_bytes(assembled['means3D']) +
                tensor_memory_bytes(assembled['scales']) +
                tensor_memory_bytes(assembled['rotations']) +
                tensor_memory_bytes(assembled['opacity']) +
                tensor_memory_bytes(assembled['shs'])
            )
            sg_memory_bytes += assembly_bytes

            # Get CUDA peak memory (real system measurement)
            cuda_peak_bytes = torch.cuda.max_memory_allocated() - cuda_mem_before if evs_measure_memory else 0

            evs_stats = {
                'original_gs_count': stats['original_count'],
                'final_gs_count': stats['final_count'],
                'total_splits': stats['total_splits'],
                'kept_original': stats['kept_count'],
                'total_children': stats['total_children'],
                'num_generations': stats['num_generations'],
                'bytes_per_gs': bytes_per_gs,
                'memory_overhead_mb': stats['memory_overhead_mb'],
                'evs_mode': 'scenegraph',
                'gs_memory_mb': sg_memory_bytes / (1024 * 1024),
                'peak_memory_mb': cuda_peak_bytes / (1024 * 1024),  # CUDA measured peak
                'fps': 1.0 / ellapsed_time if ellapsed_time > 0 else 0,
            }

        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "alpha": rendered_alpha,
            "depth": rendered_depth,
            "ellapsed_time": ellapsed_time,
            "evs_stats": evs_stats,
        }

    # -----------------------------------------
    # CPU-Offload EVS (maximum GPU memory savings)
    # Uses naive-style EVS but with CPU storage of original data
    # -----------------------------------------
    elif evs_mode == 'cpu_offload':
        # Move original data to CPU (pinned memory for faster transfer)
        transfer_start = time.perf_counter()
        device = means3D.device

        means3D_cpu = means3D.cpu().pin_memory()
        scales_cpu = scales.cpu().pin_memory()
        rotations_cpu = rotations.cpu().pin_memory()
        opacity_cpu = opacity.cpu().pin_memory()
        shs_cpu = shs.cpu().pin_memory()

        # Transfer back to GPU for this frame (working copies)
        means3D_working = means3D_cpu.to(device, non_blocking=True)
        scales_working = scales_cpu.to(device, non_blocking=True)
        rotations_working = rotations_cpu.to(device, non_blocking=True)
        opacity_working = opacity_cpu.to(device, non_blocking=True)
        shs_working = shs_cpu.to(device, non_blocking=True)
        torch.cuda.synchronize()  # Wait for transfer to complete

        transfer_time = time.perf_counter() - transfer_start

        # Calculate bytes per Gaussian dynamically
        P = means3D.shape[0]
        original_gs_bytes = (
            tensor_memory_bytes(means3D) +
            tensor_memory_bytes(scales) +
            tensor_memory_bytes(rotations) +
            tensor_memory_bytes(opacity) +
            tensor_memory_bytes(shs)
        )
        bytes_per_gs = original_gs_bytes // P if P > 0 else 236

        # Use naive-style EVS splitting (same as naive mode)
        total_splits = 0

        for pass_idx in range(evs_max_passes):
            # Call EVS split on current working set
            num_splits, split_flag, split_offsets, means_new, scales_new, rots_new, opacity_new, src_ids = evs_rasterizer.split(
                means3D=means3D_working,
                means2D=means2D,
                shs=shs_working,
                colors_precomp=colors_precomp,
                opacities=opacity_working,
                scales=scales_working,
                rotations=rotations_working,
                cov3D_precomp=cov3D_precomp)

            torch.cuda.synchronize()
            num_gs_split = torch.count_nonzero(split_flag).item()
            total_splits += num_gs_split

            # Early termination if no or few Gaussians need splitting
            if num_gs_split <= evs_min_split_threshold:
                logger.info(f"[EVS-CPU] Early termination: only {num_gs_split} Gaussians need splitting (threshold: {evs_min_split_threshold})")
                break

            # Merge: keep non-split + add children
            keep_mask = (split_flag == 0)
            if opacity_new.dim() == 1:
                opacity_new = opacity_new.unsqueeze(1)

            means_keep = means3D_working[keep_mask]
            scales_keep = scales_working[keep_mask]
            rots_keep = rotations_working[keep_mask]
            opacity_keep = opacity_working[keep_mask]
            shs_keep = shs_working[keep_mask]

            # Prepare SHs for children
            P_working = means3D_working.shape[0]
            src_ids = src_ids.to(torch.int64)
            if num_splits > 0:
                src_ids_cpu = src_ids.cpu()
                src_ids_min = int(src_ids_cpu.min().item())
                src_ids_max = int(src_ids_cpu.max().item())
                if src_ids_max >= P_working or src_ids_min < 0:
                    src_ids_clamped = torch.clamp(src_ids, 0, P_working - 1)
                else:
                    src_ids_clamped = src_ids
            else:
                src_ids_clamped = src_ids

            if not evs_debug:
                shs_new = shs_working[src_ids_clamped].repeat_interleave(2, dim=0)
            else:
                # Debug mode: color children red/green
                shs_src = shs_working[src_ids_clamped]
                S, K, _ = shs_src.shape
                shs_new = shs_src.repeat_interleave(2, dim=0)
                red = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=shs_working.dtype)
                green = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=shs_working.dtype)
                eps = 1.0
                shs_new[0::2, 0, :] = eps * red
                shs_new[1::2, 0, :] = eps * green

            # Merge into new working set for next pass
            means3D_working = torch.cat([means_keep, means_new], dim=0)
            scales_working = torch.cat([scales_keep, scales_new], dim=0)
            rotations_working = torch.cat([rots_keep, rots_new], dim=0)
            opacity_working = torch.cat([opacity_keep, opacity_new], dim=0)
            shs_working = torch.cat([shs_keep, shs_new], dim=0)

        # Final render with updated Gaussians
        rendered_image, rendered_depth, rendered_alpha, radii = evs_rasterizer(
            means3D=means3D_working,
            means2D=None,
            shs=shs_working,
            colors_precomp=None,
            opacities=opacity_working,
            scales=scales_working,
            rotations=rotations_working,
            cov3D_precomp=None
        )

        torch.cuda.synchronize()
        end_time = time.perf_counter()
        ellapsed_time = end_time - start

        # Calculate memory: only the final working set is on GPU (no clone of original)
        cpu_final_bytes = (
            tensor_memory_bytes(means3D_working) +
            tensor_memory_bytes(scales_working) +
            tensor_memory_bytes(rotations_working) +
            tensor_memory_bytes(opacity_working) +
            tensor_memory_bytes(shs_working)
        )

        # Get CUDA peak memory
        cuda_peak_bytes = torch.cuda.max_memory_allocated() - cuda_mem_before if evs_measure_memory else 0

        evs_stats = {
            'original_gs_count': P,
            'final_gs_count': means3D_working.shape[0],
            'total_splits': total_splits,
            'net_gs_increase': means3D_working.shape[0] - P,
            'bytes_per_gs': bytes_per_gs,
            'memory_overhead_bytes': total_splits * bytes_per_gs,
            'memory_overhead_mb': total_splits * bytes_per_gs / (1024 * 1024),
            'evs_mode': 'cpu_offload',
            'gs_memory_mb': cpu_final_bytes / (1024 * 1024),  # only final working set on GPU
            'peak_memory_mb': cuda_peak_bytes / (1024 * 1024),
            'transfer_time_ms': transfer_time * 1000,
            'fps': 1.0 / ellapsed_time if ellapsed_time > 0 else 0,
        }

        # Clean up CPU tensors
        del means3D_cpu, scales_cpu, rotations_cpu, opacity_cpu, shs_cpu

        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "alpha": rendered_alpha,
            "depth": rendered_depth,
            "ellapsed_time": ellapsed_time,
            "evs_stats": evs_stats,
        }

    # -----------------------------------------
    # Naive EVS (original implementation)
    # -----------------------------------------
    # Initialize working tensors with original Gaussians (clone = extra memory!)
    means3D_working = means3D.clone()
    scales_working = scales.clone()
    rotations_working = rotations.clone()
    opacity_working = opacity.clone()
    shs_working = shs.clone()

    # Track clone memory (this is the "extra" memory from cloning)
    naive_clone_bytes = (
        tensor_memory_bytes(means3D_working) +
        tensor_memory_bytes(scales_working) +
        tensor_memory_bytes(rotations_working) +
        tensor_memory_bytes(opacity_working) +
        tensor_memory_bytes(shs_working)
    )

    total_splits = 0

    for pass_idx in range(evs_max_passes):
        # Call EVS split on current working set
        num_splits, split_flag, split_offsets, means_new, scales_new, rots_new, opacity_new, src_ids = evs_rasterizer.split(
            means3D=means3D_working,
            means2D=means2D,
            shs=shs_working,
            colors_precomp=colors_precomp,
            opacities=opacity_working,
            scales=scales_working,
            rotations=rotations_working,
            cov3D_precomp=cov3D_precomp)

        torch.cuda.synchronize()
        num_gs_split = torch.count_nonzero(split_flag).item()
        total_splits += num_gs_split

        # logger.info(f"[EVS Pass {pass_idx + 1}/{evs_max_passes}] Split {num_gs_split} Gaussians, working set: {means3D_working.shape[0]}")

        # Early termination if no or few Gaussians need splitting
        if num_gs_split <= evs_min_split_threshold:
            logger.info(f"[EVS] Early termination: only {num_gs_split} Gaussians need splitting (threshold: {evs_min_split_threshold})")
            break

        # -----------------------------------------
        # Merge: keep non-split + add children
        # -----------------------------------------
        keep_mask = (split_flag == 0)
        if opacity_new.dim() == 1:
            opacity_new = opacity_new.unsqueeze(1)

        means_keep = means3D_working[keep_mask]
        scales_keep = scales_working[keep_mask]
        rots_keep = rotations_working[keep_mask]
        opacity_keep = opacity_working[keep_mask]
        shs_keep = shs_working[keep_mask]

        # Prepare SHs for children
        P = means3D_working.shape[0]
        src_ids = src_ids.to(torch.int64)
        if num_splits > 0:
            src_ids_cpu = src_ids.cpu()
            src_ids_min = int(src_ids_cpu.min().item())
            src_ids_max = int(src_ids_cpu.max().item())
            if src_ids_max >= P or src_ids_min < 0:
                src_ids_clamped = torch.clamp(src_ids, 0, P - 1)
            else:
                src_ids_clamped = src_ids
        else:
            src_ids_clamped = src_ids

        if not evs_debug:
            shs_new = shs_working[src_ids_clamped].repeat_interleave(2, dim=0)
        else:
            # Debug mode: color children red/green
            shs_src = shs_working[src_ids_clamped]
            S, K, _ = shs_src.shape
            shs_new = shs_src.repeat_interleave(2, dim=0)
            red = torch.tensor([1.0, 0.0, 0.0], device=shs_working.device, dtype=shs_working.dtype)
            green = torch.tensor([0.0, 1.0, 0.0], device=shs_working.device, dtype=shs_working.dtype)
            eps = 1.0
            shs_new[0::2, 0, :] = eps * red
            shs_new[1::2, 0, :] = eps * green

        # Merge into new working set for next pass
        means3D_working = torch.cat([means_keep, means_new], dim=0)
        scales_working = torch.cat([scales_keep, scales_new], dim=0)
        rotations_working = torch.cat([rots_keep, rots_new], dim=0)
        opacity_working = torch.cat([opacity_keep, opacity_new], dim=0)
        shs_working = torch.cat([shs_keep, shs_new], dim=0)

    # logger.info(f"[EVS Complete] Total Gaussians split: {total_splits}, final working set: {means3D_working.shape[0]}")

    # -----------------------------------------
    # Final render with updated Gaussians
    # -----------------------------------------
    rendered_image, rendered_depth, rendered_alpha, radii = evs_rasterizer(
        means3D=means3D_working,
        means2D=None,
        shs=shs_working,
        colors_precomp=None,
        opacities=opacity_working,
        scales=scales_working,
        rotations=rotations_working,
        cov3D_precomp=None
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    # torch.cuda.synchronize()
    end_time = time.perf_counter()

    ellapsed_time=end_time-start

    # Compute EVS statistics
    # Calculate precise memory for Naive approach:
    naive_final_bytes = (
        tensor_memory_bytes(means3D_working) +
        tensor_memory_bytes(scales_working) +
        tensor_memory_bytes(rotations_working) +
        tensor_memory_bytes(opacity_working) +
        tensor_memory_bytes(shs_working)
    )

    # Calculate bytes per Gaussian dynamically based on actual tensor shapes
    P = means3D.shape[0]
    bytes_per_gs = naive_clone_bytes // P if P > 0 else 236  # fallback to 236

    # For FAIR comparison with SceneGraph:
    #
    # Naive MUST clone original data at start (this is unavoidable overhead)
    # SceneGraph does NOT clone - it references original data
    #
    # So Naive's extra memory = clone + final_working_set
    # But if no split, clone IS the final (same tensor), so extra = clone only
    #
    # With splits: clone gets replaced by torch.cat results, but the PEAK memory
    # during torch.cat includes both old tensor + new elements temporarily.
    # For simplicity, we report: clone (the unavoidable overhead) + final (the result)
    if total_splits > 0:
        # Clone is separate from final (torch.cat created new tensors)
        naive_extra_bytes = naive_clone_bytes + naive_final_bytes
    else:
        # No splits: clone IS the final (same tensor, don't double count)
        naive_extra_bytes = naive_clone_bytes

    # Get CUDA peak memory (real system measurement)
    cuda_peak_bytes = torch.cuda.max_memory_allocated() - cuda_mem_before if evs_measure_memory else 0

    evs_stats = {
        'original_gs_count': means3D.shape[0],
        'final_gs_count': means3D_working.shape[0],
        'total_splits': total_splits,
        'net_gs_increase': means3D_working.shape[0] - means3D.shape[0],
        'bytes_per_gs': bytes_per_gs,  # actual bytes per Gaussian for this scene
        'memory_overhead_bytes': total_splits * bytes_per_gs,
        'memory_overhead_mb': total_splits * bytes_per_gs / (1024 * 1024),
        'evs_mode': 'naive',
        'gs_memory_mb': naive_extra_bytes / (1024 * 1024),  # extra memory from EVS
        'peak_memory_mb': cuda_peak_bytes / (1024 * 1024),  # CUDA measured peak
        'clone_memory_mb': naive_clone_bytes / (1024 * 1024),
        'final_memory_mb': naive_final_bytes / (1024 * 1024),
        'fps': 1.0 / ellapsed_time if ellapsed_time > 0 else 0,
    }

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "alpha": rendered_alpha,
        "depth": rendered_depth,
        "ellapsed_time": ellapsed_time,
        "evs_stats": evs_stats,
    }
    
def convert_pointclouds(viewpoint_camera, pc: GaussianModel, bg_color: torch.Tensor, scaling_modifier=1.0,
           override_color=None, debug=False,ply_file_paths=None,**other_args):
    """
    We use this functions for point clouds render and orientation map rendering
    
    input: gaussians points clouds
    
    return: redendered orientation map
    """
    # basic idea, we input gaussian points and use open3d for backend render
    
    points = pc.get_xyz
    scales = pc.get_scaling
    rotations = pc.get_rotation
    R=build_rotation(rotations)
    
    i = torch.arange(scales.shape[0], device='cuda')[:, None].repeat(1, 3).view(-1)
    j = scales.argsort(dim=-1, descending=True).view(-1)
    sorted_R = R[i, j].view(-1, 3, 3)# DL: get the largest variance
    sorted_S = scales[i, j].view(-1, 3)
    _dir = sorted_R[:, 0] * sorted_S[:, 0, None]##sorted_R:torch.Size([2970000, 3, 3]),sorted_S:torch.Size([2970000, 3]),_dir:torch.Size([2970000, 3])
    
    directions = _dir / _dir.norm(dim=1, keepdim=True)  # 单位化
    
    # convert to  numpy
    points_np = points.cpu().numpy()
    directions_np = directions.cpu().numpy()

    # if ply_file_paths is not None:
    #     #take one path
    #     directory = os.path.dirname(ply_file_paths[-1])
    #     np.savez(f"{directory}\\point_data.npz", points=points_np, directions=directions_np)
    #     logger.info(f"save file to {directory}\\point_data.npz")
    return 0
