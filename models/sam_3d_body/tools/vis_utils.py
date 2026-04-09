# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import torch
from sam_3d_body.visualization.renderer import Renderer
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
from utils.painter import color_list

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

visualizer = SkeletonVisualizer(line_width=2, radius=5)
visualizer.set_pose_meta(mhr70_pose_info)


def visualize_sample(img_cv2, outputs, faces, id_current):
	img_mesh = img_cv2.copy()
	img_mesh = np.ones_like(img_mesh) * 255

	if outputs is None:
		return img_mesh

	rend_img = []
	for pid, person_output in enumerate(outputs):
		renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)
		img2 = (
			renderer(
				person_output["pred_vertices"],
				person_output["pred_cam_t"],
				img_mesh.copy(),
				mesh_base_color=color_list[id_current[pid]+4],
				scene_bg_color=(1, 1, 1),
			)
			* 255
		)

		# cur_img = np.concatenate([img_cv2, img1, img2, img3], axis=1)
		rend_img.append(img2)

	return rend_img

def visualize_sample_together(img_cv2, outputs, faces, id_current):
	# Render everything together
	img_mesh = img_cv2.copy()
	img_mesh = np.ones_like(img_mesh) * 255

	if outputs is None:
		return img_mesh

	# First, sort by depth, furthest to closest
	try:
		all_depths = np.stack([tmp['pred_cam_t'] for tmp in outputs], axis=0)[:, 2]
	except:
		return img_mesh
	outputs_sorted = [outputs[idx] for idx in np.argsort(-all_depths)]

	id_sorted = np.argsort(-all_depths)   # by id not depth for consistent coloring

	# Then, put all meshes together as one super mesh
	all_pred_vertices = []
	all_faces = []
	all_color = []
	for pid, person_output in enumerate(outputs_sorted):
		all_pred_vertices.append(person_output["pred_vertices"] + person_output["pred_cam_t"])
		all_faces.append(faces + len(person_output["pred_vertices"]) * pid)
		all_color.append(color_list[id_current[id_sorted[pid]]+4])
	all_pred_vertices = np.concatenate(all_pred_vertices, axis=0)
	all_faces = np.concatenate(all_faces, axis=0)

	# Pull out a fake translation; take the closest two
	fake_pred_cam_t = (np.max(all_pred_vertices[-2*18439:], axis=0) + np.min(all_pred_vertices[-2*18439:], axis=0)) / 2
	all_pred_vertices = all_pred_vertices - fake_pred_cam_t
	
	# Render front view
	renderer = Renderer(focal_length=person_output["focal_length"], faces=all_faces)
	img_mesh = (
		renderer(
			all_pred_vertices,
			fake_pred_cam_t,
			img_mesh,
			# mesh_base_color=LIGHT_BLUE,
			mesh_base_color=all_color,
			scene_bg_color=(1, 1, 1),
		)
		* 255
	)

	return img_mesh


# ======================================================================
# PyTorch3D batch rendering helpers
# ======================================================================

_SMPL_V = 18439  # fixed SMPL vertex count per person


def _prepare_combined_mesh(
    outputs: List[Dict],
    faces: np.ndarray,
    id_current: List[int],
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, np.ndarray]]:
    """Build a depth-sorted super-mesh for one frame (same logic as
    ``visualize_sample_together``) and return GPU tensors.

    Returns (verts, faces_t, colors, focal_length, cam_t_np) or None.
    """
    try:
        all_depths = np.stack([o["pred_cam_t"] for o in outputs], axis=0)[:, 2]
    except Exception:
        return None

    sorted_idx = np.argsort(-all_depths)
    outputs_sorted = [outputs[i] for i in sorted_idx]

    all_verts, all_faces, all_colors = [], [], []
    vert_offset = 0
    for pid, person_output in enumerate(outputs_sorted):
        v = person_output["pred_vertices"] + person_output["pred_cam_t"]
        f = faces + vert_offset
        # Per-person color from palette (same indexing as original)
        c_rgb = np.array(color_list[id_current[sorted_idx[pid]] + 4], dtype=np.float32) / 255.0
        c = np.broadcast_to(c_rgb, (v.shape[0], 3)).copy()
        all_verts.append(v)
        all_faces.append(f)
        all_colors.append(c)
        vert_offset += v.shape[0]

    verts_np = np.concatenate(all_verts, axis=0)
    faces_np = np.concatenate(all_faces, axis=0)
    colors_np = np.concatenate(all_colors, axis=0)

    # Fake camera translation (same as visualize_sample_together)
    last_2 = verts_np[-2 * _SMPL_V:]
    fake_cam_t = (np.max(last_2, axis=0) + np.min(last_2, axis=0)) / 2.0
    verts_np = verts_np - fake_cam_t

    # Flip Y to convert from HMR convention (Y-up) to PyTorch3D screen
    # convention where y_screen = -f*Y/Z + cy (positive Y → screen top).
    # Without this flip the rendering is upside-down.
    verts_np[:, 1] *= -1.0

    cam_t = fake_cam_t.copy()
    cam_t[1] *= -1.0  # flip cam_t Y to match vertex flip

    focal = float(outputs[-1]["focal_length"])

    verts_t = torch.tensor(verts_np, dtype=torch.float32, device=device)
    faces_t = torch.tensor(faces_np.astype(np.int64), dtype=torch.int64, device=device)
    colors_t = torch.tensor(colors_np, dtype=torch.float32, device=device)

    return verts_t, faces_t, colors_t, focal, cam_t


def batch_render_combined(
    images: List[np.ndarray],
    outputs_list: List[Optional[List[Dict]]],
    faces: np.ndarray,
    id_current_list: List[Optional[List[int]]],
    renderer,  # PyTorch3DBatchRenderer
) -> List[np.ndarray]:
    """Batch-render the combined super-mesh for *B* frames at once.

    Returns list of *B* BGR uint8 images.
    """
    device = renderer.device
    B = len(images)
    H, W = images[0].shape[:2]

    verts_list, faces_list_t, colors_list_t = [], [], []
    focal_list, cam_t_list = [], []
    valid_indices = []

    for idx in range(B):
        outputs = outputs_list[idx]
        id_current = id_current_list[idx]
        if outputs is None or id_current is None:
            continue
        result = _prepare_combined_mesh(outputs, faces, id_current, device)
        if result is None:
            continue
        verts_t, faces_t, colors_t, focal, cam_t = result
        verts_list.append(verts_t)
        faces_list_t.append(faces_t)
        colors_list_t.append(colors_t)
        focal_list.append(focal)
        cam_t_list.append(cam_t)
        valid_indices.append(idx)

    # White images for empty frames
    white = np.ones((H, W, 3), dtype=np.uint8) * 255
    result_images = [white.copy() for _ in range(B)]

    if len(verts_list) == 0:
        return result_images

    focal_t = torch.tensor(focal_list, dtype=torch.float32, device=device)
    cam_t_t = torch.tensor(np.stack(cam_t_list), dtype=torch.float32, device=device)

    rendered = renderer.render_batch(
        verts_list=verts_list,
        faces_list=faces_list_t,
        colors_list=colors_list_t,
        focal_lengths=focal_t,
        cam_translations=cam_t_t,
        image_size=(H, W),
        bg_images=None,  # white bg
    )

    # Convert to numpy uint8 BGR
    rendered_np = (rendered.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    for i, idx in enumerate(valid_indices):
        # PyTorch3D outputs RGB; OpenCV needs BGR
        result_images[idx] = rendered_np[i, :, :, ::-1].copy()

    return result_images


def batch_render_individual(
    images: List[np.ndarray],
    outputs_list: List[Optional[List[Dict]]],
    faces: np.ndarray,
    id_current_list: List[Optional[List[int]]],
    renderer,  # PyTorch3DBatchRenderer
) -> List[List[np.ndarray]]:
    """Batch-render per-person individual meshes for *B* frames.

    Returns list of *B* lists, each containing per-person BGR uint8 images.
    """
    device = renderer.device
    B = len(images)
    H, W = images[0].shape[:2]

    faces_t = torch.tensor(faces.astype(np.int64), dtype=torch.int64, device=device)

    # Collect all (frame_idx, person_idx) pairs
    verts_list, faces_list_t, colors_list_t = [], [], []
    focal_list, cam_t_list = [], []
    index_map = []  # (frame_idx, person_idx)

    for frame_idx in range(B):
        outputs = outputs_list[frame_idx]
        id_current = id_current_list[frame_idx]
        if outputs is None or id_current is None:
            continue
        for pid, person_output in enumerate(outputs):
            v = person_output["pred_vertices"].copy()
            v[:, 1] *= -1.0  # flip Y for PyTorch3D screen convention
            cam_t = person_output["pred_cam_t"].copy()
            cam_t[1] *= -1.0  # match vertex Y flip

            c_rgb = np.array(color_list[id_current[pid] + 4], dtype=np.float32) / 255.0
            c = np.broadcast_to(c_rgb, (v.shape[0], 3)).copy()

            verts_list.append(torch.tensor(v, dtype=torch.float32, device=device))
            faces_list_t.append(faces_t)
            colors_list_t.append(torch.tensor(c, dtype=torch.float32, device=device))
            focal_list.append(float(person_output["focal_length"]))
            cam_t_list.append(cam_t)
            index_map.append((frame_idx, pid))

    # Initialize result structure
    per_frame_results = []
    for frame_idx in range(B):
        outputs = outputs_list[frame_idx]
        if outputs is None:
            per_frame_results.append([])
        else:
            white = np.ones((H, W, 3), dtype=np.uint8) * 255
            per_frame_results.append([white.copy() for _ in outputs])

    if len(verts_list) == 0:
        return per_frame_results

    focal_t = torch.tensor(focal_list, dtype=torch.float32, device=device)
    cam_t_t = torch.tensor(np.stack(cam_t_list), dtype=torch.float32, device=device)

    rendered = renderer.render_batch(
        verts_list=verts_list,
        faces_list=faces_list_t,
        colors_list=colors_list_t,
        focal_lengths=focal_t,
        cam_translations=cam_t_t,
        image_size=(H, W),
        bg_images=None,
    )

    rendered_np = (rendered.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    for i, (frame_idx, pid) in enumerate(index_map):
        per_frame_results[frame_idx][pid] = rendered_np[i, :, :, ::-1].copy()

    return per_frame_results
