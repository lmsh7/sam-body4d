
"""
Utility functions for SAM 3D Body demo notebook
"""

import os
from typing import Any, Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import trimesh

from sam_3d_body import load_sam_3d_body_hf, SAM3DBodyEstimator
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info, mhr_names
from sam_3d_body.visualization.renderer import Renderer
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer

from utils.painter import color_list

from PIL import Image

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def setup_sam_3d_body(
    hf_repo_id: str = "facebook/sam-3d-body-vith",
    detector_name: str = "vitdet",
    segmentor_name: str = "sam2",
    fov_name: str = "moge2",
    detector_path: str = "",
    segmentor_path: str = "",
    fov_path: str = "",
    device: str = "cuda",
):
    """
    Set up SAM 3D Body estimator with optional components.

    Args:
        hf_repo_id: HuggingFace repository ID for the model
        detector_name: Name of detector to use (default: "vitdet")
        segmentor_name: Name of segmentor to use (default: "sam2")
        fov_name: Name of FOV estimator to use (default: "moge2")
        detector_path: URL or path for human detector model
        segmentor_path: Path to human segmentor model (optional)
        fov_path: path for FOV estimator
        device: Device to use (default: auto-detect cuda/cpu)

    Returns:
        estimator: SAM3DBodyEstimator instance ready for inference
    """
    print(f"Loading SAM 3D Body model from {hf_repo_id}...")

    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load core model from HuggingFace
    model, model_cfg = load_sam_3d_body_hf(hf_repo_id, device=device)

    # Initialize optional components
    human_detector, human_segmentor, fov_estimator = None, None, None

    if detector_name:
        print(f"Loading human detector from {detector_name}...")
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(name=detector_name, device=device)

    if segmentor_path:
        print(f"Loading human segmentor from {segmentor_path}...")
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=segmentor_name, device=device, path=segmentor_path
        )

    if fov_name:
        print(f"Loading FOV estimator from {fov_name}...")
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name=fov_name, device=device)

    # Create estimator wrapper
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    print(f"Setup complete!")
    print(
        f"  Human detector: {'✓' if human_detector else '✗ (will use full image or manual bbox)'}"
    )
    print(
        f"  Human segmentor: {'✓' if human_segmentor else '✗ (mask inference disabled)'}"
    )
    print(f"  FOV estimator: {'✓' if fov_estimator else '✗ (will use default FOV)'}")

    return estimator


def setup_visualizer():
    """Set up skeleton visualizer with MHR70 pose info"""
    visualizer = SkeletonVisualizer(line_width=2, radius=5)
    visualizer.set_pose_meta(mhr70_pose_info)
    return visualizer


def visualize_2d_results(
    img_cv2: np.ndarray, outputs: List[Dict[str, Any]], visualizer: SkeletonVisualizer
) -> List[np.ndarray]:
    """Visualize 2D keypoints and bounding boxes"""
    results = []

    for pid, person_output in enumerate(outputs):
        img_vis = img_cv2.copy()

        # Draw keypoints
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d_vis = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img_vis = visualizer.draw_skeleton(img_vis, keypoints_2d_vis)

        # Draw bounding box
        bbox = person_output["bbox"]
        img_vis = cv2.rectangle(
            img_vis,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 0),  # Green color
            2,
        )

        # Add person ID text
        cv2.putText(
            img_vis,
            f"Person {pid}",
            (int(bbox[0]), int(bbox[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        results.append(img_vis)

    return results


def visualize_3d_mesh(
    img_cv2: np.ndarray, outputs: List[Dict[str, Any]], faces: np.ndarray
) -> List[np.ndarray]:
    """Visualize 3D mesh overlaid on image and side view"""
    results = []

    for pid, person_output in enumerate(outputs):
        # Create renderer for this person
        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)

        # 1. Original image
        img_orig = img_cv2.copy()

        # 2. Mesh overlay on original image
        img_mesh_overlay = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                img_cv2.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        ).astype(np.uint8)

        # 3. Mesh on white background (front view)
        white_img = np.ones_like(img_cv2) * 255
        img_mesh_white = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                white_img,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        ).astype(np.uint8)

        # 4. Side view
        img_mesh_side = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                white_img.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                side_view=True,
            )
            * 255
        ).astype(np.uint8)

        # Combine all views
        combined = np.concatenate(
            [img_orig, img_mesh_overlay, img_mesh_white, img_mesh_side], axis=1
        )
        results.append(combined)

    return results


def save_mesh_results(
    outputs: List[Dict[str, Any]],
    faces: np.ndarray,
    save_dir: str,
    focal_dir: str,
    image_path: str,
    id_current: List,
    export_formats: tuple = ("ply", "glb"),
):
    """Save 3D mesh results to files and return PLY file paths"""

    if outputs is None:
        return

    for pid, person_output in enumerate(outputs):
        # Create renderer for this person
        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)

        # Store individual mesh
        color = tuple(c / 255.0 for c in color_list[id_current[pid]+4])
        tmesh = renderer.vertices_to_trimesh(
            person_output["pred_vertices"], person_output["pred_cam_t"], color
        )
        frame_stem = os.path.basename(image_path)[:-4]
        if "ply" in export_formats:
            tmesh.export(f"{save_dir}/{pid+1}/{frame_stem}.ply")
        if "glb" in export_formats:
            tmesh.export(f"{save_dir}/{pid+1}/{frame_stem}.glb", file_type="glb")

        focal_length = {'focal_length': person_output["focal_length"].item(), 'camera': [float(x) for x in person_output['pred_cam_t']]}
        with open(f"{focal_dir}/{pid+1}/{os.path.basename(image_path)[:-4]}.json", "w") as f:
            json.dump(focal_length, f, indent=4)


# 18 main body joint indices for rotation export (reduce JSON size)
_BODY_JOINT_INDICES = [
    0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 41, 62, 69,
]


def _to_list(v):
    """Convert numpy/torch value to JSON-serializable Python list or scalar."""
    if v is None:
        return None
    if isinstance(v, (np.ndarray, np.generic)):
        return v.tolist()
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy().tolist()
    return v


def save_skeleton_results(
    outputs: Optional[List[Dict[str, Any]]],
    skeleton_dir: str,
    image_path: str,
    id_current: Optional[List],
):
    """Save per-frame skeleton/joint data as JSON for sports analytics."""
    frame_name = os.path.basename(image_path)[:-4]

    if outputs is None or id_current is None:
        # Write empty stub for each known subdirectory
        for sub in sorted(os.listdir(skeleton_dir)):
            sub_path = os.path.join(skeleton_dir, sub)
            if os.path.isdir(sub_path):
                with open(os.path.join(sub_path, f"{frame_name}.json"), "w") as f:
                    json.dump({"empty": True, "frame_id": frame_name}, f)
        return

    joint_names = [n.replace("-", "_") for n in mhr_names]

    for pid, person_output in enumerate(outputs):
        # Extract rotation matrices for main body joints only
        global_rots = person_output.get("pred_global_rots")
        body_rotations = None
        if global_rots is not None:
            body_rotations = {
                joint_names[ji]: _to_list(global_rots[ji])
                for ji in _BODY_JOINT_INDICES
                if ji < len(joint_names)
            }

        skeleton_data = {
            "frame_id": frame_name,
            "joint_names": joint_names,
            "keypoints_3d": _to_list(person_output["pred_keypoints_3d"]),
            "keypoints_2d": _to_list(person_output["pred_keypoints_2d"]),
            "joint_coords": _to_list(person_output["pred_joint_coords"]),
            "body_joint_rotations": body_rotations,
            "global_rot": _to_list(person_output["global_rot"]),
            "body_pose": _to_list(person_output["body_pose_params"]),
            "hand_pose": _to_list(person_output["hand_pose_params"]),
            "shape": _to_list(person_output["shape_params"]),
            "scale": _to_list(person_output["scale_params"]),
            "camera_translation": _to_list(person_output["pred_cam_t"]),
            "focal_length": _to_list(person_output["focal_length"]),
            "bbox": _to_list(person_output["bbox"]),
        }

        out_path = f"{skeleton_dir}/{pid+1}/{frame_name}.json"
        with open(out_path, "w") as f:
            json.dump(skeleton_data, f)


def aggregate_skeleton_npz(skeleton_dir: str):
    """Aggregate per-frame skeleton JSONs into a single NPZ per person."""
    for person_sub in sorted(os.listdir(skeleton_dir)):
        person_dir = os.path.join(skeleton_dir, person_sub)
        if not os.path.isdir(person_dir):
            continue

        json_files = sorted(
            f for f in os.listdir(person_dir) if f.endswith(".json")
        )
        if not json_files:
            continue

        frames_data = []
        for jf in json_files:
            with open(os.path.join(person_dir, jf)) as f:
                frames_data.append(json.load(f))

        n = len(frames_data)
        valid_mask = np.array([not d.get("empty", False) for d in frames_data])
        frame_names = [d.get("frame_id", jf[:-5]) for d, jf in zip(frames_data, json_files)]

        # Determine shapes from max across all valid frames
        first_valid = next((d for d in frames_data if not d.get("empty", False)), None)
        if first_valid is None:
            continue

        valid_frames = [d for d in frames_data if not d.get("empty", False)]
        n_kp = max(len(d["keypoints_3d"]) for d in valid_frames)
        n_jc = max(len(d["joint_coords"]) for d in valid_frames)
        n_bp = max(len(d["body_pose"]) for d in valid_frames)
        kp3d = np.full((n, n_kp, 3), np.nan)
        kp2d = np.full((n, n_kp, 2), np.nan)
        jcoords = np.full((n, n_jc, 3), np.nan)
        global_rot = np.full((n, 3), np.nan)
        body_pose = np.full((n, n_bp), np.nan)
        cam_t = np.full((n, 3), np.nan)
        focal = np.full((n,), np.nan)

        for i, d in enumerate(frames_data):
            if d.get("empty", False):
                continue
            nk = len(d["keypoints_3d"])
            kp3d[i, :nk] = d["keypoints_3d"]
            kp2d_raw = np.array(d["keypoints_2d"])
            kp2d[i, :nk] = kp2d_raw[:, :2] if kp2d_raw.shape[1] > 2 else kp2d_raw
            njc = len(d["joint_coords"])
            jcoords[i, :njc] = d["joint_coords"]
            global_rot[i] = d["global_rot"]
            nbp = len(d["body_pose"])
            body_pose[i, :nbp] = d["body_pose"]
            cam_t[i] = d["camera_translation"]
            focal[i] = d["focal_length"]

        np.savez_compressed(
            os.path.join(person_dir, "all_frames.npz"),
            keypoints_3d=kp3d,
            keypoints_2d=kp2d,
            joint_coords=jcoords,
            global_rot=global_rot,
            body_pose=body_pose,
            camera_translation=cam_t,
            focal_length=focal,
            frame_names=np.array(frame_names),
            valid_mask=valid_mask,
        )


def _build_animated_glb(meshes: List[trimesh.Trimesh], output_path: str, fps: float):
    """Build a single animated GLB from a sequence of same-topology trimesh meshes.

    Uses glTF morph targets: frame 0 is the base mesh, frames 1..N-1 are stored
    as vertex displacement morph targets.  An animation steps through them with
    one-hot weights at the given *fps*.
    """
    import pygltflib

    n_frames = len(meshes)
    base = meshes[0]
    n_verts = len(base.vertices)
    n_morph = n_frames - 1  # morph targets count

    positions = base.vertices.astype(np.float32)
    faces_idx = base.faces.astype(np.uint32).flatten()

    # Vertex colors — RGBA uint8 → float32 VEC4 for glTF
    if base.visual and hasattr(base.visual, "vertex_colors") and base.visual.vertex_colors is not None:
        vc = np.array(base.visual.vertex_colors, dtype=np.float32)[:, :4] / 255.0
    else:
        vc = np.ones((n_verts, 4), dtype=np.float32)
    vc = vc.astype(np.float32)

    # Morph target deltas
    deltas = []
    for i in range(1, n_frames):
        d = (meshes[i].vertices - base.vertices).astype(np.float32)
        deltas.append(d)

    # ── Build binary buffer ──
    blobs = []

    def add_blob(data: bytes) -> tuple:
        """Append *data* to blob list, return (offset, length). Pad to 4-byte."""
        offset = sum(len(b) for b in blobs)
        blobs.append(data)
        pad = (4 - len(data) % 4) % 4
        if pad:
            blobs.append(b"\x00" * pad)
        return offset, len(data)

    # 0: indices
    idx_off, idx_len = add_blob(faces_idx.tobytes())
    # 1: base positions
    pos_off, pos_len = add_blob(positions.tobytes())
    # 2: vertex colors
    vc_off, vc_len = add_blob(vc.tobytes())
    # 3..3+n_morph-1: morph deltas
    morph_offsets = []
    for d in deltas:
        off, ln = add_blob(d.tobytes())
        morph_offsets.append((off, ln))

    # Animation data: time input + weights output
    times = np.linspace(0.0, (n_frames - 1) / fps, n_frames, dtype=np.float32)
    time_off, time_len = add_blob(times.tobytes())

    # Weights: n_frames rows × n_morph columns, one-hot stepping
    # Frame 0: all zeros (base), frame k (k>=1): weight[k-1]=1
    weights = np.zeros((n_frames, n_morph), dtype=np.float32)
    for k in range(1, n_frames):
        weights[k, k - 1] = 1.0
    wt_off, wt_len = add_blob(weights.tobytes())

    total_len = sum(len(b) for b in blobs)

    # ── Accessors & BufferViews ──
    buffer_views = []
    accessors = []

    def _add_view_accessor(byte_off, byte_len, comp_type, acc_type, count,
                           amin=None, amax=None, target=None):
        bv_idx = len(buffer_views)
        bv = pygltflib.BufferView(buffer=0, byteOffset=byte_off, byteLength=byte_len)
        if target is not None:
            bv.target = target
        buffer_views.append(bv)
        acc = pygltflib.Accessor(
            bufferView=bv_idx,
            componentType=comp_type,
            count=count,
            type=acc_type,
        )
        if amin is not None:
            acc.min = amin
        if amax is not None:
            acc.max = amax
        accessors.append(acc)
        return len(accessors) - 1

    # acc 0: indices
    acc_idx = _add_view_accessor(
        idx_off, idx_len, pygltflib.UNSIGNED_INT, pygltflib.SCALAR,
        len(faces_idx),
        amin=[int(faces_idx.min())], amax=[int(faces_idx.max())],
        target=pygltflib.ELEMENT_ARRAY_BUFFER,
    )
    # acc 1: base positions
    acc_pos = _add_view_accessor(
        pos_off, pos_len, pygltflib.FLOAT, pygltflib.VEC3,
        n_verts,
        amin=positions.min(axis=0).tolist(), amax=positions.max(axis=0).tolist(),
        target=pygltflib.ARRAY_BUFFER,
    )
    # acc 2: vertex colors
    acc_col = _add_view_accessor(
        vc_off, vc_len, pygltflib.FLOAT, pygltflib.VEC4,
        n_verts,
        target=pygltflib.ARRAY_BUFFER,
    )
    # acc 3..3+n_morph-1: morph target positions
    morph_acc_ids = []
    for i, (moff, mln) in enumerate(morph_offsets):
        d = deltas[i]
        mid = _add_view_accessor(
            moff, mln, pygltflib.FLOAT, pygltflib.VEC3,
            n_verts,
            amin=d.min(axis=0).tolist(), amax=d.max(axis=0).tolist(),
        )
        morph_acc_ids.append(mid)

    # acc for animation time input
    acc_time = _add_view_accessor(
        time_off, time_len, pygltflib.FLOAT, pygltflib.SCALAR,
        n_frames,
        amin=[float(times[0])], amax=[float(times[-1])],
    )
    # acc for animation weights output
    acc_weights = _add_view_accessor(
        wt_off, wt_len, pygltflib.FLOAT, pygltflib.SCALAR,
        n_frames * n_morph,
    )

    # ── Mesh with morph targets ──
    targets = [pygltflib.Attributes(POSITION=mid) for mid in morph_acc_ids]

    mesh = pygltflib.Mesh(
        primitives=[
            pygltflib.Primitive(
                attributes=pygltflib.Attributes(POSITION=acc_pos, COLOR_0=acc_col),
                indices=acc_idx,
                targets=targets,
            )
        ],
        weights=[0.0] * n_morph,
    )

    # ── Animation ──
    animation = pygltflib.Animation(
        samplers=[
            pygltflib.AnimationSampler(
                input=acc_time,
                output=acc_weights,
                interpolation="STEP",
            )
        ],
        channels=[
            pygltflib.AnimationChannel(
                sampler=0,
                target=pygltflib.AnimationChannelTarget(
                    node=0,
                    path="weights",
                ),
            )
        ],
    )

    # ── Assemble GLTF2 ──
    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(mesh=0)],
        meshes=[mesh],
        accessors=accessors,
        bufferViews=buffer_views,
        buffers=[pygltflib.Buffer(byteLength=total_len)],
        animations=[animation],
    )
    gltf.set_binary_blob(b"".join(blobs))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gltf.save(output_path)
    print(f"[OK] Animated GLB ({n_frames} frames, {n_morph} morph targets): {output_path}")


def aggregate_mesh_glb(mesh_dir: str, fps: float = 30.0):
    """Aggregate per-frame mesh files into a single animated GLB per person."""
    for person_sub in sorted(os.listdir(mesh_dir)):
        person_path = os.path.join(mesh_dir, person_sub)
        if not os.path.isdir(person_path):
            continue

        # Prefer PLY; fall back to per-frame GLB (skip animated.glb itself)
        mesh_files = sorted(f for f in os.listdir(person_path) if f.endswith(".ply"))
        if not mesh_files:
            mesh_files = sorted(
                f for f in os.listdir(person_path)
                if f.endswith(".glb") and f != "animated.glb"
            )
        if len(mesh_files) < 2:
            continue

        meshes = [trimesh.load(os.path.join(person_path, f), force="mesh") for f in mesh_files]
        output_path = os.path.join(person_path, "animated.glb")
        _build_animated_glb(meshes, output_path, fps)


def display_results_grid(
    images: List[np.ndarray], titles: List[str], figsize_per_image: tuple = (6, 6)
):
    """Display multiple images in a grid"""
    n_images = len(images)
    if n_images == 0:
        print("No images to display")
        return

    # Calculate grid dimensions
    cols = min(3, n_images)  # Max 3 columns
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(figsize_per_image[0] * cols, figsize_per_image[1] * rows)
    )

    # Handle single image case
    if n_images == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = axes.flatten()

    for i, (img, title) in enumerate(zip(images, titles)):
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Convert BGR to RGB if needed
            if img.dtype == np.uint8 and np.mean(img) > 1:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img
        else:
            img_rgb = img

        axes[i].imshow(img_rgb)
        axes[i].set_title(title)
        axes[i].axis("off")

    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def process_image_with_mask(estimator, image_path: str, mask_path: str, idx_path, idx_dict, mhr_shape_scale_dict, occ_dict, batch_kps=None, kps_id=None, cam_int=None, iou_dict=None, predictor=None, inference_type="full"):
    """
    Process image with external mask input.

    When all frames are non-occluded, batches all people together for a single
    process_frames call (backbone runs once instead of once per person).
    Falls back to per-person processing when occlusion completion is active.
    """
    n_frames = len(image_path)
    obj_ids = sorted(map(int, occ_dict.keys()))

    # Check if any frame has occlusion for any person
    all_no_occ = all(
        occ_dict[oid][i] == 1
        for oid in obj_ids
        for i in range(n_frames)
    )

    if not all_no_occ:
        return _process_image_with_mask_per_person(
            estimator, image_path, mask_path, idx_path, idx_dict,
            mhr_shape_scale_dict, occ_dict, batch_kps=batch_kps,
            kps_id=kps_id, cam_int=cam_int, iou_dict=iou_dict,
            predictor=predictor, inference_type=inference_type,
        )

    # ---- Batched path: aggregate all people per frame ----
    image_batch = []
    bbox_batch = []
    mask_batch = []
    id_batch = []
    kps_batch_list = []
    empty_frame_list = []

    for i in range(n_frames):
        mask_img = np.array(Image.open(mask_path[i]).convert('P'))
        H, W = mask_img.shape

        frame_bboxes = []
        frame_masks = []
        frame_ids = []
        frame_kps = []

        for obj_id in obj_ids:
            mask_binary = np.zeros_like(mask_img, dtype=np.uint8)
            mask_binary[mask_img == obj_id] = 255

            # Mute objects near image margin
            mask_cp = mask_binary.copy()
            margin_h, margin_w = int(H * 0.05), int(W * 0.05)
            mask_cp[:margin_h, :] = mask_cp[-margin_h:, :] = 0
            mask_cp[:, :margin_w] = mask_cp[:, -margin_w:] = 0
            if mask_cp.max() == 0:
                mask_binary = mask_cp

            if mask_binary.max() > 0:
                coords = cv2.findNonZero(mask_binary)
                x, y, w, h = cv2.boundingRect(coords)
                frame_bboxes.append(np.array([[x, y, x + w, y + h]], dtype=np.float32))
                frame_masks.append(mask_binary)
                frame_ids.append(obj_id)
                if batch_kps is not None:
                    frame_kps.append(batch_kps[obj_id - 1][i])
            # else: invalid mask — process_frames padding will fill this slot

        if len(frame_ids) == 0:
            empty_frame_list.append(i)
            continue

        image_batch.append(image_path[i])
        bbox_batch.append(np.stack(frame_bboxes, axis=0).squeeze(axis=1))  # (N, 4)
        mask_batch.append(np.stack(frame_masks, axis=0))              # (N, H, W)
        id_batch.append(frame_ids)
        if batch_kps is not None:
            kps_batch_list.append(np.stack(frame_kps, axis=0))

    if len(empty_frame_list) > 0:
        for oid in obj_ids:
            for i in sorted(empty_frame_list, reverse=True):
                occ_dict[oid].pop(i)

    if batch_kps is None:
        kps_batch_list = None

    all_outputs = estimator.process_frames(
        image_batch, bboxes=bbox_batch, masks=mask_batch,
        id_batch=id_batch, idx_path={}, idx_dict={},
        mhr_shape_scale_dict=mhr_shape_scale_dict,
        kps_batch=kps_batch_list, occ_dict=None,
        use_mask=True, kps_id=kps_id, cam_int=cam_int,
        inference_type=inference_type,
    )

    # Reconstruct output with empty frames inserted back
    final_outputs = []
    final_ids = []
    out_idx = 0
    for i in range(n_frames):
        if i in empty_frame_list:
            final_outputs.append([])
            final_ids.append([])
        else:
            final_outputs.append(all_outputs[out_idx])
            final_ids.append(id_batch[out_idx])
            out_idx += 1

    return final_outputs, final_ids, empty_frame_list


def _process_image_with_mask_per_person(estimator, image_path, mask_path, idx_path, idx_dict, mhr_shape_scale_dict, occ_dict, batch_kps=None, kps_id=None, cam_int=None, iou_dict=None, predictor=None, inference_type="full"):
    """
    Original per-person processing path. Used when occlusion completion is
    active and different people may use different reference images per frame.
    """
    n_frames = len(image_path)
    obj_ids = sorted(map(int, occ_dict.keys()))
    empty_dict = {}
    id_batch = []
    mask_outputs_dict = {}

    # infer per object
    for obj_id in obj_ids:
        # prepare data (HMR for non-occ first, followed by occ)
        # load in batches
        no_occ_image_batch = []
        no_occ_bbox_batch = []
        no_occ_mask_batch = []
        no_occ_kps_batch = []
        no_occ_id_batch = []
        no_occ_empty_frame_list = []
        _occ_image_batch = []
        _occ_bbox_batch = []
        _occ_mask_batch = []
        _occ_kps_batch = []
        _occ_image_batch_ori = []
        _occ_id_batch = []
        _occ_empty_frame_list = []

        occ_idx = occ_dict[obj_id]
        # for each frame:
        for i in range(n_frames):
            # Load mask
            mask = np.array(Image.open(mask_path[i]).convert('P'))

            no_occ_mask_list = []
            no_occ_bbox_list = []
            no_occ_kp_list = []
            no_occ_id_current = []
            _occ_mask_list = []
            _occ_bbox_list = []
            _occ_kp_list = []
            _occ_id_current = []

            if occ_idx[i] == 0:
                if kps_id is not None:
                    mask_com = np.array(Image.open(os.path.join(idx_path[obj_id]['masks'], f"{kps_id[0]:08d}.png")).convert('P'))
                else:
                    mask_com = np.array(Image.open(os.path.join(idx_path[obj_id]['masks'], f"{i:08d}.png")).convert('P'))
                zero_mask = np.zeros_like(mask_com)
                zero_mask[mask_com==obj_id] = 255
                mask_binary = zero_mask.astype(np.uint8)
                _occ_mask_list.append(mask_binary)
                # Compute bounding box from mask (required by refactored code)
                # Find all non-zero pixels in the mask
                coords = cv2.findNonZero(mask_binary)
                if mask_binary.max() > 0:
                    _occ_id_current.append(obj_id)
                # Get bounding box from mask contours
                x, y, w, h = cv2.boundingRect(coords)
                bbox = np.array([[x, y, x + w, y + h]], dtype=np.float32)
                # print(f"Computed bbox from mask: {bbox[0]}")
                _occ_bbox_list.append(bbox)
                if batch_kps is not None:
                    _occ_kp_list.append(batch_kps[obj_id-1][i])  # N x 3

                if len(_occ_bbox_list) == 0:
                    _occ_empty_frame_list.append(i)
                else:
                    _occ_id_batch.append(_occ_id_current)
                    bbox = np.stack(_occ_bbox_list, axis=0)  # TODO: sometimes empty
                    if batch_kps is not None:
                        _occ_kps_batch.append(np.stack(_occ_kp_list, axis=0))
                    mask_binary = np.stack(_occ_mask_list, axis=0)
                    # Process with external mask and computed bbox
                    # Note: The mask needs to match the number of bboxes (1 bbox -> 1 mask)
                    _occ_image_batch.append(os.path.join(idx_path[obj_id]['images'], f"{i:08d}.jpg"))
                    _occ_image_batch_ori.append(image_path[i])
                    _occ_mask_batch.append(mask_binary)
                    _occ_bbox_batch.append(bbox)
            else:
                zero_mask = np.zeros_like(mask)
                zero_mask[mask==obj_id] = 255
                mask_binary = zero_mask.astype(np.uint8)

                # mute objects near margin
                H, W = mask_binary.shape
                zero_mask_cp = np.zeros_like(mask)
                zero_mask_cp[mask==obj_id] = 255
                mask_binary_cp = zero_mask_cp.astype(np.uint8)
                mask_binary_cp[:int(H*0.05), :] = mask_binary_cp[-int(H*0.05):, :] = mask_binary_cp[:, :int(W*0.05)] = mask_binary_cp[:, -int(W*0.05):] = 0
                if mask_binary_cp.max() == 0:   # margin objects
                    mask_binary = mask_binary_cp

                no_occ_mask_list.append(mask_binary)
                # Compute bounding box from mask (required by refactored code)
                # Find all non-zero pixels in the mask
                coords = cv2.findNonZero(mask_binary)

                if mask_binary.max() > 0:
                    no_occ_id_current.append(obj_id)

                # Get bounding box from mask contours
                x, y, w, h = cv2.boundingRect(coords)
                bbox = np.array([[x, y, x + w, y + h]], dtype=np.float32)

                # print(f"Computed bbox from mask: {bbox[0]}")
                no_occ_bbox_list.append(bbox)
                if batch_kps is not None:
                    no_occ_kp_list.append(batch_kps[obj_id-1][i])  # N x 3

                if len(no_occ_bbox_list) == 0:
                    no_occ_empty_frame_list.append(i)
                else:
                    no_occ_id_batch.append(no_occ_id_current)
                    bbox = np.stack(no_occ_bbox_list, axis=0)  # TODO: sometimes empty
                    if batch_kps is not None:
                        no_occ_kps_batch.append(np.stack(no_occ_kp_list, axis=0))
                    mask_binary = np.stack(no_occ_mask_list, axis=0)
                    # Process with external mask and computed bbox
                    # Note: The mask needs to match the number of bboxes (1 bbox -> 1 mask)
                    no_occ_image_batch.append(image_path[i])
                    no_occ_mask_batch.append(mask_binary)
                    no_occ_bbox_batch.append(bbox)

        if len(no_occ_empty_frame_list) > 0:
            for occ_k, occ_v in occ_dict.items():
                for i in sorted(no_occ_empty_frame_list, reverse=True):
                    occ_v.pop(i)
        if len(_occ_empty_frame_list) > 0:
            for occ_k, occ_v in occ_dict.items():
                for i in sorted(_occ_empty_frame_list, reverse=True):
                    occ_v.pop(i)

        empty_dict[f"{obj_id}-occ"] = _occ_empty_frame_list
        empty_dict[f"{obj_id}-no_occ"] = no_occ_empty_frame_list

        if batch_kps is None:
            no_occ_kps_batch = None
            _occ_kps_batch = None

        if len(no_occ_image_batch) > 0:
            no_occ_outputs = estimator.process_frames(no_occ_image_batch, bboxes=no_occ_bbox_batch, masks=no_occ_mask_batch, id_batch=[[1] for idb in range(len(no_occ_image_batch))], idx_path={}, idx_dict={}, mhr_shape_scale_dict=mhr_shape_scale_dict, kps_batch=no_occ_kps_batch, occ_dict=None, use_mask=True, kps_id=kps_id, cam_int=cam_int, inference_type=inference_type)
        if len(_occ_image_batch) > 0:
            _occ_outputs = estimator.process_frames(_occ_image_batch, bboxes=_occ_bbox_batch, masks=_occ_mask_batch, id_batch=[[1] for idb in range(len(_occ_image_batch))], idx_path={}, idx_dict={}, mhr_shape_scale_dict=mhr_shape_scale_dict, kps_batch=_occ_kps_batch, occ_dict=None, use_mask=True, kps_id=kps_id, _occ_image_batch_ori=_occ_image_batch_ori, cam_int=cam_int, inference_type=inference_type)

        oid_outputs = []
        ia, ib = 0, 0
        for oi in occ_idx:
            if oi == 1:
                oid_outputs.append(no_occ_outputs[ia])
                ia += 1
            else:
                oid_outputs.append(_occ_outputs[ib])
                ib += 1

        mask_outputs_dict[obj_id] = oid_outputs

    final_outputs = []
    id_batch = []
    empty_frame_list = []
    for i in range(n_frames):
        i_outputs = []
        i_batch = []
        for obj_id in obj_ids:  # sorted
            if mask_outputs_dict[obj_id][i][0]['bbox'][0]+mask_outputs_dict[obj_id][i][0]['bbox'][2] > 0:   # 0 0 0 0 (no objects)
                i_outputs.append(mask_outputs_dict[obj_id][i][0])   # always = 1
                i_batch.append(obj_id)

        final_outputs.append(i_outputs)
        id_batch.append(i_batch)
        if len(i_batch) == 0:
            empty_frame_list.append(i)

    return final_outputs, id_batch, empty_frame_list


def process_image_with_bbox(estimator, image_path: str, bboxes, idx_path, idx_dict, mhr_shape_scale_dict, occ_dict, batch_kps=None, flip=False, cam_int=None):
    """
    Process image with external mask input.

    Note: The refactored code requires bboxes to be provided along with masks.
    This function automatically computes bboxes from the mask.
    """
    # load in batches
    image_batch = []
    bbox_batch = []
    kps_batch = []
    n = len(image_path)
    id_batch = []
    empty_frame_list = []
    obj_ids = [oi+1 for oi in range(len(bboxes))]
    for i in range(n):
        bbox_list = []
        kp_list = []
        id_current = []
        
        for obj_id in obj_ids:
            id_current.append(obj_id)
            # Get bounding box from mask contours
            x, y, x2, y2 = bboxes[obj_id-1][i][0].item(), bboxes[obj_id-1][i][1].item(), bboxes[obj_id-1][i][2].item(), bboxes[obj_id-1][i][3].item()
            bbox = np.array([[x, y, x2, y2]], dtype=np.float32)
            # print(f"Computed bbox from mask: {bbox[0]}")
            bbox_list.append(bbox)
            if batch_kps is not None:
                kp_list.append(batch_kps[obj_id-1][i])  # N x 3

        if len(bbox_list) == 0:
            empty_frame_list.append(i)
            continue

        id_batch.append(id_current)
        bbox = np.stack(bbox_list, axis=0)  # TODO: sometimes empty
        if batch_kps is not None:
            kps_batch.append(np.stack(kp_list, axis=0))
        # mask_binary = np.stack(mask_list, axis=0)
        # Process with external mask and computed bbox
        # Note: The mask needs to match the number of bboxes (1 bbox -> 1 mask)
        image_batch.append(image_path[i])
        bbox_batch.append(bbox)
    
    if len(empty_frame_list) > 0:
        for occ_k, occ_v in occ_dict.items():
            for i in sorted(empty_frame_list, reverse=True):
                occ_v.pop(i)

    outputs = estimator.process_frames(image_batch, bboxes=bbox_batch, masks=None, id_batch=id_batch, idx_path=idx_path, idx_dict=idx_dict, mhr_shape_scale_dict=mhr_shape_scale_dict, occ_dict=occ_dict, kps_batch=kps_batch, flip=flip, cam_int=cam_int)   # use_mask=False default

    return outputs, id_batch, empty_frame_list
