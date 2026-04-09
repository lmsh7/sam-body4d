import os
ROOT = os.path.dirname(__file__)
os.environ["GRADIO_TEMP_DIR"] = os.path.join(ROOT, "gradio_tmp")

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'models', 'sam_3d_body'))
sys.path.append(os.path.join(current_dir, 'models', 'diffusion_vas'))

import uuid
from datetime import datetime

def gen_id():
    t = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    u = uuid.uuid4().hex[:8]
    return f"{t}_{u}"

import time
import cv2
import glob
import random
import gradio as gr
import numpy as np
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

from utils import draw_point_marker, mask_painter, images_to_mp4, DAVIS_PALETTE, jpg_folder_to_mp4, is_super_long_or_wide, keep_largest_component, is_skinny_mask, bbox_from_mask, gpu_profile, resize_mask_with_unique_label

from models.sam_3d_body.sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from models.sam_3d_body.notebook.utils import process_image_with_mask, save_mesh_results, save_skeleton_results, aggregate_skeleton_npz, aggregate_mesh_glb
from models.sam_3d_body.tools.vis_utils import visualize_sample_together, visualize_sample, batch_render_combined, batch_render_individual
from models.diffusion_vas.demo import init_amodal_segmentation_model, init_rgb_model, init_depth_model, load_and_transform_masks, load_and_transform_rgbs, rgb_to_depth

import torch
import concurrent.futures

_VIS_WORKERS = int(os.environ.get("SAM_VIS_WORKERS", "4"))
# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")
if device.type == "cuda":
    # use bfloat16 for the entire notebook
    # torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 3 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

# ===============================
# Global runtime objects
# ===============================
CONFIG = None           # loaded config
sam3_model = None       # your SAM-3 model instance
sam3_image_model = None
image_predictor = None
predictor = None        # sam-3 predictor
inference_state = None  # sam-3 inference_state
RUNTIME = {}            # global dict to store runtime data per video/run


def build_sam3_from_config(cfg):
    """
    Construct and return your SAM-3 model from config.
    You replace this with your real init code.
    """
    from models.sam3.sam3.model_builder import build_sam3_video_model

    sam3_model = build_sam3_video_model(checkpoint_path=cfg.sam3['ckpt_path'])
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone

    return sam3_model, predictor


def build_sam3_3d_body_config(cfg, device=None):
    mhr_path = cfg.sam_3d_body['mhr_path']
    fov_path = cfg.sam_3d_body['fov_path']
    fast_cfg = cfg.sam_3d_body.get('fast_mode', None)
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_sam_3d_body(
        cfg.sam_3d_body['ckpt_path'], device=device, mhr_path=mhr_path, fast_cfg=fast_cfg
    )

    human_detector, human_segmentor, fov_estimator = None, None, None
    from models.sam_3d_body.tools.build_fov_estimator import FOVEstimator
    fov_estimator = FOVEstimator(name='moge2', device=device, path=fov_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    return estimator


def _assign_gpu_devices(num_workers):
    """Assign GPU devices for multi-GPU pipeline.

    All GPUs host diffusion workers. Depth model and HMR share GPU 0 and
    GPU (n-1) respectively — they run before/after diffusion, never concurrently,
    so sharing is safe and maximizes diffusion parallelism.

    GPU layout (4 GPUs, 2 obj_ids):
        GPU 0: diffusion worker 0 + depth_model (sequential, not concurrent)
        GPU 1: diffusion worker 1
        GPU 2: (idle for 2 objs, used when obj_ids >= 3)
        GPU 3: (idle for 2 objs) + SAM 3D Body (HMR) after diffusion completes
    """
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus >= 2:
        depth_device = torch.device("cuda:0")
        worker_devices = [torch.device(f"cuda:{i % n_gpus}") for i in range(num_workers)]
        hmr_device = torch.device(f"cuda:{n_gpus - 1}")
    else:
        depth_device = torch.device("cuda") if n_gpus > 0 else torch.device("cpu")
        worker_devices = [depth_device] * num_workers
        hmr_device = depth_device
    return depth_device, worker_devices, hmr_device


def build_diffusion_vas_config(cfg, num_workers=2):
    model_path_mask = cfg.completion['model_path_mask']
    model_path_rgb = cfg.completion['model_path_rgb']
    depth_encoder = cfg.completion['depth_encoder']
    model_path_depth = cfg.completion['model_path_depth']
    max_occ_len = min(cfg.completion['max_occ_len'], cfg.sam_3d_body['batch_size'])

    depth_device, worker_devices, hmr_device = _assign_gpu_devices(num_workers)

    # Create independent diffusion workers on different GPUs
    diffusion_workers = []
    for i, w_dev in enumerate(worker_devices):
        dev_str = str(w_dev)
        print(f"[INIT] Creating diffusion worker {i} on {dev_str}")
        pm = init_amodal_segmentation_model(model_path_mask, device=dev_str)
        pr = init_rgb_model(model_path_rgb, device=dev_str)
        gen = torch.Generator(device='cpu').manual_seed(23)
        diffusion_workers.append({
            'pipeline_mask': pm,
            'pipeline_rgb': pr,
            'generator': gen,
            'device': w_dev,
        })

    depth_model = init_depth_model(model_path_depth, depth_encoder, device=str(depth_device))

    return diffusion_workers, depth_model, max_occ_len, depth_device, hmr_device


def init_runtime(config_path: str = os.path.join(ROOT, "configs", "body4d.yaml")):
    """Initialize CONFIG, SAM3_MODEL, and global RUNTIME dict."""
    global CONFIG, sam3_model, predictor, inference_state, sam3_3d_body_model, RUNTIME, OUTPUT_DIR, \
        diffusion_workers, depth_model, max_occ_len
    CONFIG = OmegaConf.load(config_path)
    sam3_model, predictor = build_sam3_from_config(CONFIG)

    if CONFIG.completion.get('enable', False):
        diffusion_workers, depth_model, max_occ_len, depth_device, hmr_device = build_diffusion_vas_config(CONFIG)
    else:
        diffusion_workers, depth_model, max_occ_len = [], None, None
        hmr_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    sam3_3d_body_model = build_sam3_3d_body_config(CONFIG, device=hmr_device)

    OUTPUT_DIR = os.path.join(CONFIG.runtime['output_dir'], gen_id())
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    RUNTIME = {}  # clear any old state
    RUNTIME['batch_size'] = CONFIG.sam_3d_body.get('batch_size', 1)
    RUNTIME['detection_resolution'] = CONFIG.completion.get('detection_resolution', [256, 512])
    RUNTIME['completion_resolution'] = CONFIG.completion.get('completion_resolution', [512, 1024])
    RUNTIME['smpl_export'] = CONFIG.runtime.get('smpl_export', False)

# ===============================
# Paths & supported formats
# ===============================

EXAMPLE_1 = os.path.join(ROOT, "assets", "examples", "example1.mp4")
EXAMPLE_2 = os.path.join(ROOT, "assets", "examples", "example2.mp4")
EXAMPLE_3 = os.path.join(ROOT, "assets", "examples", "example3.mp4")

SUPPORTED_EXTS = {".mp4",}

# ===============================
# Video utilities
# ===============================

def read_video_metadata(path: str):
    """Return FPS and total frame count."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    RUNTIME['video_fps'] = fps
    return fps, total


def read_frame_at(path: str, idx: int):
    """Read a specific frame (by index) from a video file."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def get_thumb(path: str):
    """Return the first frame of a video as thumbnail, or None if missing."""
    if not os.path.exists(path):
        return None
    return read_frame_at(path, 0)


EX1_THUMB = get_thumb(EXAMPLE_1)
EX2_THUMB = get_thumb(EXAMPLE_2)
EX3_THUMB = get_thumb(EXAMPLE_3)


# ===============================
# Prepare video
# ===============================

def prepare_video(path: str):
    """
    Common helper:
    - validate path & extension
    - read fps & total frames
    - read first frame
    - init slider & time text
    """
    if path is None:
        return (
            None,  # video_state
            1.0,   # fps_state
            None,  # current_frame
            gr.update(minimum=0, maximum=0, value=0),  # frame_slider
            "00:00 / 00:00",  # time_text
        )

    if not os.path.exists(path):
        raise gr.Error(f"Video not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise gr.Error(
            f"Unsupported video format {ext}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTS))}"
        )

    fps, total = read_video_metadata(path)
    if fps <= 0 or total <= 0:
        raise gr.Error("Invalid video metadata (fps or frame count).")

    first_frame = read_frame_at(path, 0)
    if first_frame is None:
        raise gr.Error("Failed to read first frame.")

    slider_cfg = gr.update(minimum=0, maximum=total - 1, value=0)

    dur = total / fps
    total_text = f"{int(dur // 60):02d}:{int(dur % 60):02d}"
    time_text = f"00:00 / {total_text}"

    inference_state = predictor.init_state(video_path=path)
    predictor.clear_all_points_in_video(inference_state)
    RUNTIME['inference_state'] = inference_state
    RUNTIME['clicks'] = {}
    RUNTIME['id'] = 1
    RUNTIME['objects'] = {} # points
    RUNTIME['masks'] = {}   # masks
    RUNTIME['out_obj_ids'] = []

    return path, fps, first_frame, slider_cfg, time_text


from typing import List, Sequence

def cap_consecutive_ones_by_iou(
    flag: Sequence[int],
    iou: Sequence[float],
    max_keep: int = 3,
) -> List[int]:
    """
    Output rule:
      - If flag[i] == 0 -> output[i] = 1
      - If flag[i] == 1 -> for each consecutive run of 1s:
          - If run_length <= max_keep: keep all as 1
          - If run_length >  max_keep: keep only the indices of the top `max_keep`
            IoU values within the run as 1, set the rest to 0.
            (Tie-breaking: if IoU is the same, prefer the smaller index for stability.)

    Args:
        flag: A 0/1 sequence indicating positions to be processed. Runs of consecutive 1s
              are handled together.
        iou:  A float sequence (same length as `flag`), used to rank elements inside each
              run of consecutive 1s when the run is longer than `max_keep`.
        max_keep: Maximum number of 1s to keep within any consecutive-ones run.

    Returns:
        out: A list of 0/1 integers with the same length as `flag`, following the rules above.

    Raises:
        ValueError: If `flag` and `iou` have different lengths.
    """
    n = len(flag)
    if len(iou) != n:
        raise ValueError(f"len(flag)={n} != len(iou)={len(iou)}")

    # Initialize:
    # - positions where flag==0 are forced to 1
    # - positions where flag==1 are set to 0 first, and will be selected back to 1 per-run
    out = [1 if flag[i] == 0 else 0 for i in range(n)]

    i = 0
    while i < n:
        if flag[i] != 1:
            i += 1
            continue

        # Find a consecutive run of 1s: [i, j)
        j = i
        while j < n and flag[j] == 1:
            j += 1

        run_idx = list(range(i, j))
        if len(run_idx) <= max_keep:
            # Short run: keep all
            for k in run_idx:
                out[k] = 1
        else:
            # Long run: keep top `max_keep` by IoU within the run.
            # Sort by (-IoU, index) to ensure stable tie-breaking.
            top = sorted(run_idx, key=lambda k: (-float(iou[k]), k))[:max_keep]
            for k in top:
                out[k] = 1

        i = j

    return out

# ===============================
# Handlers
# ===============================

def on_upload(file_obj):
    """Handle uploaded video file."""
    if file_obj is None:
        return prepare_video(None)
    return prepare_video(file_obj.name)

def on_example_select(evt: gr.SelectData):
    """
    Handle click in the examples gallery.
    evt.index is the clicked item index (0, 1, 2, ...).
    """
    idx = evt.index
    if isinstance(idx, (list, tuple)):  # gallery 有时给 (row, col)
        idx = idx[0]

    if idx == 0:
        path = EXAMPLE_1
    elif idx == 1:
        path = EXAMPLE_2
    elif idx == 2:
        path = EXAMPLE_3
    else:
        raise gr.Error("Unknown example index.")

    return prepare_video(path)

def update_frame(idx, path, fps):
    """Update current frame + time text when slider moves."""
    if path is None:
        return None, "00:00 / 00:00"

    idx = int(idx)
    frame = read_frame_at(path, idx)
    if frame is None:
        raise gr.Error(f"Failed to read frame {idx}.")

    cur_sec = idx / fps if fps > 0 else 0.0
    cur_text = f"{int(cur_sec // 60):02d}:{int(cur_sec % 60):02d}"

    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    dur = total / fps if fps > 0 else 0.0
    end_text = f"{int(dur // 60):02d}:{int(dur % 60):02d}"

    return frame, f"{cur_text} / {end_text}"

def on_click(evt: gr.SelectData, point_type: str, video_path: str, frame_idx: int):
    """
    Handle click on the current frame:
    1. Read the corresponding frame from video (by frame_idx).
    2. Compute width/height (for sanity / later use).
    3. Draw a + / - marker at the clicked position.
    4. Return updated image.
    """
    if video_path is None:
        # No video loaded, nothing to do
        return None

    # 1) Read the frame corresponding to current slider value
    frame = read_frame_at(video_path, int(frame_idx))
    if frame is None:
        raise gr.Error(f"Failed to read frame {frame_idx} from video.")

    # 2) Get image size
    width, height = frame.size
    # (If you want to log it)
    print(f"[FRAME] size = {width} x {height}")

    # 3) Get click coordinates and point type
    x, y = evt.index  # evt.index = (x, y) in pixel coords
    print(f"[CLICK] ({x}, {y}), type={point_type}")

    # 4) Draw marker and return updated image
    point_type_norm = point_type.lower()
    if point_type_norm not in ("positive", "negative"):
        point_type_norm = "positive"

    try:
        clicks = RUNTIME['clicks'][frame_idx]
        clicks.append((x, y, point_type))
    except:
        clicks = [(x, y, point_type)]

    pts = []
    lbs = []
    for (x, y, t) in clicks:
        pts.append([int(x), int(y)])
        lbs.append(1 if t.lower() == "positive" else 0)
    input_point = np.array(pts, dtype=np.int32)
    input_label = np.array(lbs, dtype=np.int32)
    
    try:
        RUNTIME['clicks'][frame_idx] = clicks
    except:
        RUNTIME['clicks'][frame_idx] = {}
        RUNTIME['clicks'][frame_idx] = clicks

    prompts = {}
    prompts[RUNTIME['id']] = input_point, input_label

    rel_points = [[x / width, y / height] for x, y in input_point]
    points_tensor = torch.tensor(rel_points, dtype=torch.float32)
    points_labels_tensor = torch.tensor(input_label, dtype=torch.int32)

    _, RUNTIME['out_obj_ids'], low_res_masks, video_res_masks = predictor.add_new_points_or_box(
        inference_state=RUNTIME['inference_state'],
        frame_idx=frame_idx,
        obj_id=RUNTIME['id'],
        points=points_tensor,
        labels=points_labels_tensor,
    )
    mask_np = (video_res_masks[-1, 0].detach().cpu().numpy() > 0)
    mask = (mask_np > 0).astype(np.uint8) * 255
    
    painted_image = mask_painter(np.array(frame, dtype=np.uint8), mask, mask_color=4+RUNTIME['id'])
    RUNTIME['masks'][RUNTIME['id']] = {frame_idx: mask}

    for k, v in RUNTIME['masks'].items():
        if k == RUNTIME['id']:
            continue
        if frame_idx in v:
            mask = v[frame_idx]
            painted_image = mask_painter(painted_image, mask, mask_color=4+k)

    frame = Image.fromarray(painted_image)

    updated = draw_point_marker(frame, x, y, point_type_norm)
    return updated

def add_target(targets, selected):
    """Add new target and select it by default."""
    name = f"Target {len(targets) + 1}"
    
    if RUNTIME['clicks'] == {}:
        return targets, selected, gr.update(choices=targets, value=selected)

    targets = targets + [name]
    selected = selected + [name]

    RUNTIME['objects'][RUNTIME['id']] = RUNTIME['clicks']
    RUNTIME['id'] += 1
    RUNTIME['clicks'] = {}

    return targets, selected, gr.update(choices=targets, value=selected)

def toggle_upload(open_state: bool):
    """Toggle upload panel visibility and button label."""
    new_state = not open_state
    label = (
        "Upload Video (click to close)"
        if new_state
        else "Upload Video (click to open)"
    )
    return new_state, gr.update(visible=new_state), gr.update(value=label)

def on_mask_generation(video_path: str):
    """
    Mask generation across the video.
    Currently runs SAM-3 propagation and renders a mask video.
    """
    print("[DEBUG] Mask Generation button clicked.")
    if video_path is None:
        raise gr.Error("No video loaded.")

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores, iou_scores in predictor.propagate_in_video(
        RUNTIME['inference_state'],
        start_frame_idx=0,
        max_frame_num_to_track=1800,
        reverse=False,
        propagate_preflight=True,
    ):
        video_segments[frame_idx] = {
            out_obj_id: (video_res_masks[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(RUNTIME['out_obj_ids'])
        } 

    # render the segmentation results every few frames
    vis_frame_stride = 1
    out_h = RUNTIME['inference_state']['video_height']
    out_w = RUNTIME['inference_state']['video_width']
    img_to_video = []

    IMAGE_PATH = os.path.join(OUTPUT_DIR, 'images') # for sam3-3d-body
    MASKS_PATH = os.path.join(OUTPUT_DIR, 'masks')  # for sam3-3d-body
    MASKS_PATH_VIS = os.path.join(OUTPUT_DIR, 'masks_vis')  # for sam3-3d-body
    os.makedirs(IMAGE_PATH, exist_ok=True)
    os.makedirs(MASKS_PATH, exist_ok=True)
    os.makedirs(MASKS_PATH_VIS, exist_ok=True)

    for out_frame_idx in range(0, len(video_segments), vis_frame_stride):
        img = RUNTIME['inference_state']['images'][out_frame_idx].detach().float().cpu()
        img = (img + 1) / 2
        img = img.clamp(0, 1)
        img = F.interpolate(
            img.unsqueeze(0),
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        img = img.permute(1, 2, 0)
        img = (img.numpy() * 255).astype("uint8")
        img_pil = Image.fromarray(img).convert('RGB')
        msk = np.zeros_like(img[:, :, 0])
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            mask = (out_mask[0] > 0).astype(np.uint8) * 255
            img = mask_painter(img, mask, mask_color=4 + out_obj_id)
            msk[mask == 255] = out_obj_id
        img_to_video.append(img)

        msk_pil = Image.fromarray(msk).convert('P')
        msk_pil.putpalette(DAVIS_PALETTE)
        img_pil.save(os.path.join(IMAGE_PATH, f"{out_frame_idx:08d}.jpg"))
        msk_pil.save(os.path.join(MASKS_PATH, f"{out_frame_idx:08d}.png"))
        mask = (out_mask[0] == 0).astype(np.uint8) * 255
        img = mask_painter(img, mask, mask_color=1, mask_alpha=0.5)
        img_vis = Image.fromarray(img).convert('P')
        img_vis.save(os.path.join(MASKS_PATH_VIS, f"{out_frame_idx:08d}.png"))

    out_video_path = os.path.join(OUTPUT_DIR, f"video_mask_{time.time():.0f}.mp4")
    images_to_mp4(img_to_video, out_video_path, fps=RUNTIME['video_fps'])

    return out_video_path

def draw_keypoints_with_index(
    image: np.ndarray,
    keypoints,  # np.ndarray or torch.Tensor, (N,2)
    radius: int = 3,
    point_color=(0, 255, 0),
    text_color=(255, 255, 255),
    text_scale: float = 0.4,
    text_thickness: int = 1,
    offset=(5, -5),
) -> np.ndarray:
    out = image.copy()
    H, W = out.shape[:2]

    # accept numpy or torch, normalize to numpy float32
    if isinstance(keypoints, torch.Tensor):
        kps = keypoints.detach().cpu().float().numpy()
    else:
        kps = np.asarray(keypoints, dtype=np.float32)

    if kps.ndim != 2 or kps.shape[1] != 2:
        raise ValueError(f"keypoints must be (N,2), got {kps.shape}")

    for i, (x, y) in enumerate(kps):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        xi, yi = int(round(float(x))), int(round(float(y)))
        if 0 <= xi < W and 0 <= yi < H:
            cv2.circle(out, (xi, yi), radius, point_color, -1, cv2.LINE_AA)
            cv2.putText(
                out, str(i),
                (xi + offset[0], yi + offset[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale, text_color, text_thickness,
                cv2.LINE_AA,
            )

    return out

def mask_completion_and_iou_init(pred_amodal_masks, pred_res, obj_id, batch_masks, i, W, H):
    obj_ratio_dict_obj_id = None
    iou_dict_obj_id = None
    occ_dict_obj_id = None
    idx_dict_obj_id = None
    idx_path_obj_id = None
    
    # for completion
    pred_amodal_masks_com = [np.array(img.resize((pred_res[1], pred_res[0]))) for img in pred_amodal_masks]
    pred_amodal_masks_com = np.array(pred_amodal_masks_com).astype('uint8')
    pred_amodal_masks_com = (pred_amodal_masks_com.sum(axis=-1) > 600).astype('uint8')
    pred_amodal_masks_com = [keep_largest_component(pamc) for pamc in pred_amodal_masks_com]
    # for iou
    pred_amodal_masks = [np.array(img.resize((W, H))) for img in pred_amodal_masks]
    pred_amodal_masks = np.array(pred_amodal_masks).astype('uint8')
    pred_amodal_masks = (pred_amodal_masks.sum(axis=-1) > 600).astype('uint8')
    pred_amodal_masks = [keep_largest_component(pamc) for pamc in pred_amodal_masks]    # avoid small noisy masks
    # compute iou
    masks = [(np.array(Image.open(bm).convert('P'))==obj_id).astype('uint8') for bm in batch_masks]
    ious = []
    masks_margin_shrink = [bm.copy() for bm in masks]
    mask_H, mask_W = masks_margin_shrink[0].shape
    occlusion_threshold = 0.55
    for bi, (a, b) in enumerate(zip(masks, pred_amodal_masks)):
        # mute objects near margin
        zero_mask_cp = np.zeros_like(masks_margin_shrink[bi])
        zero_mask_cp[masks_margin_shrink[bi]==1] = 255
        mask_binary_cp = zero_mask_cp.astype(np.uint8)
        mask_binary_cp[:int(mask_H*0.05), :] = mask_binary_cp[-int(mask_H*0.05):, :] = mask_binary_cp[:, :int(mask_W*0.05)] = mask_binary_cp[:, -int(mask_W*0.05):] = 0
        if mask_binary_cp.max() == 0:   # margin objects
            ious.append(occlusion_threshold)
            continue
        area_a = (a > 0).sum()
        area_b = (b > 0).sum()
        if area_a == 0 and area_b == 0:
            ious.append(occlusion_threshold)
        elif area_a > area_b:
            ious.append(occlusion_threshold)
        else:
            inter = np.logical_and(a > 0, b > 0).sum()
            uni = np.logical_or(a > 0, b > 0).sum()
            obj_iou = inter / (uni + 1e-6)
            ious.append(obj_iou)

        if i == 0 and bi == 0:
            if ious[0] < occlusion_threshold:
                obj_ratio_dict_obj_id = bbox_from_mask(b)
            else:
                obj_ratio_dict_obj_id = bbox_from_mask(a)

    # remove fake completions (empty or from MARGINs)
    for pi, pamc in enumerate(pred_amodal_masks_com):
        # zero predictions, back to original masks
        if masks[pi].sum() > pred_amodal_masks[pi].sum():
            ious[pi] = occlusion_threshold
            pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res[0], pred_res[1], obj_id)
        # elif len(obj_ratio_dict)>0 and not are_bboxes_similar(bbox_from_mask(pred_amodal_masks[pi]), obj_ratio_dict[obj_id]):
        #     ious[pi] = occlusion_threshold
        #     pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res_hi[0], pred_res_hi[1], obj_id)
        elif is_super_long_or_wide(pred_amodal_masks[pi], obj_id):
            ious[pi] = occlusion_threshold
            pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res[0], pred_res[1], obj_id)
        elif is_skinny_mask(pred_amodal_masks[pi]):
            ious[pi] = occlusion_threshold
            pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res[0], pred_res[1], obj_id)
        # elif masks[pi].sum() == 0: # TODO: recover empty masks in future versions (to avoid severe fake completion)
        #     ious[pi] = occlusion_threshold
        #     pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res_hi[0], pred_res_hi[1], obj_id)

    # confirm occlusions & save masks (for HMR)
    iou_dict_obj_id = [float(iou_) for iou_ in ious]
    arr = iou_dict_obj_id[:]
    for isb in range(1, len(arr) - 1):
        if arr[isb] == occlusion_threshold and arr[isb-1] < occlusion_threshold and arr[isb+1] < occlusion_threshold:
            arr[isb] = 0.0

    iou_dict_obj_id = arr
    occ_dict_obj_id = [1 if ix >= occlusion_threshold else 0 for ix in iou_dict_obj_id]
    start, end = (idxs := [ix for ix,x in enumerate(iou_dict_obj_id) if x < occlusion_threshold]) and (idxs[0], idxs[-1]) or (None, None)

    if start is not None and end is not None:
        start = max(0, start-2)
        end = min(len(pred_amodal_masks), end+2)
        idx_dict_obj_id = (start, end)
        completion_path = ''.join(random.choices('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=4))
        completion_image_path = f'{OUTPUT_DIR}/completion/{completion_path}/images'
        completion_masks_path = f'{OUTPUT_DIR}/completion/{completion_path}/masks'
        os.makedirs(completion_image_path, exist_ok=True)
        os.makedirs(completion_masks_path, exist_ok=True)
        idx_path_obj_id = {'images': completion_image_path, 'masks': completion_masks_path}
    return obj_ratio_dict_obj_id, iou_dict_obj_id, occ_dict_obj_id, idx_dict_obj_id, idx_path_obj_id

def mask_completion_and_iou_final(pred_amodal_masks, pred_res, obj_id, batch_masks, W, H, iou_dict_obj_id, occ_dict_obj_id, idx_path_obj_id, keep_idx):    
    keep_id = [io for io, vo in enumerate(keep_idx) if vo == 1]
    batch_masks_ = [batch_masks[io] for io in keep_id]

    # for completion
    zero_com = np.zeros_like(np.array(pred_amodal_masks[0].resize((pred_res[1], pred_res[0])))[:,:,0])
    pred_amodal_masks_com = [np.array(img.resize((pred_res[1], pred_res[0]))) for img in pred_amodal_masks]
    pred_amodal_masks_com = np.array(pred_amodal_masks_com).astype('uint8')
    pred_amodal_masks_com = (pred_amodal_masks_com.sum(axis=-1) > 600).astype('uint8')
    pred_amodal_masks_com = [keep_largest_component(pamc) for pamc in pred_amodal_masks_com]
    # for iou
    pred_amodal_masks = [np.array(img.resize((W, H))) for img in pred_amodal_masks]
    pred_amodal_masks = np.array(pred_amodal_masks).astype('uint8')
    pred_amodal_masks = (pred_amodal_masks.sum(axis=-1) > 600).astype('uint8')
    pred_amodal_masks = [keep_largest_component(pamc) for pamc in pred_amodal_masks]    # avoid small noisy masks
    # compute iou
    masks = [(np.array(Image.open(bm).convert('P'))==obj_id).astype('uint8') for bm in batch_masks_]
    ious = []
    masks_margin_shrink = [bm.copy() for bm in masks]
    mask_H, mask_W = masks_margin_shrink[0].shape
    occlusion_threshold = 0.65
    for bi, (a, b) in enumerate(zip(masks, pred_amodal_masks)):
        # mute objects near margin
        zero_mask_cp = np.zeros_like(masks_margin_shrink[bi])
        zero_mask_cp[masks_margin_shrink[bi]==1] = 255
        mask_binary_cp = zero_mask_cp.astype(np.uint8)
        mask_binary_cp[:int(mask_H*0.05), :] = mask_binary_cp[-int(mask_H*0.05):, :] = mask_binary_cp[:, :int(mask_W*0.05)] = mask_binary_cp[:, -int(mask_W*0.05):] = 0
        if mask_binary_cp.max() == 0:   # margin objects
            ious.append(occlusion_threshold)
            continue
        area_a = (a > 0).sum()
        area_b = (b > 0).sum()
        if area_a == 0 and area_b == 0:
            ious.append(occlusion_threshold)
        elif area_a > area_b:
            ious.append(occlusion_threshold)
        else:
            inter = np.logical_and(a > 0, b > 0).sum()
            uni = np.logical_or(a > 0, b > 0).sum()
            obj_iou = inter / (uni + 1e-6)
            ious.append(obj_iou)

    # remove fake completions (empty or from MARGINs)
    for pi, pamc in enumerate(pred_amodal_masks_com):
        # zero predictions, back to original masks
        if masks[pi].sum() > pred_amodal_masks[pi].sum():
            ious[pi] = occlusion_threshold
            pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res[0], pred_res[1], obj_id)
        # elif len(obj_ratio_dict)>0 and not are_bboxes_similar(bbox_from_mask(pred_amodal_masks[pi]), obj_ratio_dict[obj_id]):
        #     ious[pi] = occlusion_threshold
        #     pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res_hi[0], pred_res_hi[1], obj_id)
        elif is_super_long_or_wide(pred_amodal_masks[pi], obj_id):
            ious[pi] = occlusion_threshold
            pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res[0], pred_res[1], obj_id)
        elif is_skinny_mask(pred_amodal_masks[pi]):
            ious[pi] = occlusion_threshold
            pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res[0], pred_res[1], obj_id)
        # elif masks[pi].sum() == 0: # TODO: recover empty masks in future versions (to avoid severe fake completion)
        #     ious[pi] = occlusion_threshold
        #     pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res_hi[0], pred_res_hi[1], obj_id)

    # confirm occlusions & save masks (for HMR)
    iou_dict_obj_id_ = [float(iou_) for iou_ in ious]
    arr = iou_dict_obj_id_[:]
    for isb in range(1, len(arr) - 1):
        if arr[isb] == occlusion_threshold and arr[isb-1] < occlusion_threshold and arr[isb+1] < occlusion_threshold:
            arr[isb] = 0.0

    iou_dict_obj_id_ = arr
    occ_dict_obj_id_ = [1 if ix >= occlusion_threshold else 0 for ix in iou_dict_obj_id_]
    
    completion_masks_path = idx_path_obj_id['masks']
    current_id = 0  # within start & end
    final_pred_amodal_masks_com = []
    for ki, keep_id in enumerate(keep_idx): # all within batch_size
        if keep_id == 0:
            final_pred_amodal_masks_com.append(zero_com)
            continue
        
        occ_dict_obj_id[ki] = occ_dict_obj_id_[current_id]
        iou_dict_obj_id[ki] = iou_dict_obj_id_[current_id]
        
        if occ_dict_obj_id_[current_id] == 1: # only save heavy occluded results
            current_id += 1
            final_pred_amodal_masks_com.append(zero_com)
            continue
        final_pred_amodal_masks_com.append(pred_amodal_masks_com[current_id])
        mask_idx_ = pred_amodal_masks[current_id].copy()
        mask_idx_[mask_idx_ > 0] = obj_id
        mask_idx_ = Image.fromarray(mask_idx_).convert('P')
        mask_idx_.putpalette(DAVIS_PALETTE)
        mask_idx_.save(os.path.join(completion_masks_path, f"{ki:08d}.png"))
        current_id += 1
    
    return iou_dict_obj_id, occ_dict_obj_id, final_pred_amodal_masks_com

def _run_obj_diffusion(worker, obj_id, modal_pixels, depth_pixels, batch_masks,
                       i, batch_size, pred_res, pred_res_hi, W, H,
                       output_dir, occ_dict_len, depth_pixels_hi=None,
                       num_inference_steps=25):
    """Run the full diffusion pipeline for a single obj_id on a specific GPU worker.

    Returns a dict with keys: obj_ratio, iou_dict, occ_dict, idx_dict, idx_path.
    """
    w_dev = worker['device']
    torch.cuda.set_device(w_dev)
    _t_start = time.time()

    pipeline_mask = worker['pipeline_mask']
    pipeline_rgb = worker['pipeline_rgb']
    gen = worker['generator']

    # --- Low-res mask prediction (occlusion detection) ---
    modal_batch = modal_pixels[:, i:i + batch_size, :, :, :].to(w_dev)
    depth_batch = depth_pixels[:, i:i + batch_size, :, :, :].to(w_dev)

    pred_amodal_masks = pipeline_mask(
        modal_batch,
        depth_batch,
        height=pred_res[0],
        width=pred_res[1],
        num_frames=modal_batch.shape[1],
        num_inference_steps=num_inference_steps,
        decode_chunk_size=8,
        motion_bucket_id=127,
        fps=8,
        noise_aug_strength=0.02,
        min_guidance_scale=1.5,
        max_guidance_scale=1.5,
        generator=gen,
    ).frames[0]

    print(f"  [TIMER] pipeline_mask (low-res, obj {obj_id}, {w_dev}): {time.time() - _t_start:.2f}s")

    obj_ratio_dict_obj_id, iou_dict_obj_id, occ_dict_obj_id, idx_dict_obj_id, idx_path_obj_id = \
        mask_completion_and_iou_init(pred_amodal_masks, pred_res, obj_id, batch_masks, i, W, H)

    result = {
        'obj_ratio': obj_ratio_dict_obj_id,
        'iou_dict': iou_dict_obj_id,
        'occ_dict': occ_dict_obj_id,
        'idx_dict': idx_dict_obj_id,
        'idx_path': idx_path_obj_id,
    }

    # --- Hi-res mask + RGB completion (if occlusion detected) ---
    if idx_dict_obj_id is not None and idx_path_obj_id is not None:
        start, end = idx_dict_obj_id
        completion_image_path = idx_path_obj_id['images']

        modal_pixels_current, ori_shape = load_and_transform_masks(
            output_dir + "/masks", resolution=pred_res_hi, obj_id=obj_id)
        rgb_pixels_current, _, _ = load_and_transform_rgbs(
            output_dir + "/images", resolution=pred_res_hi)

        modal_pixels_current = modal_pixels_current[:, i:i + batch_size, :, :, :]
        modal_pixels_current = modal_pixels_current[:, start:end]

        rgb_pixels_current = rgb_pixels_current[:, i:i + batch_size, :, :, :][:, start:end]
        modal_obj_mask = (modal_pixels_current > 0).float()
        modal_background = 1 - modal_obj_mask
        rgb_pixels_current = (rgb_pixels_current + 1) / 2
        modal_rgb_pixels = rgb_pixels_current * modal_obj_mask + modal_background
        modal_rgb_pixels = modal_rgb_pixels * 2 - 1

        keep_idx = cap_consecutive_ones_by_iou(occ_dict_obj_id[start:end], iou_dict_obj_id[start:end])
        mask_idx = torch.tensor(keep_idx, device='cpu').bool()

        _t_hires = time.time()
        pred_amodal_masks_ = pipeline_mask(
            modal_pixels_current[:, mask_idx].to(w_dev),
            depth_pixels_hi[:, i:i + batch_size, :, :, :][:, start:end][:, mask_idx].to(w_dev),
            height=pred_res_hi[0],
            width=pred_res_hi[1],
            num_frames=sum(keep_idx),
            num_inference_steps=num_inference_steps,
            decode_chunk_size=8,
            motion_bucket_id=127,
            fps=8,
            noise_aug_strength=0.02,
            min_guidance_scale=1.5,
            max_guidance_scale=1.5,
            generator=gen,
        ).frames[0]
        print(f"  [TIMER] pipeline_mask (hi-res, obj {obj_id}, {w_dev}): {time.time() - _t_hires:.2f}s")

        iou_dict_obj_id, occ_dict_obj_id, pred_amodal_masks_com = mask_completion_and_iou_final(
            pred_amodal_masks_,
            pred_res_hi,
            obj_id,
            batch_masks,
            W, H,
            iou_dict_obj_id,
            occ_dict_obj_id,
            idx_path_obj_id,
            [0]*start + keep_idx + [0]*(occ_dict_len - end),
        )
        result['iou_dict'] = iou_dict_obj_id
        result['occ_dict'] = occ_dict_obj_id

        _t_rgb = time.time()
        print(f"  content completion by diffusion-vas (obj {obj_id}, {w_dev}) ...")
        keep_idx = cap_consecutive_ones_by_iou(occ_dict_obj_id[start:end], iou_dict_obj_id[start:end])
        mask_idx = torch.tensor(keep_idx, device='cpu').bool()
        pred_amodal_masks_current = pred_amodal_masks_com[start:end]
        pred_amodal_masks_current = [xxx for xxx, mmm in zip(pred_amodal_masks_current, keep_idx) if mmm == 1]
        modal_mask_union = (modal_pixels_current[:, mask_idx][0, :, 0, :, :].cpu().numpy() > 0).astype('uint8')
        pred_amodal_masks_current = np.logical_or(pred_amodal_masks_current, modal_mask_union).astype('uint8')
        pred_amodal_masks_tensor = torch.from_numpy(
            np.where(pred_amodal_masks_current == 0, -1, 1)
        ).float().unsqueeze(0).unsqueeze(2).repeat(1, 1, 3, 1, 1)

        pred_amodal_rgb = pipeline_rgb(
            modal_rgb_pixels[:, mask_idx].to(w_dev),
            pred_amodal_masks_tensor.to(w_dev),
            height=pred_res_hi[0],
            width=pred_res_hi[1],
            num_frames=sum(keep_idx),
            num_inference_steps=num_inference_steps,
            decode_chunk_size=8,
            motion_bucket_id=127,
            fps=8,
            noise_aug_strength=0.02,
            min_guidance_scale=1.5,
            max_guidance_scale=1.5,
            generator=gen,
        ).frames[0]
        print(f"  [TIMER] pipeline_rgb (completion, obj {obj_id}, {w_dev}): {time.time() - _t_rgb:.2f}s")

        pred_i = 0
        save_i = start - 1
        for keep_i, occ_i in zip(keep_idx, occ_dict_obj_id[start:end]):
            save_i += 1
            if occ_i == 1:
                if keep_i == 1:
                    pred_i += 1
                continue
            if keep_i == 1:
                rgb_i = np.array(pred_amodal_rgb[pred_i]).astype('uint8')
                rgb_i = cv2.resize(rgb_i, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(completion_image_path, f"{save_i:08d}.jpg"),
                            cv2.cvtColor(rgb_i, cv2.COLOR_RGB2BGR))
                pred_i += 1
                continue

    print(f"  [TIMER] obj {obj_id} diffusion total ({w_dev}): {time.time() - _t_start:.2f}s")
    return obj_id, result


def on_4d_generation(video_path: str):
    """4D generation: diffusion-based occlusion recovery + HMR mesh estimation."""
    print("[DEBUG] 4D Generation button clicked.")

    IMAGE_PATH = os.path.join(OUTPUT_DIR, 'images')
    MASKS_PATH = os.path.join(OUTPUT_DIR, 'masks')
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.tiff", "*.webp"]
    images_list = sorted(
        [image for ext in image_extensions for image in glob.glob(os.path.join(IMAGE_PATH, ext))]
    )
    masks_list = sorted(
        [image for ext in image_extensions for image in glob.glob(os.path.join(MASKS_PATH, ext))]
    )

    os.makedirs(f"{OUTPUT_DIR}/rendered_frames", exist_ok=True)
    for obj_id in RUNTIME['out_obj_ids']:
        os.makedirs(f"{OUTPUT_DIR}/mesh_4d_individual/{obj_id}", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/focal_4d_individual/{obj_id}", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/rendered_frames_individual/{obj_id}", exist_ok=True)
        if RUNTIME.get('smpl_export', False):
            os.makedirs(f"{OUTPUT_DIR}/skeleton_4d_individual/{obj_id}", exist_ok=True)

    batch_size = RUNTIME['batch_size']
    n = len(images_list)

    pred_res = RUNTIME['detection_resolution']
    pred_res_hi = RUNTIME['completion_resolution']
    w, h = Image.open(images_list[0]).size
    pred_res = pred_res if h < w else pred_res[::-1]
    pred_res_hi = pred_res_hi if h < w else pred_res_hi[::-1]

    modal_pixels_list = []
    depth_pixels_hi = None
    if len(diffusion_workers) > 0:
        for obj_id in RUNTIME['out_obj_ids']:
            modal_pixels, ori_shape = load_and_transform_masks(OUTPUT_DIR + "/masks", resolution=pred_res, obj_id=obj_id)
            modal_pixels_list.append(modal_pixels)
        rgb_pixels, _, raw_rgb_pixels = load_and_transform_rgbs(OUTPUT_DIR + "/images", resolution=pred_res)
        depth_pixels = rgb_to_depth(rgb_pixels, depth_model)
        # Pre-compute hi-res depth (shared across all obj workers, avoids concurrent depth_model access)
        rgb_pixels_hi, _, _ = load_and_transform_rgbs(OUTPUT_DIR + "/images", resolution=pred_res_hi)
        depth_pixels_hi = rgb_to_depth(rgb_pixels_hi, depth_model)

    mhr_shape_scale_dict = {}
    obj_ratio_dict = {}

    print("Running FOV estimator ...")
    input_image = np.array(Image.open(images_list[0])).astype('uint8')
    cam_int = sam3_3d_body_model.fov_estimator.get_cam_intrinsics(input_image)

    num_workers = len(diffusion_workers)

    # Initialize rendering backend
    _render_backend = CONFIG.get("sam_3d_body", {}).get("rendering", {}).get("backend", "pyrender")
    _mesh_cfg = CONFIG.get("sam_3d_body", {}).get("mesh_export", {})
    _mesh_export_enabled = _mesh_cfg.get("enable", True)
    _mesh_export_formats = tuple(_mesh_cfg.get("formats", ["ply", "glb"]))
    _animated_glb = _mesh_cfg.get("animated_glb", True)

    gpu_renderer = None
    if _render_backend == "pytorch3d":
        from sam_3d_body.visualization.renderer_pytorch3d import PyTorch3DBatchRenderer
        gpu_renderer = PyTorch3DBatchRenderer(device=device)
        print(f"[INFO] Using PyTorch3D GPU batch renderer on {device}")
    elif _render_backend == "nvdiffrast":
        from sam_3d_body.visualization.renderer_nvdiffrast import NvDiffrastBatchRenderer
        gpu_renderer = NvDiffrastBatchRenderer(device=device)
        print(f"[INFO] Using nvdiffrast GPU batch renderer on {device}")

    for i in tqdm(range(0, n, batch_size)):
        _t_batch_start = time.time()
        batch_images = images_list[i:i + batch_size]
        batch_masks  = masks_list[i:i + batch_size]

        W, H = Image.open(batch_masks[0]).size

        idx_dict = {}
        idx_path = {}
        occ_dict = {}
        iou_dict = {}
        if len(modal_pixels_list) > 0:
            _t_occ_start = time.time()
            print("detect occlusions (multi-GPU parallel) ...")

            # Submit diffusion work for each obj_id to different GPU workers in parallel
            futures = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                for wi, (modal_pixels, obj_id) in enumerate(zip(modal_pixels_list, RUNTIME['out_obj_ids'])):
                    worker = diffusion_workers[wi % num_workers]
                    fut = executor.submit(
                        _run_obj_diffusion,
                        worker, obj_id, modal_pixels, depth_pixels, batch_masks,
                        i, batch_size, pred_res, pred_res_hi, W, H,
                        OUTPUT_DIR, len(batch_masks), depth_pixels_hi,
                        CONFIG.completion.get('num_inference_steps', 25),
                    )
                    futures[fut] = obj_id

                # Collect results
                for fut in concurrent.futures.as_completed(futures):
                    obj_id, result = fut.result()
                    if result['obj_ratio'] is not None:
                        obj_ratio_dict[obj_id] = result['obj_ratio']
                    if result['iou_dict'] is not None:
                        iou_dict[obj_id] = result['iou_dict']
                    if result['occ_dict'] is not None:
                        occ_dict[obj_id] = result['occ_dict']
                    if result['idx_dict'] is not None:
                        idx_dict[obj_id] = result['idx_dict']
                    if result['idx_path'] is not None:
                        idx_path[obj_id] = result['idx_path']

        else:
            for obj_id in RUNTIME['out_obj_ids']:
                occ_dict[obj_id] = [1] * len(batch_masks)

        if len(modal_pixels_list) > 0:
            print(f"  [TIMER] occlusion total: {time.time() - _t_occ_start:.2f}s")

        _t_hmr_start = time.time()
        mask_outputs, id_batch, empty_frame_list = process_image_with_mask(sam3_3d_body_model, batch_images, batch_masks, idx_path, idx_dict, mhr_shape_scale_dict, occ_dict, cam_int=cam_int, iou_dict=iou_dict, predictor=predictor)
        print(f"  [TIMER] HMR (process_image_with_mask): {time.time() - _t_hmr_start:.2f}s")

        _t_vis_start = time.time()

        # Build per-frame output lists
        outputs_by_frame = []
        ids_by_frame = []
        image_paths = []
        num_empth_ids = 0
        for frame_id in range(len(batch_images)):
            image_path = batch_images[frame_id]
            if frame_id in empty_frame_list:
                outputs_by_frame.append(None)
                ids_by_frame.append(None)
                num_empth_ids += 1
            else:
                outputs_by_frame.append(mask_outputs[frame_id-num_empth_ids])
                ids_by_frame.append(id_batch[frame_id-num_empth_ids])
            image_paths.append(image_path)

        if gpu_renderer is not None:
            # ---- PyTorch3D GPU batch rendering path ----
            _t_rd = time.time()
            imgs = [cv2.imread(p) for p in image_paths]
            _t_imread = time.time() - _t_rd

            _t_rd = time.time()
            combined_imgs = batch_render_combined(imgs, outputs_by_frame, sam3_3d_body_model.faces, ids_by_frame, gpu_renderer)
            _t_combined = time.time() - _t_rd

            _t_rd = time.time()
            individual_imgs = batch_render_individual(imgs, outputs_by_frame, sam3_3d_body_model.faces, ids_by_frame, gpu_renderer)
            _t_individual = time.time() - _t_rd

            def _save_io(frame_idx):
                image_path = image_paths[frame_idx]
                frame_stem = os.path.basename(image_path)[:-4]
                cv2.imwrite(f"{OUTPUT_DIR}/rendered_frames/{frame_stem}.jpg", combined_imgs[frame_idx])
                for pi, pimg in enumerate(individual_imgs[frame_idx]):
                    cv2.imwrite(f"{OUTPUT_DIR}/rendered_frames_individual/{pi+1}/{frame_stem}_{pi+1}.jpg", pimg)
                if _mesh_export_enabled and outputs_by_frame[frame_idx] is not None:
                    save_mesh_results(
                        outputs=outputs_by_frame[frame_idx],
                        faces=sam3_3d_body_model.faces,
                        save_dir=f"{OUTPUT_DIR}/mesh_4d_individual",
                        focal_dir=f"{OUTPUT_DIR}/focal_4d_individual",
                        image_path=image_path,
                        id_current=ids_by_frame[frame_idx],
                        export_formats=_mesh_export_formats,
                    )
                if RUNTIME.get('smpl_export', False) and outputs_by_frame[frame_idx] is not None:
                    save_skeleton_results(
                        outputs=outputs_by_frame[frame_idx],
                        skeleton_dir=f"{OUTPUT_DIR}/skeleton_4d_individual",
                        image_path=image_path,
                        id_current=ids_by_frame[frame_idx],
                    )

            _t_rd = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=_VIS_WORKERS) as pool:
                list(pool.map(_save_io, range(len(image_paths))))
            _t_saveio = time.time() - _t_rd

            print(f"  [TIMER] vis detail: imread={_t_imread:.2f}s combined={_t_combined:.2f}s individual={_t_individual:.2f}s save_io={_t_saveio:.2f}s")
        else:
            # ---- Legacy PyRender path ----
            frame_args = list(zip(image_paths, outputs_by_frame, ids_by_frame))

            def _render_and_save(args):
                image_path, mask_output, id_current = args
                img = cv2.imread(image_path)
                rend_img = visualize_sample_together(img, mask_output, sam3_3d_body_model.faces, id_current)
                cv2.imwrite(
                    f"{OUTPUT_DIR}/rendered_frames/{os.path.basename(image_path)[:-4]}.jpg",
                    rend_img.astype(np.uint8),
                )
                rend_img_list = visualize_sample(img, mask_output, sam3_3d_body_model.faces, id_current)
                for ri, rend_img in enumerate(rend_img_list):
                    cv2.imwrite(
                        f"{OUTPUT_DIR}/rendered_frames_individual/{ri+1}/{os.path.basename(image_path)[:-4]}_{ri+1}.jpg",
                        rend_img.astype(np.uint8),
                    )
                if _mesh_export_enabled:
                    save_mesh_results(
                        outputs=mask_output,
                        faces=sam3_3d_body_model.faces,
                        save_dir=f"{OUTPUT_DIR}/mesh_4d_individual",
                        focal_dir=f"{OUTPUT_DIR}/focal_4d_individual",
                        image_path=image_path,
                        id_current=id_current,
                        export_formats=_mesh_export_formats,
                    )
                if RUNTIME.get('smpl_export', False):
                    save_skeleton_results(
                        outputs=mask_output,
                        skeleton_dir=f"{OUTPUT_DIR}/skeleton_4d_individual",
                        image_path=image_path,
                        id_current=id_current,
                    )

            if _VIS_WORKERS > 1 and len(frame_args) > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(_VIS_WORKERS, len(frame_args))) as pool:
                    for _ in pool.map(_render_and_save, frame_args):
                        pass
            else:
                for args in frame_args:
                    _render_and_save(args)
        print(f"  [TIMER] vis+mesh+IO: {time.time() - _t_vis_start:.2f}s")

        print(f"  [TIMER] batch {i//batch_size} total: {time.time() - _t_batch_start:.2f}s")

    if RUNTIME.get('smpl_export', False):
        aggregate_skeleton_npz(f"{OUTPUT_DIR}/skeleton_4d_individual")

    if _mesh_export_enabled and _animated_glb:
        aggregate_mesh_glb(f"{OUTPUT_DIR}/mesh_4d_individual", fps=RUNTIME['video_fps'])

    out_4d_path = os.path.join(OUTPUT_DIR, f"4d_{time.time():.0f}.mp4")
    jpg_folder_to_mp4(f"{OUTPUT_DIR}/rendered_frames", out_4d_path, fps=RUNTIME['video_fps'])

    return out_4d_path


on_mask_generation = gpu_profile(on_mask_generation)
on_4d_generation   = gpu_profile(on_4d_generation)


# ===============================
# UI Layout
# ===============================

with gr.Blocks(title="SAM-Body4D") as demo:
    # States
    video_state = gr.State(None)
    fps_state = gr.State(1.0)
    point_type_state = gr.State("positive")   # "positive" or "negative"
    targets_state = gr.State([])
    selected_targets_state = gr.State([])
    upload_open_state = gr.State(False)

    with gr.Row():
        # -------- Left column: examples + frame + controls --------
        with gr.Column(scale=1):
            gr.Markdown("### Example Videos")

            examples_gallery = gr.Gallery(
                value=[
                    (EX1_THUMB, "Example 1"),
                    (EX2_THUMB, "Example 2"),
                    (EX3_THUMB, "Example 3"),
                ],
                show_label=False,
                columns=3,
                height=160,
            )

            current_frame = gr.Image(
                label="Current Frame (click to annotate)",
                interactive=True,
                sources=[],
            )

            toggle_upload_btn = gr.Button(
                "Upload Video (click to open)",
                size="sm",
                variant="secondary",
            )

            upload_panel = gr.Row(visible=False)
            with upload_panel:
                upload = gr.File(
                    label="Video File",
                    file_count="single",
                )

            frame_slider = gr.Slider(
                minimum=0,
                maximum=0,
                value=0,
                step=1,
                label="Frame Index",
            )

            time_text = gr.Text("00:00 / 00:00", label="Time")

            point_radio = gr.Radio(
                choices=["Positive", "Negative"],
                value="Positive",
                label="Point Type",
                interactive=True,
            )

            targets_box = gr.CheckboxGroup(
                label="Targets",
                choices=[],
                value=[],
            )
            add_target_btn = gr.Button("Add Target")

        # -------- Right column: result image + buttons + 4D video --------
        with gr.Column(scale=1):
            result_display = gr.Video(label="Segmentation Result")
            with gr.Row():
                mask_gen_btn = gr.Button("Mask Generation")
                gen4d_btn = gr.Button("4D Generation")
            fourd_display = gr.Video(label="4D Result")

    # ===============================
    # Event bindings
    # ===============================

    # Toggle upload panel
    toggle_upload_btn.click(
        fn=toggle_upload,
        inputs=[upload_open_state],
        outputs=[upload_open_state, upload_panel, toggle_upload_btn],
    )

    # Upload → load video
    upload.change(
        fn=on_upload,
        inputs=[upload],
        outputs=[video_state, fps_state, current_frame, frame_slider, time_text],
    )

    # Click example thumbnail in gallery → load that example
    examples_gallery.select(
        fn=on_example_select,
        inputs=None,
        outputs=[video_state, fps_state, current_frame, frame_slider, time_text],
    )

    # Slider → update frame + time
    frame_slider.change(
        fn=update_frame,
        inputs=[frame_slider, video_state, fps_state],
        outputs=[current_frame, time_text],
    )

    point_radio.change(
        fn=lambda v: v.lower(),   # "Positive" / "Negative" → "positive" / "negative"
        inputs=[point_radio],
        outputs=[point_type_state],
    )

    # Click on current frame
    current_frame.select(
        fn=on_click,
        inputs=[point_type_state, video_state, frame_slider],
        outputs=[current_frame],
    )

    # Add target
    add_target_btn.click(
        fn=add_target,
        inputs=[targets_state, selected_targets_state],
        outputs=[targets_state, selected_targets_state, targets_box],
    )

    mask_gen_btn.click(
        fn=on_mask_generation,
        inputs=[video_state], 
        outputs=[result_display],
    )

    gen4d_btn.click(
        fn=on_4d_generation,
        inputs=[video_state],      
        outputs=[fourd_display],  
    )


if __name__ == "__main__":

    init_runtime()
    demo.launch()
