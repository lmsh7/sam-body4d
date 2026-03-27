"""
Post-processing script for SAM-Body4D skeleton exports.
Computes sports statistics (movement distance, velocity, joint angles) from
exported skeleton data and outputs CSV files.

Usage:
    python scripts/export_sports_stats.py --input_dir outputs/<timestamp> --fps 30
"""

import argparse
import csv
import json
import os
import glob
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# MHR70 joint index constants
# ---------------------------------------------------------------------------
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_HIP = 9
RIGHT_HIP = 10
LEFT_KNEE = 11
RIGHT_KNEE = 12
LEFT_ANKLE = 13
RIGHT_ANKLE = 14
RIGHT_WRIST = 41
LEFT_WRIST = 62
NECK = 69

KEY_JOINTS = {
    "nose": NOSE,
    "neck": NECK,
    "left_shoulder": LEFT_SHOULDER,
    "right_shoulder": RIGHT_SHOULDER,
    "left_elbow": LEFT_ELBOW,
    "right_elbow": RIGHT_ELBOW,
    "left_wrist": LEFT_WRIST,
    "right_wrist": RIGHT_WRIST,
    "left_hip": LEFT_HIP,
    "right_hip": RIGHT_HIP,
    "left_knee": LEFT_KNEE,
    "right_knee": RIGHT_KNEE,
    "left_ankle": LEFT_ANKLE,
    "right_ankle": RIGHT_ANKLE,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_person_data(person_dir: str) -> dict:
    """Load skeleton data for one person from NPZ or fallback to JSONs."""
    npz_path = os.path.join(person_dir, "all_frames.npz")
    if os.path.exists(npz_path):
        data = dict(np.load(npz_path, allow_pickle=True))
        # frame_names may be stored as object array
        if "frame_names" in data:
            data["frame_names"] = list(data["frame_names"])
        return data

    # Fallback: load individual JSONs
    json_files = sorted(glob.glob(os.path.join(person_dir, "*.json")))
    if not json_files:
        return None

    frames = []
    for jf in json_files:
        with open(jf) as f:
            frames.append(json.load(f))

    n = len(frames)
    valid_mask = np.array([not d.get("empty", False) for d in frames])
    frame_names = [d.get("frame_id", Path(jf).stem) for d, jf in zip(frames, json_files)]

    first_valid = next((d for d in frames if not d.get("empty", False)), None)
    if first_valid is None:
        return None

    n_joints = len(first_valid["keypoints_3d"])
    kp3d = np.full((n, n_joints, 3), np.nan)
    for i, d in enumerate(frames):
        if not d.get("empty", False):
            kp3d[i] = np.array(d["keypoints_3d"])

    return {
        "keypoints_3d": kp3d,
        "valid_mask": valid_mask,
        "frame_names": frame_names,
    }


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------
def compute_hip_center(kp3d: np.ndarray) -> np.ndarray:
    """Compute hip center as midpoint of left and right hip. Shape: (N, 3)."""
    return (kp3d[:, LEFT_HIP, :] + kp3d[:, RIGHT_HIP, :]) / 2.0


def compute_cumulative_distance(positions: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Compute per-frame cumulative Euclidean distance. Shape: (N,)."""
    n = len(positions)
    cum_dist = np.zeros(n)
    for t in range(1, n):
        if valid[t] and valid[t - 1]:
            step = np.linalg.norm(positions[t] - positions[t - 1])
            cum_dist[t] = cum_dist[t - 1] + step
        else:
            cum_dist[t] = cum_dist[t - 1]
    return cum_dist


def compute_velocity(positions: np.ndarray, valid: np.ndarray, fps: float) -> np.ndarray:
    """Compute per-frame instantaneous speed (scalar). Shape: (N,)."""
    n = len(positions)
    vel = np.full(n, np.nan)
    for t in range(1, n):
        if valid[t] and valid[t - 1]:
            vel[t] = np.linalg.norm(positions[t] - positions[t - 1]) * fps
    return vel


def compute_acceleration(velocity: np.ndarray, valid: np.ndarray, fps: float) -> np.ndarray:
    """Compute per-frame acceleration from velocity. Shape: (N,)."""
    n = len(velocity)
    acc = np.full(n, np.nan)
    for t in range(2, n):
        if valid[t] and valid[t - 1] and not np.isnan(velocity[t]) and not np.isnan(velocity[t - 1]):
            acc[t] = (velocity[t] - velocity[t - 1]) * fps
    return acc


def compute_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Compute angle at p2 formed by vectors p2->p1 and p2->p3, in degrees."""
    v1 = p1 - p2
    v2 = p3 - p2
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return np.nan
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def compute_joint_angles(kp3d: np.ndarray, valid: np.ndarray) -> dict:
    """Compute per-frame joint angles for key joints. Returns dict of (N,) arrays."""
    n = len(kp3d)
    angles = {
        "left_knee_angle": np.full(n, np.nan),
        "right_knee_angle": np.full(n, np.nan),
        "left_elbow_angle": np.full(n, np.nan),
        "right_elbow_angle": np.full(n, np.nan),
        "left_hip_angle": np.full(n, np.nan),
        "right_hip_angle": np.full(n, np.nan),
    }
    for t in range(n):
        if not valid[t]:
            continue
        kp = kp3d[t]
        # Knee: hip -> knee -> ankle
        angles["left_knee_angle"][t] = compute_angle(kp[LEFT_HIP], kp[LEFT_KNEE], kp[LEFT_ANKLE])
        angles["right_knee_angle"][t] = compute_angle(kp[RIGHT_HIP], kp[RIGHT_KNEE], kp[RIGHT_ANKLE])
        # Elbow: shoulder -> elbow -> wrist
        angles["left_elbow_angle"][t] = compute_angle(kp[LEFT_SHOULDER], kp[LEFT_ELBOW], kp[LEFT_WRIST])
        angles["right_elbow_angle"][t] = compute_angle(kp[RIGHT_SHOULDER], kp[RIGHT_ELBOW], kp[RIGHT_WRIST])
        # Hip: shoulder -> hip -> knee
        angles["left_hip_angle"][t] = compute_angle(kp[LEFT_SHOULDER], kp[LEFT_HIP], kp[LEFT_KNEE])
        angles["right_hip_angle"][t] = compute_angle(kp[RIGHT_SHOULDER], kp[RIGHT_HIP], kp[RIGHT_KNEE])
    return angles


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
PER_FRAME_HEADER = [
    "person_id", "frame_idx", "frame_name",
    "hip_x", "hip_y", "hip_z",
    "velocity_hip", "accel_hip",
    "velocity_left_ankle", "velocity_right_ankle",
    "left_knee_angle", "right_knee_angle",
    "left_elbow_angle", "right_elbow_angle",
    "left_hip_angle", "right_hip_angle",
    "cumulative_distance_hip",
    "cumulative_distance_left_ankle",
    "cumulative_distance_right_ankle",
]

SUMMARY_HEADER = [
    "person_id",
    "total_distance_hip",
    "total_distance_left_ankle",
    "total_distance_right_ankle",
    "avg_velocity_hip",
    "max_velocity_hip",
    "total_frames",
    "valid_frames",
]


def _fmt(v):
    """Format a value for CSV output."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (np.floating, float)):
        return f"{v:.6f}"
    return str(v)


def process_person(person_id: str, person_dir: str, fps: float) -> tuple:
    """Process one person and return (per_frame_rows, summary_row)."""
    data = load_person_data(person_dir)
    if data is None:
        return [], None

    kp3d = data["keypoints_3d"]
    valid = data["valid_mask"].astype(bool)
    frame_names = data["frame_names"]
    n = len(kp3d)

    # Positions
    hip_center = compute_hip_center(kp3d)
    left_ankle = kp3d[:, LEFT_ANKLE, :]
    right_ankle = kp3d[:, RIGHT_ANKLE, :]

    # Distances
    cum_dist_hip = compute_cumulative_distance(hip_center, valid)
    cum_dist_la = compute_cumulative_distance(left_ankle, valid)
    cum_dist_ra = compute_cumulative_distance(right_ankle, valid)

    # Velocities
    vel_hip = compute_velocity(hip_center, valid, fps)
    vel_la = compute_velocity(left_ankle, valid, fps)
    vel_ra = compute_velocity(right_ankle, valid, fps)

    # Acceleration
    acc_hip = compute_acceleration(vel_hip, valid, fps)

    # Joint angles
    angles = compute_joint_angles(kp3d, valid)

    # Build per-frame rows
    rows = []
    for t in range(n):
        rows.append([
            person_id, t, frame_names[t],
            hip_center[t, 0], hip_center[t, 1], hip_center[t, 2],
            vel_hip[t], acc_hip[t],
            vel_la[t], vel_ra[t],
            angles["left_knee_angle"][t], angles["right_knee_angle"][t],
            angles["left_elbow_angle"][t], angles["right_elbow_angle"][t],
            angles["left_hip_angle"][t], angles["right_hip_angle"][t],
            cum_dist_hip[t], cum_dist_la[t], cum_dist_ra[t],
        ])

    # Summary
    valid_vel = vel_hip[~np.isnan(vel_hip)]
    summary = [
        person_id,
        cum_dist_hip[-1] if n > 0 else 0.0,
        cum_dist_la[-1] if n > 0 else 0.0,
        cum_dist_ra[-1] if n > 0 else 0.0,
        np.mean(valid_vel) if len(valid_vel) > 0 else np.nan,
        np.max(valid_vel) if len(valid_vel) > 0 else np.nan,
        n,
        int(valid.sum()),
    ]

    return rows, summary


def main():
    parser = argparse.ArgumentParser(
        description="Compute sports statistics from SAM-Body4D skeleton exports"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Path to the output directory containing skeleton_4d_individual/"
    )
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="Video frame rate for velocity/acceleration calculations (default: 30)"
    )
    parser.add_argument(
        "--output_csv", type=str, default=None,
        help="Path for per-frame CSV output (default: {input_dir}/sports_stats.csv)"
    )
    args = parser.parse_args()

    skeleton_dir = os.path.join(args.input_dir, "skeleton_4d_individual")
    if not os.path.isdir(skeleton_dir):
        print(f"Error: skeleton directory not found: {skeleton_dir}")
        print("Make sure to run the pipeline with smpl_export: true first.")
        return

    output_csv = args.output_csv or os.path.join(args.input_dir, "sports_stats.csv")
    summary_csv = os.path.join(os.path.dirname(output_csv), "sports_stats_summary.csv")

    # Discover person subdirectories
    person_dirs = sorted([
        d for d in os.listdir(skeleton_dir)
        if os.path.isdir(os.path.join(skeleton_dir, d))
    ])

    if not person_dirs:
        print("No person data found.")
        return

    all_rows = []
    all_summaries = []

    for pid in person_dirs:
        person_path = os.path.join(skeleton_dir, pid)
        print(f"Processing person {pid}...")
        rows, summary = process_person(pid, person_path, args.fps)
        all_rows.extend(rows)
        if summary is not None:
            all_summaries.append(summary)

    # Write per-frame CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(PER_FRAME_HEADER)
        for row in all_rows:
            writer.writerow([_fmt(v) for v in row])
    print(f"Per-frame stats written to: {output_csv}")

    # Write summary CSV
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SUMMARY_HEADER)
        for row in all_summaries:
            writer.writerow([_fmt(v) for v in row])
    print(f"Summary stats written to: {summary_csv}")

    # Print summary to console
    print("\n=== Sports Statistics Summary ===")
    for s in all_summaries:
        print(f"\nPerson {s[0]}:")
        print(f"  Total distance (hip center): {s[1]:.4f}")
        print(f"  Total distance (left ankle): {s[2]:.4f}")
        print(f"  Total distance (right ankle): {s[3]:.4f}")
        print(f"  Avg velocity (hip): {s[4]:.4f}" if not np.isnan(s[4]) else "  Avg velocity (hip): N/A")
        print(f"  Max velocity (hip): {s[5]:.4f}" if not np.isnan(s[5]) else "  Max velocity (hip): N/A")
        print(f"  Frames: {s[7]}/{s[6]} valid")


if __name__ == "__main__":
    main()
