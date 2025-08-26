import argparse
import os
import shutil
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.graph import route_through_array
from skimage.util import invert
from scipy.ndimage import convolve
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import yaml
from datetime import datetime
import subprocess

FRAME_OUTPUT_DIR = "./Tracking/Output/worm_tracks"
AREA_THRESHOLD = 50
MAX_AGE = 35

def get_git_commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "not-a-git-repo"

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 2)
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 3
    )
    return binary

def extract_worm_masks(binary, area_threshold):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    masks = []
    height = binary.shape[0]
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= area_threshold:
            y = stats[i, cv2.CC_STAT_TOP]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            if y + h >= height - 5:
                continue
            mask = (labels == i).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            masks.append(mask)
    return masks

def get_skeleton_points(mask, num_points):
    mask_dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    skeleton = skeletonize(mask_dilated > 0)

    if np.count_nonzero(skeleton) < 2:
        return None

    kernel = np.ones((3, 3), dtype=int)
    neighbor_count = convolve(skeleton.astype(int), kernel, mode='constant') * skeleton
    endpoints = np.column_stack(np.where(neighbor_count == 2))

    if len(endpoints) < 2:
        return None

    start = tuple(endpoints[0])
    end = tuple(endpoints[-1])
    try:
        path, _ = route_through_array(invert(skeleton).astype(np.float32), start, end, fully_connected=True)
    except Exception:
        return None

    path = np.array(path)
    if len(path) < 2:
        return None

    indices = np.linspace(0, len(path) - 1, num=num_points, dtype=int)
    return path[indices]

def compute_cost_matrix(current_pts, prev_pts):
    cost = np.zeros((len(prev_pts), len(current_pts)))
    for i, prev in enumerate(prev_pts):
        for j, curr in enumerate(current_pts):
            centroid_dist = np.linalg.norm(np.mean(prev, axis=0) - np.mean(curr, axis=0))
            shape_dist = np.mean(np.linalg.norm(prev - curr, axis=1))
            cost[i, j] = 0.7 * centroid_dist + 0.3 * shape_dist
    return cost

def draw_tracks(frame, worm_keypoints, worm_ids, keypoints_per_worm):
    keypoint_colors = [
        (255, 0, 0), (255, 64, 0), (255, 128, 0), (255, 191, 0), (255, 255, 0),
        (191, 255, 0), (128, 255, 0), (64, 255, 0), (0, 255, 0), (0, 255, 64),
        (0, 255, 128), (0, 255, 191), (0, 255, 255), (0, 191, 255), (0, 128, 255)
    ]
    for i, points in enumerate(worm_keypoints):
        for k, pt in enumerate(points):
            color = keypoint_colors[k] if k < len(keypoint_colors) else (255, 255, 255)
            x, y = int(pt[1]), int(pt[0])
            cv2.circle(frame, (x, y), 4, color, -1)
            if k > 0:
                pt1 = (int(points[k - 1][1]), int(points[k - 1][0]))
                pt2 = (int(pt[1]), int(pt[0]))
                cv2.line(frame, pt1, pt2, color, 2)
        worm_id = worm_ids[i]
        cv2.putText(frame, f"ID {worm_id}", tuple(points[0][::-1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame

def run_tracking(video_path, video_output_dir, keypoints_per_worm, area_threshold, max_age, show_video, output_name=None):

    if os.path.exists(FRAME_OUTPUT_DIR):
        shutil.rmtree(FRAME_OUTPUT_DIR)
    os.makedirs(FRAME_OUTPUT_DIR, exist_ok=True)
    os.makedirs(video_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    frame_idx = 0
    track_memory = []
    next_id = 0
    keypoint_tracks = {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(total_frames), desc="Processing frames", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        binary = preprocess_frame(frame)
        masks = extract_worm_masks(binary, area_threshold)
        current_keypoints = []

        for mask in masks:
            keypoints = get_skeleton_points(mask, keypoints_per_worm)
            if keypoints is not None:
                current_keypoints.append(keypoints)

        current_ids = [-1] * len(current_keypoints)

        if len(track_memory) > 0:
            prev_keypoints = [t["keypoints"] for t in track_memory if t["age"] <= max_age]
            prev_ids = [t["id"] for t in track_memory if t["age"] <= max_age]
            cost = compute_cost_matrix(current_keypoints, prev_keypoints)
            row_ind, col_ind = linear_sum_assignment(cost)

            for i, j in zip(row_ind, col_ind):
                if i < len(prev_keypoints) and j < len(current_keypoints):
                    if cost[i, j] < 80:
                        prev_vec = prev_keypoints[i][-1] - prev_keypoints[i][0]
                        curr_vec = current_keypoints[j][-1] - current_keypoints[j][0]
                        if np.dot(prev_vec, curr_vec) < 0:
                            current_keypoints[j] = current_keypoints[j][::-1]
                        current_ids[j] = prev_ids[i]

        for j in range(len(current_keypoints)):
            if current_ids[j] == -1:
                curr_centroid = np.mean(current_keypoints[j], axis=0)
                too_close = False
                for track in track_memory:
                    prev_centroid = np.mean(track["keypoints"], axis=0)
                    if np.linalg.norm(curr_centroid - prev_centroid) < 50:
                        too_close = True
                        break
                if not too_close:
                    current_ids[j] = next_id
                    next_id += 1
                else:
                    current_keypoints[j] = None

        filtered_ids = []
        filtered_keypoints = []
        for cid, kp in zip(current_ids, current_keypoints):
            if kp is not None and cid != -1:
                filtered_ids.append(cid)
                filtered_keypoints.append(kp)
        current_ids = filtered_ids
        current_keypoints = filtered_keypoints

        updated_tracks = []
        for tid, kps in zip(current_ids, current_keypoints):
            updated_tracks.append({"id": tid, "keypoints": kps, "age": 0})

        for old_track in track_memory:
            if old_track["id"] not in current_ids and old_track["age"] < max_age:
                updated_tracks.append({"id": old_track["id"], "keypoints": old_track["keypoints"], "age": old_track["age"] + 1})

        track_memory = updated_tracks

        for worm_id, keypoints in zip(current_ids, current_keypoints):
            if worm_id not in keypoint_tracks:
                keypoint_tracks[worm_id] = [[] for _ in range(keypoints_per_worm)]
            for i in range(keypoints_per_worm):
                keypoint_tracks[worm_id][i].append(keypoints[i])

        annotated = draw_tracks(frame.copy(), current_keypoints, current_ids, keypoints_per_worm)
        cv2.imwrite(os.path.join(FRAME_OUTPUT_DIR, f"frame_{frame_idx:04d}.png"), annotated)
        frame_idx += 1

    cap.release()

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_name and not output_name.lower().endswith(".mp4"):
        output_name += ".mp4"

    output_video_filename = output_name if output_name else f"{base_name}.mp4"
    output_video_path = os.path.join(video_output_dir, output_video_filename)

    image_files = sorted([f for f in os.listdir(FRAME_OUTPUT_DIR) if f.endswith(".png")])
    if not image_files:
        print("No frames saved. No video generated.")
        return

    first_image = cv2.imread(os.path.join(FRAME_OUTPUT_DIR, image_files[0]))
    height, width, _ = first_image.shape
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 60, (width, height))

    for filename in tqdm(image_files, desc="Generating video", unit="frame"):
        frame = cv2.imread(os.path.join(FRAME_OUTPUT_DIR, filename))
        out.write(frame)

    out.release()
    print(f"Tracking complete.")
    print(f"Frames saved in: {FRAME_OUTPUT_DIR}")
    print(f"Video saved as: {output_video_path}")

    # Save tracking metadata to YAML
    metadata = {
        "git_version": get_git_commit_hash(),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input": video_path,
        "parameters": {
            "keypoints": keypoints_per_worm,
            "min_area": area_threshold,
            "max_age": max_age
        },
        "output_video": output_video_path,
        "total_frames": frame_idx
    }
    metadata_path = os.path.join(video_output_dir, f"tracking_metadata.yaml")
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f)
    print(f"Metadata saved at: {metadata_path}")

    # Save worm keypoints to .npz (per worm: [frame][keypoint][y,x])
    keypoints_npz_path = os.path.join(video_output_dir, f"worm_keypoints.npz")
    np.savez_compressed(keypoints_npz_path,
    **{str(worm_id): np.array(frames) for worm_id, frames in keypoint_tracks.items()})

    print(f"Worm keypoints saved at: {keypoints_npz_path}")
   

    if show_video:
        try:
            os.startfile(output_video_path)
        except Exception as e:
            print(f"Could not open video: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Worm tracking from video using skeleton-based interpolation.\n\nExample:\n  python worm_tracker.py input.mov output_dir --keypoints 15 --min-area 50 --max-age 35",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('video_path', type=str, help='Path to input video')
    parser.add_argument('video_output_dir', type=str, help='Directory to save output video')
    parser.add_argument('--keypoints', type=int, default=15, help='Number of keypoints per worm (default: 15)')
    parser.add_argument('--min-area', type=int, default=50, help='Minimum area of worm region (default: 50)')
    parser.add_argument('--max-age', type=int, default=35, help='Maximum age to track missing worms (default: 35)')
    parser.add_argument('--show', action='store_true', help='Display the output video after processing')
    parser.add_argument('--output-name', type=str, default=None, help='Custom name for the output video file (e.g., output.mp4)')

    args = parser.parse_args()
    run_tracking(args.video_path, args.video_output_dir, args.keypoints, args.min_area, args.max_age, args.show, args.output_name)


if __name__ == "__main__":
    main()
