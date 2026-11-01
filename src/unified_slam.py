import os
import glob
import shutil
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
import cv2
import time
import argparse
from tqdm import tqdm

print("--- Initializing Unified SLAM Script ---")

# ==============================================================================
# --- 1. CONFIGURATION PARAMETERS ---
# ==============================================================================

CONFIG = {
    # Data Paths
    'CSV_INPUT_DIR': 'Combined_RVP_csv', # Directory containing timestamped CSV LiDAR scans
    'GROUND_TRUTH_TRAJ_PATH': 'campus_train0_gt.txt', # Ground truth trajectory file (TUM format)
    'OUTPUT_DIR': 'slam_results_unified', # Directory for saving results (estimated trajectory, map, plots)

    # SLAM Parameters
    'NUM_SCANS_TO_PROCESS': None, # Set to an integer to limit scans, None for all
    'VOXEL_DOWNSAMPLE_SIZE': 0.2, # Voxel size for downsampling individual scans before ICP
    'MAP_VOXEL_DOWNSAMPLE_SIZE': 0.4, # Voxel size for downsampling the accumulated map
    'ICP_MAX_CORRESPONDENCE_DISTANCE': 0.8, # Max distance for point correspondences in ICP
    'ICP_MAX_ITERATIONS': 50, # Max iterations for ICP convergence
    'ICP_FITNESS_THRESHOLD': 0.6, # Min fitness score for successful ICP registration

    # Visualization Parameters
    'MAX_PLOT_POINTS_3D': 750000, # Max points to plot in 3D visualization for performance
    'ANIMATION_FPS': 10, # Frames per second for 2D trajectory animation
    'SAVE_ANIMATION_VIDEO': False, # Whether to save the 2D animation as a GIF
    'LIVE_MODE_DELAY_SEC': 0.1, # Delay between processing scans in live simulation mode

    # 2D Grid Mapping Parameters
    'MAP_RESOLUTION': 0.1, # Resolution of the 2D occupancy grid map (meters/pixel)
    'MAP_PADDING': 5.0, # Padding around the trajectory for map boundaries
    'LOG_ODDS_OCC': np.log(0.85 / (1 - 0.85)), # Log-odds increment for occupied cells
    'LOG_ODDS_FREE': np.log(0.4 / (1 - 0.4)), # Log-odds decrement for free cells
    'LOG_ODDS_MAX': 5.0, # Max log-odds value
    'LOG_ODDS_MIN': -5.0, # Min log-odds value
    'GRID_MAP_MIN_HEIGHT': 0.2, # Min height for points to be considered in 2D grid map
    'GRID_MAP_MAX_HEIGHT': 1.5, # Max height for points to be considered in 2D grid map
}

# Ensure output directory exists
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

# Define output filenames based on CONFIG
ESTIMATED_TRAJ_FILENAME = os.path.join(CONFIG['OUTPUT_DIR'], "estimated_trajectory_icp.txt")
MAP_FILENAME = os.path.join(CONFIG['OUTPUT_DIR'], "icp_map.pcd")
TRAJ_COMPARISON_PLOT_FILENAME = os.path.join(CONFIG['OUTPUT_DIR'], "trajectory_comparison_plot.png")
ANIMATION_VIDEO_FILENAME = os.path.join(CONFIG['OUTPUT_DIR'], "full_slam_visualization.gif")

print("Configuration set.")
print("-" * 60 + "\n")

# ==============================================================================
# --- 2. HELPER FUNCTIONS ---
# ==============================================================================

def load_csv_as_point_cloud(csv_filepath):
    """Reads a CSV file and converts it to an Open3D PointCloud object."""
    try:
        df = pd.read_csv(csv_filepath)
        if not all(col in df.columns for col in ['x', 'y', 'z']):
            print(f"Warning: CSV {os.path.basename(csv_filepath)} is missing x, y, or z columns. Skipping.")
            return None
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(df[['x', 'y', 'z']].values)
        return pcd
    except Exception as e:
        print(f"Error loading CSV {os.path.basename(csv_filepath)}: {e}")
        return None

def matrix_to_tum_format(timestamp, matrix):
    """Converts a 4x4 transformation matrix to a TUM format string."""
    t = matrix[:3, 3]
    rot_matrix = matrix[:3, :3]
    try:
        q = R.from_matrix(rot_matrix).as_quat()
    except ValueError:
        # Fallback to identity quaternion if rotation matrix is invalid
        q = np.array([0, 0, 0, 1])
    return f"{timestamp} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}"

def read_tum_trajectory(filepath, num_poses=None):
    """
    Reads a trajectory file in TUM format (timestamp tx ty tz qx qy qz qw)
    and returns timestamps and a list of 4x4 transformation matrices.
    """
    try:
        data = np.loadtxt(filepath, dtype=float, delimiter=' ')
        if num_poses is not None and len(data) > num_poses:
            print(f"INFO: Reading the first {num_poses} poses from {os.path.basename(filepath)}.")
            data = data[:num_poses]
        else:
            print(f"INFO: Reading all {len(data)} poses from {os.path.basename(filepath)}.")

        timestamps = data[:, 0]
        translations = data[:, 1:4]
        quaternions = data[:, 4:8]

        poses = []
        for t, q in zip(translations, quaternions):
            transform_matrix = np.identity(4)
            transform_matrix[:3, :3] = R.from_quat(q).as_matrix()
            transform_matrix[:3, 3] = t
            poses.append(transform_matrix)

        return timestamps, poses
    except FileNotFoundError:
        print(f"🚨 ERROR: Trajectory file not found at '{filepath}'")
        return None, None
    except Exception as e:
        print(f"🚨 ERROR: Failed to read or parse {os.path.basename(filepath)}: {e}")
        return None, None

def preprocess_scan(scan_pcd, voxel_downsample_size):
    """Downsamples a point cloud and estimates its normals."""
    down_pcd = scan_pcd.voxel_down_sample(voxel_downsample_size)
    down_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    return down_pcd

# ==============================================================================
# --- 3. SLAM PIPELINE (ICP ODOMETRY) ---
# ==============================================================================

def run_icp_slam_pipeline(csv_input_dir, num_scans_to_process, live_mode=False, live_delay=0.1):
    """
    Executes the ICP-based SLAM pipeline.
    Can run in batch mode (loading all scans) or live simulation mode (processing one by one).
    """
    print("--- Starting SLAM Pipeline ---")

    if not os.path.isdir(csv_input_dir):
        print(f"⚠️ Error: CSV input directory '{csv_input_dir}' not found. Aborting SLAM.")
        return [], None

    csv_file_paths = sorted(glob.glob(os.path.join(csv_input_dir, "*.csv")))
    if not csv_file_paths:
        print(f"⚠️ Error: No CSV files found in '{csv_input_dir}'. Aborting SLAM.")
        return [], None

    if num_scans_to_process and len(csv_file_paths) > num_scans_to_process:
        print(f"Limiting processing to the first {num_scans_to_process} scans.")
        csv_file_paths = csv_file_paths[:num_scans_to_process]

    total_scans = len(csv_file_paths)
    print(f"Found {total_scans} CSV files to process.")

    # Initialization
    global_poses = []
    map_accumulator = o3d.geometry.PointCloud()
    previous_relative_transform = np.identity(4)
    estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()

    # Live mode visualization setup
    if live_mode:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Live SLAM Map", width=1280, height=720)
        # Add initial coordinate frame
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(origin_frame)
        # Add a placeholder for the map
        vis.add_geometry(map_accumulator)
        view_control = vis.get_view_control()
        # Set a reasonable initial camera view for live mode
        view_control.set_lookat([0, 0, 0])
        view_control.set_up([0, 0, 1])
        view_control.set_front([0, -1, 0])
        view_control.set_zoom(0.5)

    for i, csv_file in enumerate(tqdm(csv_file_paths, desc="Processing Scans")):
        current_timestamp = float(os.path.basename(csv_file).replace('scan_', '').replace('.csv', ''))

        # Load current scan
        current_scan_original = load_csv_as_point_cloud(csv_file)
        if current_scan_original is None or not current_scan_original.has_points():
            print(f"Warning: Skipping empty or invalid scan: {csv_file}")
            continue

        current_scan_downsampled = preprocess_scan(current_scan_original, CONFIG['VOXEL_DOWNSAMPLE_SIZE'])

        if i == 0:
            # Initialize with the first scan
            initial_pose = np.identity(4)
            global_poses.append(initial_pose)
            map_accumulator = o3d.geometry.PointCloud(current_scan_original)
            map_accumulator.transform(initial_pose)
            if live_mode:
                vis.update_geometry(map_accumulator)
                vis.poll_events()
                vis.update_renderer()
            print(f"Initialized SLAM state with first scan ({csv_file}).")
            continue # Move to the next scan

        # Previous scan is the target, current is the source
        target_scan_downsampled = preprocess_scan(global_poses[-1].T @ map_accumulator, CONFIG['VOXEL_DOWNSAMPLE_SIZE'])
        # A simpler way to get a target for ICP from the previous pose and current map
        # is to transform the current scan by the inverse of the last pose and use it as source
        # and the accumulated map as target. However, the original code uses previous scan as target.
        # Let's stick to the original logic of ICP between two consecutive scans for relative motion.
        # So, target_scan_downsampled should be the *previous* downsampled scan.
        # This requires storing the previous downsampled scan.

        # Let's adjust the logic slightly to align with the original ICP script:
        # ICP aligns current_scan (source) to previous_scan (target).
        # We need to keep track of the previous original and downsampled scans.

        # For the first iteration (i=1, after i=0 initialization):
        # target_scan_down = all_scans_downsampled[0]
        # source_scan_down = all_scans_downsampled[1]

        # To handle this in a streaming fashion, we need to store the previous scan.
        if i == 1:
            prev_scan_original = o3d.geometry.PointCloud(global_poses[0].T @ map_accumulator) # This is the first scan in its own frame
            prev_scan_downsampled = preprocess_scan(prev_scan_original, CONFIG['VOXEL_DOWNSAMPLE_SIZE'])
        else:
            # For subsequent iterations, the previous_scan is the one from the last loop
            pass # prev_scan_downsampled is already set from the end of the previous loop

        # Perform ICP registration
        trans_init = previous_relative_transform # Use previous relative transform as initial guess
        try:
            reg_icp = o3d.pipelines.registration.registration_icp(
                source=current_scan_downsampled,
                target=prev_scan_downsampled, # Align current to previous
                max_correspondence_distance=CONFIG['ICP_MAX_CORRESPONDENCE_DISTANCE'],
                init=trans_init,
                estimation_method=estimation_method,
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=CONFIG['ICP_MAX_ITERATIONS'])
            )

            if reg_icp.fitness > CONFIG['ICP_FITNESS_THRESHOLD']:
                current_relative_transform = reg_icp.transformation
                previous_relative_transform = current_relative_transform
            else:
                print(f"Warning: ICP fitness too low ({reg_icp.fitness:.4f}) at scan {i}. Assuming no motion.")
                current_relative_transform = np.identity(4)

        except Exception as e:
            print(f"ICP registration failed at step {i}: {e}. Assuming no motion.")
            current_relative_transform = np.identity(4)

        # Update global pose
        current_global_pose = global_poses[-1] @ current_relative_transform
        global_poses.append(current_global_pose)

        # Add transformed current scan to map accumulator
        current_scan_transformed = o3d.geometry.PointCloud(current_scan_original)
        current_scan_transformed.transform(current_global_pose)
        map_accumulator += current_scan_transformed

        # Periodically downsample the map
        if (i + 1) % 30 == 0:
            map_accumulator = map_accumulator.voxel_down_sample(CONFIG['MAP_VOXEL_DOWNSAMPLE_SIZE'])

        if (i + 1) % 25 == 0 or (i + 1) == total_scans - 1:
            print(f"Processed scan {i+1}/{total_scans}. Fitness: {reg_icp.fitness:.4f}, RMSE: {reg_icp.inlier_rmse:.4f}")

        # Update for next iteration: current scan becomes previous scan
        prev_scan_downsampled = current_scan_downsampled

        if live_mode:
            # Update visualization
            vis.update_geometry(map_accumulator)
            # Update camera to follow robot
            current_translation = current_global_pose[:3, 3]
            view_control.set_lookat(current_translation)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(live_delay)

    # Final downsampling of the map
    map_accumulator = map_accumulator.voxel_down_sample(CONFIG['MAP_VOXEL_DOWNSAMPLE_SIZE'])
    print("\nLidar Odometry processing finished.")

    # Close live visualization window if open
    if live_mode:
        vis.destroy_window()

    # Save Trajectory and Map
    print("\n--- Saving Final Trajectory and Map ---")
    with open(ESTIMATED_TRAJ_FILENAME, 'w') as f:
        # Use a simple sequential timestamp if actual timestamps are not available from CSV names
        # For the RVP dataset, filenames contain timestamps, so we can extract them.
        # Let's adjust to use the actual timestamps from the filenames for saving.
        for idx, pose_matrix in enumerate(global_poses):
            if idx < len(csv_file_paths):
                timestamp_str = os.path.basename(csv_file_paths[idx]).replace('scan_', '').replace('.csv', '')
                timestamp = float(timestamp_str)
            else:
                timestamp = float(idx) # Fallback if global_poses somehow has more entries than csv_file_paths
            f.write(matrix_to_tum_format(timestamp, pose_matrix) + '\n')

    o3d.io.write_point_cloud(MAP_FILENAME, map_accumulator)
    print(f"Results saved successfully to '{CONFIG['OUTPUT_DIR']}' directory.")
    print("-" * 60 + "\n")

    return global_poses, map_accumulator

# ==============================================================================
# --- 4. TRAJECTORY EVALUATION ---
# ==============================================================================

def run_trajectory_evaluation(estimated_traj_path, ground_truth_traj_path, output_dir):
    """Compares the estimated trajectory with the ground truth."""
    print("--- Starting Trajectory Evaluation ---")

    if not os.path.exists(estimated_traj_path):
        print(f"⚠️ Error: Estimated trajectory not found at '{estimated_traj_path}'. Cannot evaluate.")
        return
    if not os.path.exists(ground_truth_traj_path):
        print(f"⚠️ Error: Ground truth trajectory not found at '{ground_truth_traj_path}'. Cannot evaluate.")
        return

    # Read trajectory data
    # Determine the number of poses from the estimated trajectory to ensure consistent comparison
    with open(estimated_traj_path, 'r') as f:
        num_poses_est = sum(1 for line in f if line.strip() and not line.strip().startswith('#'))

    print(f"Reading {num_poses_est} poses from estimated and ground truth trajectories.")
    est_timestamps, est_poses = read_tum_trajectory(estimated_traj_path, num_poses=num_poses_est)
    gt_timestamps, gt_poses = read_tum_trajectory(ground_truth_traj_path, num_poses=num_poses_est)

    if est_poses is None or gt_poses is None or len(est_poses) != len(gt_poses) or not est_poses:
        print("Could not load trajectories or pose counts do not match. Aborting evaluation.")
        return

    # Align Trajectories (Horn's method via ICP)
    print("\nAligning estimated trajectory to ground truth...")
    est_points = np.array([pose[:3, 3] for pose in est_poses])
    gt_points = np.array([pose[:3, 3] for pose in gt_poses])

    est_pcd = o3d.geometry.PointCloud()
    est_pcd.points = o3d.utility.Vector3dVector(est_points)
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_points)

    # Use ICP to find the rigid body transformation (rotation, translation)
    # that best aligns the estimated trajectory to the ground truth.
    # A large threshold (e.g., 1.0) is used as we are aligning trajectories (sequences of points)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        est_pcd, gt_pcd, 1.0, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    alignment_transform = reg_p2p.transformation
    aligned_est_poses = [alignment_transform @ pose for pose in est_poses]
    print("Alignment complete.")

    # Compute Absolute Trajectory Error (ATE)
    print("\nComputing Absolute Trajectory Error (ATE)...")
    errors = []
    for aligned_est_pose, gt_pose in zip(aligned_est_poses, gt_poses):
        error_trans = aligned_est_pose[:3, 3] - gt_pose[:3, 3]
        errors.append(np.linalg.norm(error_trans))

    ate_rmse = np.sqrt(np.mean(np.square(errors)))
    print(f"✅ ATE (RMSE): {ate_rmse:.4f} meters")

    # Compute Accuracy as a Percentage of Trajectory Length
    if len(gt_points) > 1:
        gt_start_point = gt_points[0]
        displacements = np.linalg.norm(gt_points - gt_start_point, axis=1)
        max_gt_displacement = np.max(displacements)

        if max_gt_displacement > 0:
            ate_percentage = (ate_rmse / max_gt_displacement) * 100
            accuracy_percentage = max(0, 100 - ate_percentage)
            print(f"   - ATE as % of Max GT Displacement: {ate_percentage:.2f}%")
            print(f"   - Estimated Accuracy: {accuracy_percentage:.2f}%")

    # Compute Relative Trajectory Error (RTE)
    print("\nComputing Relative Trajectory Error (RTE)...")
    relative_errors = []
    for i in range(len(aligned_est_poses) - 1):
        rel_est = np.linalg.inv(aligned_est_poses[i]) @ aligned_est_poses[i+1]
        rel_gt = np.linalg.inv(gt_poses[i]) @ gt_poses[i+1]
        error_relative_transform = np.linalg.inv(rel_gt) @ rel_est
        relative_errors.append(np.linalg.norm(error_relative_transform[:3, 3]))

    if relative_errors:
        rte_rmse = np.sqrt(np.mean(np.square(relative_errors)))
        print(f"✅ RTE (RMSE over adjacent poses): {rte_rmse:.4f} meters")

    # Plot Trajectories
    print("\nPlotting trajectories for comparison...")
    aligned_est_points = np.array([pose[:3, 3] for pose in aligned_est_poses])

    plt.figure(figsize=(12, 10))
    plt.plot(aligned_est_points[:, 0], aligned_est_points[:, 1], 'r--', label='Estimated Trajectory (Aligned)', linewidth=2)
    plt.plot(gt_points[:, 0], gt_points[:, 1], 'b-', label='Ground Truth Trajectory', linewidth=2)
    plt.plot(aligned_est_points[0, 0], aligned_est_points[0, 1], 'go', markersize=10, label='Start')
    plt.plot(aligned_est_points[-1, 0], aligned_est_points[-1, 1], 'rx', markersize=10, mew=2, label='End (Estimated)')
    plt.plot(gt_points[-1, 0], gt_points[-1, 1], 'bx', markersize=10, mew=2, label='End (Ground Truth)')

    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Estimated vs. Ground Truth Trajectory (After Alignment)')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')

    plt.savefig(TRAJ_COMPARISON_PLOT_FILENAME)
    print(f"📈 Plot saved to '{TRAJ_COMPARISON_PLOT_FILENAME}'")
    plt.show()

    print("\n--- Trajectory Evaluation Complete ---")
    print("-" * 60 + "\n")

# ==============================================================================
# --- 5. VISUALIZATION FUNCTIONS ---
# ==============================================================================

def run_3d_map_visualization(final_map, estimated_poses):
    """Displays the final generated 3D point cloud map."""
    if not final_map or not final_map.has_points():
        print("No final map to display.")
        return

    print("--- Displaying Final Generated 3D Map ---")
    print("A new window will open with the 3D map. Close it to end the script.")

    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    # Add the estimated trajectory as a line set
    if estimated_poses:
        points = np.array([pose[:3, 3] for pose in estimated_poses])
        lines = [[i, i+1] for i in range(len(points)-1)]
        colors = [[1, 0, 0] for _ in range(len(lines))] # Red color for trajectory
        traj_line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        traj_line_set.colors = o3d.utility.Vector3dVector(colors)
        geometries = [final_map, origin_frame, traj_line_set]
    else:
        geometries = [final_map, origin_frame]

    try:
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Final ICP SLAM Map",
            width=1280,
            height=720
        )
    except Exception as e:
        print(f"Could not open 3D visualization window: {e}")
        print(f"The final map has been saved to '{MAP_FILENAME}' for local viewing with a PCD viewer.")

def run_2d_animation_visualization(csv_input_dir, estimated_traj_path, save_video=False, video_filename="animation.gif"):
    """
    Creates and displays a 2D animation of the SLAM process.
    Aggregates scans and shows robot movement over time.
    """
    print("\n--- Preparing 2D Animation Visualization ---")

    est_poses, est_translations = read_tum_trajectory(estimated_traj_path)
    all_csv_files = sorted([os.path.join(csv_input_dir, f) for f in os.listdir(csv_input_dir) if f.endswith('.csv')])

    num_poses = len(est_poses) if est_poses else 0
    num_scans = len(all_csv_files)
    num_frames = min(num_poses, num_scans)

    if num_frames == 0:
        print("CRITICAL ERROR: Could not find any scans or poses for animation. Aborting.")
        return

    print(f"\n--- Data Counts for Animation ---")
    print(f"Found {num_poses} estimated poses.")
    print(f"Found {num_scans} CSV scan files.")
    print(f"--> Running animation for {num_frames} frames (the smallest available count).")

    est_poses = est_poses[:num_frames]
    est_translations = est_translations[:num_frames]
    csv_files = all_csv_files[:num_frames]

    print(f"\n--- Aggregating all {num_frames} LiDAR scans for background map. This will take some time... ---")
    all_points_list = []
    for i in tqdm(range(num_frames), desc="Aggregating Scans"):
        pcd = load_csv_as_point_cloud(csv_files[i])
        if pcd and not pcd.is_empty():
            pcd = pcd.voxel_down_sample(voxel_size=CONFIG['VOXEL_DOWNSAMPLE_SIZE']) # Use smaller voxel for animation background
            pcd.transform(est_poses[i])
            all_points_list.append(np.asarray(pcd.points))

    full_point_cloud = np.concatenate(all_points_list, axis=0) if all_points_list else np.array([])
    print(f"--- Full point cloud map created with {len(full_point_cloud)} points. ---")

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor('black')

    if len(full_point_cloud) > CONFIG['MAX_PLOT_POINTS_3D']:
        print(f"--- Subsampling background points to {CONFIG['MAX_PLOT_POINTS_3D']} for performance ---")
        idx = np.random.choice(len(full_point_cloud), size=CONFIG['MAX_PLOT_POINTS_3D'], replace=False)
        plot_points = full_point_cloud[idx]
    else:
        plot_points = full_point_cloud

    if plot_points.size > 0:
        ax.scatter(plot_points[:, 0], plot_points[:, 1], s=0.1, c='grey', alpha=0.5)

    est_line, = ax.plot([], [], 'r-', linewidth=2, label='Estimated Trajectory')
    robot_marker, = ax.plot([], [], 'go', markersize=10, label='Robot Position')

    ax.legend()
    ax.set_title("SLAM Visualization - Frame 0")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.axis('equal')

    def animate(i):
        est_line.set_data(est_translations[:i+1, 0], est_translations[:i+1, 1])
        current_pose = est_poses[i]
        robot_marker.set_data([current_pose[0, 3]], [current_pose[1, 3]])
        ax.set_title(f"SLAM Visualization - Frame {i+1}/{num_frames}")
        return [est_line, robot_marker]

    anim = FuncAnimation(
        fig,
        animate,
        frames=num_frames,
        blit=False,
        interval=1000/CONFIG['ANIMATION_FPS'],
        repeat=False
    )

    if save_video:
        if not os.path.exists(CONFIG['OUTPUT_DIR']):
            os.makedirs(CONFIG['OUTPUT_DIR'])
        print(f"\n--- Saving animation to '{video_filename}' (this will be slow for many frames)... ---")
        pbar = tqdm(total=num_frames, desc="Saving Video")
        anim.save(video_filename, writer='pillow', fps=CONFIG['ANIMATION_FPS'], dpi=150, progress_callback=lambda i, n: pbar.update(1))
        pbar.close()
        print(f"--- Animation saved successfully to '{video_filename}'. ---")
    else:
        # Use a non-Agg backend for interactive display
        plt.show()
        print("\n--- Displaying live animation window. Close the window to exit. ---")

# ==============================================================================
# --- 6. 2D OCCUPANCY GRID MAPPING ---
# ==============================================================================

def run_2d_grid_mapping(csv_input_dir, estimated_traj_path, output_dir):
    """Generates and visualizes a 2D occupancy grid map."""
    print("\n--- Starting 2D Occupancy Grid Mapping ---")

    # Load estimated trajectory
    est_timestamps, est_poses = read_tum_trajectory(estimated_traj_path)
    if est_poses is None or not est_poses:
        print("Error: Could not load estimated trajectory for grid mapping. Aborting.")
        return

    # Build a map of available scan files for robust loading
    scan_file_map = {}
    try:
        all_scan_files = os.listdir(csv_input_dir)
        for filename in all_scan_files:
            if filename.startswith('scan_') and filename.endswith('.csv'):
                timestamp_from_file = filename[5:-4]
                scan_file_map[timestamp_from_file] = os.path.join(csv_input_dir, filename)
        print(f"Created a map of {len(scan_file_map)} scan files.")
    except FileNotFoundError:
        print(f"ERROR: The scan directory '{csv_input_dir}' was not found. Aborting grid mapping.")
        return

    def load_lidar_scan_robust(timestamp_from_pose):
        """
        Loads a Lidar scan by finding the best matching key in the file map.
        This handles differences in decimal precision between trajectory and filenames.
        """
        for file_timestamp, path in scan_file_map.items():
            if timestamp_from_pose.startswith(file_timestamp):
                try:
                    df = pd.read_csv(path, usecols=['x', 'y', 'z'])
                    return df.values
                except Exception as e:
                    print(f"Error reading scan file {path}: {e}")
                    return None
        return None

    # Determine map boundaries from trajectory
    positions = np.array([p[:3, 3][:2] for p in est_poses]) # Only X, Y
    min_x, min_y = positions.min(axis=0) - CONFIG['MAP_PADDING']
    max_x, max_y = positions.max(axis=0) + CONFIG['MAP_PADDING']
    map_width = int(np.ceil((max_x - min_x) / CONFIG['MAP_RESOLUTION']))
    map_height = int(np.ceil((max_y - min_y) / CONFIG['MAP_RESOLUTION']))

    occupancy_grid = np.zeros((map_height, map_width), dtype=np.float32)
    print(f"Map initialized with size: {map_width} x {map_height} pixels")

    def world_to_map(x, y):
        """Converts world coordinates (arrays) to map coordinates (arrays)."""
        map_x = np.floor((x - min_x) / CONFIG['MAP_RESOLUTION']).astype(int)
        map_y = np.floor((y - min_y) / CONFIG['MAP_RESOLUTION']).astype(int)
        return map_x, map_y

    print("\n--- Starting 2D map generation...")
    for i, global_pose in enumerate(tqdm(est_poses, desc="Building Grid Map")):
        timestamp_str = str(est_timestamps[i]) # Use timestamp from trajectory for lookup
        scan_points_local = load_lidar_scan_robust(timestamp_str)

        if scan_points_local is None:
            # print(f"Warning: Skipping scan for timestamp {timestamp_str} (not found or invalid).")
            continue

        # Transform local scan points to global coordinates
        # Rotation part of the global_pose
        rotation_matrix = global_pose[:3, :3]
        # Translation part of the global_pose
        translation_vector = global_pose[:3, 3]

        # Apply rotation first, then translation
        global_points = (rotation_matrix @ scan_points_local.T).T + translation_vector

        # Filter points by height
        height_filter = (global_points[:, 2] > CONFIG['GRID_MAP_MIN_HEIGHT']) & \
                        (global_points[:, 2] < CONFIG['GRID_MAP_MAX_HEIGHT'])
        filtered_points = global_points[height_filter]

        if filtered_points.size == 0:
            continue

        # Convert robot position and filtered points to map coordinates
        sensor_map_x, sensor_map_y = world_to_map(translation_vector[0], translation_vector[1])
        points_map_x, points_map_y = world_to_map(filtered_points[:, 0], filtered_points[:, 1])

        # Update free space (ray tracing)
        for mx, my in zip(points_map_x, points_map_y):
            # Ensure line drawing coordinates are within map bounds
            if 0 <= sensor_map_x < map_width and 0 <= sensor_map_y < map_height and \
               0 <= mx < map_width and 0 <= my < map_height:
                # cv2.line modifies in place and takes (col, row) for points
                cv2.line(occupancy_grid, (sensor_map_x, sensor_map_y), (mx, my),
                         color=CONFIG['LOG_ODDS_FREE'], thickness=1)

        # Update occupied cells
        valid_points_mask = (points_map_x >= 0) & (points_map_x < map_width) & \
                            (points_map_y >= 0) & (points_map_y < map_height)
        occupancy_grid[points_map_y[valid_points_mask], points_map_x[valid_points_mask]] += CONFIG['LOG_ODDS_OCC']

        # Clip log odds values
        np.clip(occupancy_grid, CONFIG['LOG_ODDS_MIN'], CONFIG['LOG_ODDS_MAX'], out=occupancy_grid)

    print("2D Map generation complete!")

    # Visualize the final map
    print("\n--- Displaying the final 2D Occupancy Grid Map ---")
    def log_odds_to_prob(log_odds_grid):
        return 1.0 - (1.0 / (1.0 + np.exp(log_odds_grid)))

    map_for_display = log_odds_to_prob(occupancy_grid)
    # Set unknown areas (log_odds close to 0) to a neutral gray (0.5 probability)
    map_for_display[np.isclose(occupancy_grid, 0.0)] = 0.5

    plt.figure(figsize=(15, 15))
    plt.imshow(map_for_display, cmap='gray_r', origin='lower')
    plt.title("Generated 2D Occupancy Grid Map")
    plt.xlabel("Map X (pixels)")
    plt.ylabel("Map Y (pixels)")
    plt.show()
    print("-" * 60 + "\n")

# ==============================================================================
# --- 7. MAIN EXECUTION BLOCK ---
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified LiDAR SLAM Project Script")
    parser.add_argument('--mode', type=str, default='dataset',
                        choices=['dataset', 'live'],
                        help="Operation mode: 'dataset' for batch processing, 'live' for simulated live processing.")
    parser.add_argument('--csv_dir', type=str, default=CONFIG['CSV_INPUT_DIR'],
                        help="Path to the directory containing LiDAR CSV scans.")
    parser.add_argument('--gt_path', type=str, default=CONFIG['GROUND_TRUTH_TRAJ_PATH'],
                        help="Path to the ground truth trajectory file (TUM format).")
    parser.add_argument('--num_scans', type=int, default=CONFIG['NUM_SCANS_TO_PROCESS'],
                        help="Number of scans to process (None for all).")
    parser.add_argument('--no_eval', action='store_true',
                        help="Skip trajectory evaluation.")
    parser.add_argument('--no_3d_vis', action='store_true',
                        help="Skip final 3D map visualization.")
    parser.add_argument('--no_2d_anim', action='store_true',
                        help="Skip 2D animation visualization.")
    parser.add_argument('--no_grid_map', action='store_true',
                        help="Skip 2D occupancy grid mapping.")
    parser.add_argument('--save_anim_video', action='store_true',
                        help="Save 2D animation as a GIF (only if --no_2d_anim is not set).")
    parser.add_argument('--live_delay', type=float, default=CONFIG['LIVE_MODE_DELAY_SEC'],
                        help="Delay in seconds between processing scans in live mode.")

    args = parser.parse_args()

    # Update CONFIG with command-line arguments
    CONFIG['CSV_INPUT_DIR'] = args.csv_dir
    CONFIG['GROUND_TRUTH_TRAJ_PATH'] = args.gt_path
    CONFIG['NUM_SCANS_TO_PROCESS'] = args.num_scans
    CONFIG['SAVE_ANIMATION_VIDEO'] = args.save_anim_video
    CONFIG['LIVE_MODE_DELAY_SEC'] = args.live_delay

    # Ensure output directory exists based on updated config
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

    print(f"\n--- Running in '{args.mode}' mode ---")

    # --- Run SLAM Pipeline ---
    global_poses, final_map = run_icp_slam_pipeline(
        csv_input_dir=CONFIG['CSV_INPUT_DIR'],
        num_scans_to_process=CONFIG['NUM_SCANS_TO_PROCESS'],
        live_mode=(args.mode == 'live'),
        live_delay=CONFIG['LIVE_MODE_DELAY_SEC']
    )

    if not global_poses:
        print("SLAM pipeline did not produce any poses. Skipping subsequent steps.")
        return

    # --- Run Evaluation against Ground Truth ---
    if not args.no_eval:
        run_trajectory_evaluation(
            estimated_traj_path=ESTIMATED_TRAJ_FILENAME,
            ground_truth_traj_path=CONFIG['GROUND_TRUTH_TRAJ_PATH'],
            output_dir=CONFIG['OUTPUT_DIR']
        )
    else:
        print("Skipping trajectory evaluation as requested.")

    # --- Final Visualizations ---
    if not args.no_3d_vis:
        run_3d_map_visualization(final_map, global_poses)
    else:
        print("Skipping 3D map visualization as requested.")

    if not args.no_2d_anim:
        run_2d_animation_visualization(
            csv_input_dir=CONFIG['CSV_INPUT_DIR'],
            estimated_traj_path=ESTIMATED_TRAJ_FILENAME,
            save_video=CONFIG['SAVE_ANIMATION_VIDEO'],
            video_filename=ANIMATION_VIDEO_FILENAME
        )
    else:
        print("Skipping 2D animation visualization as requested.")

    # --- 2D Occupancy Grid Mapping ---
    if not args.no_grid_map:
        run_2d_grid_mapping(
            csv_input_dir=CONFIG['CSV_INPUT_DIR'],
            estimated_traj_path=ESTIMATED_TRAJ_FILENAME,
            output_dir=CONFIG['OUTPUT_DIR']
        )
    else:
        print("Skipping 2D occupancy grid mapping as requested.")

    print("\n>>> End of Unified SLAM Script <<<")

if __name__ == "__main__":
    main()
