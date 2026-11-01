import os
import glob
import shutil
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

print("--- Initializing Configuration ---")


CSV_INPUT_DIR = '/home/etss/Desktop/SLAM/10-06-25/Combined_RVP_csv'
GROUND_TRUTH_TRAJ_PATH = '/home/etss/Desktop/SLAM/10-06-25/campus_train0_gt.txt'
OUTPUT_DIR = "slam_results"
ESTIMATED_TRAJ_FILENAME = os.path.join(OUTPUT_DIR, "estimated_trajectory_icp.txt")
MAP_FILENAME = os.path.join(OUTPUT_DIR, "icp_map.pcd")

os.makedirs(OUTPUT_DIR, exist_ok=True)
NUM_SCANS_TO_PROCESS = 8300
VOXEL_DOWNSAMPLE_SIZE = 0.2
MAP_VOXEL_DOWNSAMPLE_SIZE = 0.4
ICP_MAX_CORRESPONDENCE_DISTANCE = 0.8
ICP_MAX_ITERATIONS = 50
ICP_FITNESS_THRESHOLD = 0.6

print("Configuration set.")
print("-" * 60 + "\n")

  

def load_csv_as_point_cloud(csv_filepath):
    """Reads a CSV file and converts it to an Open3D PointCloud object."""
    try:
        df = pd.read_csv(csv_filepath)
        if not all(col in df.columns for col in ['x', 'y', 'z']):
            print(f"Warning: CSV {os.path.basename(csv_filepath)} is missing x, y, or z columns.")
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
        q = np.array([0, 0, 0, 1])  
    return f"{timestamp} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}"

def read_tum_trajectory(filepath, num_poses=None):
    """Reads a trajectory file in TUM format (timestamp tx ty tz qx qy qz qw)."""
    try:
        data = np.loadtxt(filepath, dtype=float, delimiter=' ')
        if num_poses is not None:
            data = data[:num_poses]

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
    except Exception as e:
        print(f"Error reading trajectory file {os.path.basename(filepath)}: {e}")
        return None, None

def run_slam_pipeline():
    """Main function to execute the entire SLAM and evaluation process."""
    print("--- Starting SLAM Pipeline ---")

    # --- Step 3.1: Load CSV Scans ---
    print("\n--- Loading CSV Scans ---")
    if not os.path.isdir(CSV_INPUT_DIR):
        print(f"⚠️ Error: CSV input directory '{CSV_INPUT_DIR}' not found.")
        return

    csv_file_paths = sorted(glob.glob(os.path.join(CSV_INPUT_DIR, "*.csv")))
    if not csv_file_paths:
        print(f"⚠️ Error: No CSV files found in '{CSV_INPUT_DIR}'.")
        return

    if NUM_SCANS_TO_PROCESS and len(csv_file_paths) > NUM_SCANS_TO_PROCESS:
        print(f"Limiting processing to the first {NUM_SCANS_TO_PROCESS} scans.")
        csv_file_paths = csv_file_paths[:NUM_SCANS_TO_PROCESS]

    all_scans_original = []
    all_timestamps = []
    print(f"Found {len(csv_file_paths)} CSV files to load.")
    for i, csv_file in enumerate(csv_file_paths):
        pcd = load_csv_as_point_cloud(csv_file)
        if pcd is not None and pcd.has_points():
            all_scans_original.append(pcd)
            # Using index as a simple timestamp placeholder
            all_timestamps.append(float(i))

    if not all_scans_original:
        print("\nNo valid Lidar scans were loaded. SLAM pipeline cannot proceed.")
        return
    print(f"\nSuccessfully loaded {len(all_scans_original)} Lidar scans.")

    # --- Step 3.2: Preprocess Scans (Downsample & Estimate Normals) ---
    print("\n--- Preprocessing Scans ---")
    all_scans_downsampled = []
    print(f"Downsampling {len(all_scans_original)} scans and estimating normals...")
    for scan in all_scans_original:
        down_pcd = scan.voxel_down_sample(VOXEL_DOWNSAMPLE_SIZE)
        # Normals are essential for the robust point-to-plane ICP variant
        down_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
        all_scans_downsampled.append(down_pcd)
    print("Preprocessing complete.")

    # --- Step 3.3: Initialize SLAM State ---
    print("\n--- Initializing SLAM State ---")
    global_poses = [np.identity(4)]
    map_accumulator = o3d.geometry.PointCloud(all_scans_original[0])
    map_accumulator.transform(global_poses[0])
    print("SLAM state and map initialized with the first scan.")

    # --- Step 3.4: ICP Odometry Main Loop ---
    print("\n--- Starting Point-to-Plane ICP Odometry Main Loop ---")
    previous_relative_transform = np.identity(4)  # Motion model prior
    estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()

    for i in range(len(all_scans_downsampled) - 1):
        target_scan_down = all_scans_downsampled[i]
        source_scan_down = all_scans_downsampled[i + 1]

        # Use the previous transformation as an initial guess for the current one
        trans_init = previous_relative_transform

        try:
            reg_icp = o3d.pipelines.registration.registration_icp(
                source=source_scan_down,
                target=target_scan_down,
                max_correspondence_distance=ICP_MAX_CORRESPONDENCE_DISTANCE,
                init=trans_init,
                estimation_method=estimation_method,
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_MAX_ITERATIONS)
            )

            # Check for successful registration
            if reg_icp.fitness > ICP_FITNESS_THRESHOLD:
                current_relative_transform = reg_icp.transformation
                previous_relative_transform = current_relative_transform  # Update motion prior
            else:
                # If registration is unreliable, assume no motion as a fallback
                current_relative_transform = np.identity(4)

        except Exception as e:
            print(f"ICP registration failed at step {i}: {e}. Assuming no motion.")
            current_relative_transform = np.identity(4)

        # Update global pose and map
        current_global_pose = global_poses[-1] @ current_relative_transform
        global_poses.append(current_global_pose)

        # Add the transformed original (not downsampled) scan to the map
        source_copy = o3d.geometry.PointCloud(all_scans_original[i + 1])
        source_copy.transform(current_global_pose)
        map_accumulator += source_copy

        # Periodically downsample the map to prevent it from becoming too dense
        if (i + 1) % 30 == 0:
            map_accumulator = map_accumulator.voxel_down_sample(MAP_VOXEL_DOWNSAMPLE_SIZE)

        if (i + 1) % 25 == 0 or (i + 1) == len(all_scans_downsampled) - 1:
            print(f"Processed scan {i+1}/{len(all_scans_downsampled)-1}. Fitness: {reg_icp.fitness:.4f}, RMSE: {reg_icp.inlier_rmse:.4f}")

    # Final downsampling of the map
    map_accumulator = map_accumulator.voxel_down_sample(MAP_VOXEL_DOWNSAMPLE_SIZE)
    print("\nLidar Odometry processing finished.")

    # --- Step 3.5: Save Trajectory and Map ---
    print("\n--- Saving Final Trajectory and Map ---")
    print(f"Saving trajectory to {ESTIMATED_TRAJ_FILENAME}...")
    with open(ESTIMATED_TRAJ_FILENAME, 'w') as f:
        for i, pose_matrix in enumerate(global_poses):
            timestamp = all_timestamps[i]
            f.write(matrix_to_tum_format(timestamp, pose_matrix) + '\n')

    print(f"Saving map to {MAP_FILENAME}...")
    o3d.io.write_point_cloud(MAP_FILENAME, map_accumulator)
    print(f"Results saved successfully to '{OUTPUT_DIR}' directory.")
    print("-" * 60 + "\n")

    return global_poses, map_accumulator

# ==============================================================================
# --- 4. EVALUATION ---
# ==============================================================================

def run_evaluation():
    """Compares the estimated trajectory with the ground truth."""
    print("--- Starting Trajectory Evaluation ---")

    # Check if trajectory files exist
    if not os.path.exists(ESTIMATED_TRAJ_FILENAME):
        print(f"⚠️ Error: Estimated trajectory not found at '{ESTIMATED_TRAJ_FILENAME}'. Cannot evaluate.")
        return
    if not os.path.exists(GROUND_TRUTH_TRAJ_PATH):
        print(f"⚠️ Error: Ground truth trajectory not found at '{GROUND_TRUTH_TRAJ_PATH}'. Cannot evaluate.")
        return

    # Read trajectory data
    num_poses = len(open(ESTIMATED_TRAJ_FILENAME).readlines())
    print(f"Reading {num_poses} poses from estimated and ground truth trajectories.")
    est_timestamps, est_poses = read_tum_trajectory(ESTIMATED_TRAJ_FILENAME)
    gt_timestamps, gt_poses = read_tum_trajectory(GROUND_TRUTH_TRAJ_PATH, num_poses=num_poses)

    if est_poses is None or gt_poses is None or len(est_poses) != len(gt_poses):
        print("Could not load trajectories or pose counts do not match. Aborting evaluation.")
        return

    # --- Step 4.1: Align Trajectories (Horn's method via ICP) ---
    print("\nAligning estimated trajectory to ground truth...")
    est_points = np.array([pose[:3, 3] for pose in est_poses])
    gt_points = np.array([pose[:3, 3] for pose in gt_poses])

    est_pcd = o3d.geometry.PointCloud()
    est_pcd.points = o3d.utility.Vector3dVector(est_points)
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_points)

    # Use ICP to find the rigid body transformation (rotation, translation)
    # that best aligns the estimated trajectory to the ground truth.
    reg_p2p = o3d.pipelines.registration.registration_icp(
        est_pcd, gt_pcd, 1.0, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    alignment_transform = reg_p2p.transformation
    aligned_est_poses = [alignment_transform @ pose for pose in est_poses]
    print("Alignment complete.")

    # --- Step 4.2: Compute Absolute Trajectory Error (ATE) ---
    print("\nComputing Absolute Trajectory Error (ATE)...")
    errors = []
    for aligned_est_pose, gt_pose in zip(aligned_est_poses, gt_poses):
        error_trans = aligned_est_pose[:3, 3] - gt_pose[:3, 3]
        errors.append(np.linalg.norm(error_trans))

    ate_rmse = np.sqrt(np.mean(np.square(errors)))
    print(f"ATE (RMSE): {ate_rmse:.4f} meters")

    # --- Step 4.3: Compute Relative Trajectory Error (RTE) ---
    print("\nComputing Relative Trajectory Error (RTE)...")
    relative_errors = []
    for i in range(len(aligned_est_poses) - 1):
        rel_est = np.linalg.inv(aligned_est_poses[i]) @ aligned_est_poses[i+1]
        rel_gt = np.linalg.inv(gt_poses[i]) @ gt_poses[i+1]
        error_relative_transform = np.linalg.inv(rel_gt) @ rel_est
        relative_errors.append(np.linalg.norm(error_relative_transform[:3, 3]))

    if relative_errors:
        rte_rmse = np.sqrt(np.mean(np.square(relative_errors)))
        print(f"RTE (RMSE over adjacent poses): {rte_rmse:.4f} meters")

    # --- Step 4.4: Plot Trajectories ---
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

    plot_filename = os.path.join(OUTPUT_DIR, "trajectory_comparison_plot.png")
    plt.savefig(plot_filename)
    print(f"Trajectory plot saved to {plot_filename}")
    plt.show()

    print("\n--- Trajectory Evaluation Complete ---")
    print("-" * 60 + "\n")


# ==============================================================================
# --- 5. MAIN EXECUTION BLOCK ---
# ==============================================================================

if __name__ == "__main__":
    # --- Run the SLAM Pipeline ---
    global_poses, final_map = run_slam_pipeline()

    if final_map:
        # --- Run Evaluation against Ground Truth ---
        run_evaluation()

        # --- Final Visualization ---
        print("--- Displaying Final Generated Map ---")
        print("A new window will open with the 3D map. Close it to end the script.")

        # Create a coordinate frame for the origin to show orientation
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

        try:
            o3d.visualization.draw_geometries(
                [final_map, origin_frame],
                window_name="Final ICP SLAM Map",
                width=1280,
                height=720
            )
        except Exception as e:
            print(f"Could not open 3D visualization window: {e}")
            print("The final map has been saved to 'icp_map.pcd' for local viewing with a PCD viewer.")

    print("\n>>> End of SLAM and Evaluation Script <<<")