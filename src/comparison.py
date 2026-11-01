import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def run_evaluation():
    """
    Main function to execute the trajectory evaluation pipeline.
    """
    # ==========================================================================
    # --- 1. CONFIGURATION: Set file paths and parameters here ---
    # ==========================================================================
    print("--- Initializing Configuration ---")

    estimated_traj_path = '/home/etss/Desktop/SLAM/10-06-25/slam_results/estimated_trajectory_icp.txt'
    ground_truth_traj_path = '/home/etss/Desktop/SLAM/10-06-25/campus_train0_gt.txt'

    num_poses_to_compare = 8300

    output_dir = "evaluation_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Number of poses to compare set to: {num_poses_to_compare}")
    print("-" * 60 + "\n")

    # ==========================================================================
    # --- 2. HELPER FUNCTION: To read trajectory files ---
    # ==========================================================================
    def read_tum_trajectory(filepath, num_poses=None):
        """
        Reads a trajectory file in TUM format (timestamp tx ty tz qx qy qz qw)
        and returns timestamps and a list of 4x4 transformation matrices.
        """
        try:
            # Load the entire file first.
            data = np.loadtxt(filepath, dtype=float)

            # If a specific number of poses is requested, slice the data.
            if num_poses is not None and len(data) > num_poses:
                print(f"INFO: Reading the first {num_poses} poses from {os.path.basename(filepath)}.")
                data = data[:num_poses]
            else:
                print(f"INFO: Reading all {len(data)} poses from {os.path.basename(filepath)}.")

            timestamps = data[:, 0]
            translations = data[:, 1:4]
            quaternions = data[:, 4:8]  # Format: [qx, qy, qz, qw]

            poses = []
            for t, q in zip(translations, quaternions):
                transform_matrix = np.identity(4)
                # Use Scipy to convert quaternion to a 3x3 rotation matrix.
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

    # ==========================================================================
    # --- 3. EXECUTION: Load, Align, and Evaluate Trajectories ---
    # ==========================================================================

    # --- Check if files exist before proceeding ---
    if not os.path.exists(estimated_traj_path) or not os.path.exists(ground_truth_traj_path):
        print("🚨 ERROR: One or both trajectory files are missing. Please check the paths in the configuration.")
        return

    # --- Load data from files ---
    print("--- Loading Trajectory Files ---")
    est_timestamps, est_poses = read_tum_trajectory(estimated_traj_path, num_poses_to_compare)
    gt_timestamps, gt_poses = read_tum_trajectory(ground_truth_traj_path, num_poses_to_compare)

    # --- Validate loaded data ---
    if est_poses is None or gt_poses is None:
        print("\nCould not load trajectories. Aborting evaluation.")
        return

    if len(est_poses) != len(gt_poses):
        print(f"\n⚠️ WARNING: Pose counts do not match after loading.")
        print(f"  - Estimated poses: {len(est_poses)}")
        print(f"  - Ground truth poses: {len(gt_poses)}")
        print("  Evaluation cannot proceed with mismatched pose counts.")
        return

    if not est_poses:
        print("No poses were loaded. Cannot evaluate.")
        return

    print(f"\nSuccessfully loaded {len(est_poses)} poses from both trajectories for comparison.")

    # --- Align estimated trajectory to ground truth ---
    print("\n--- Aligning Trajectories using ICP ---")
    try:
        # Extract the translation components (x, y, z) to create point clouds
        est_points = np.array([pose[:3, 3] for pose in est_poses])
        gt_points = np.array([pose[:3, 3] for pose in gt_poses])

        # Create Open3D PointCloud objects
        est_pcd = o3d.geometry.PointCloud()
        est_pcd.points = o3d.utility.Vector3dVector(est_points)

        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(gt_points)

        # Use ICP to find the rigid body transformation to align the trajectories
        trans_init = np.identity(4)
        threshold = 1.0  # Distance threshold for ICP correspondence
        reg_p2p = o3d.pipelines.registration.registration_icp(
            est_pcd, gt_pcd, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        alignment_transform = reg_p2p.transformation
        print("Alignment complete.")

        # Apply the found transformation to the entire estimated pose history
        aligned_est_poses = [alignment_transform @ pose for pose in est_poses]

        # --- Compute Absolute Trajectory Error (ATE) ---
        print("\n--- Computing Absolute Trajectory Error (ATE) ---")
        errors = []
        for aligned_est_pose, gt_pose in zip(aligned_est_poses, gt_poses):
            error_trans = aligned_est_pose[:3, 3] - gt_pose[:3, 3]
            errors.append(np.linalg.norm(error_trans))

        ate_rmse = np.sqrt(np.mean(np.square(errors)))
        print(f"✅ ATE (RMSE): {ate_rmse:.4f} meters")

        # --- Compute Accuracy as a Percentage of Trajectory Length ---
        if len(gt_points) > 1:
            gt_start_point = gt_points[0]
            displacements = np.linalg.norm(gt_points - gt_start_point, axis=1)
            max_gt_displacement = np.max(displacements)

            if max_gt_displacement > 0:
                ate_percentage = (ate_rmse / max_gt_displacement) * 100
                accuracy_percentage = max(0, 100 - ate_percentage)
                print(f"   - ATE as % of Max GT Displacement: {ate_percentage:.2f}%")
                print(f"   - Estimated Accuracy: {accuracy_percentage:.2f}%")

        # --- Compute Relative Trajectory Error (RTE) ---
        print("\n--- Computing Relative Trajectory Error (RTE) ---")
        relative_errors = []
        for i in range(len(aligned_est_poses) - 1):
            # Calculate the pose change between two consecutive frames
            rel_est = np.linalg.inv(aligned_est_poses[i]) @ aligned_est_poses[i+1]
            rel_gt = np.linalg.inv(gt_poses[i]) @ gt_poses[i+1]

            # The error is the difference between these two relative transformations
            error_relative_transform = np.linalg.inv(rel_gt) @ rel_est
            relative_errors.append(np.linalg.norm(error_relative_transform[:3, 3]))

        if relative_errors:
            rte_rmse = np.sqrt(np.mean(np.square(relative_errors)))
            print(f"✅ RTE (RMSE over adjacent poses): {rte_rmse:.4f} meters")

        # --- Plot the results ---
        print("\n--- Plotting Trajectories ---")
        aligned_est_points = np.array([pose[:3, 3] for pose in aligned_est_poses])

        plt.figure(figsize=(12, 10))
        plt.plot(aligned_est_points[:, 0], aligned_est_points[:, 1], 'r--', label='Estimated Trajectory (Aligned)', linewidth=2)
        plt.plot(gt_points[:, 0], gt_points[:, 1], 'b-', label='Ground Truth Trajectory', linewidth=2)

        # Mark start and end points for clarity
        plt.plot(aligned_est_points[0, 0], aligned_est_points[0, 1], 'go', markersize=10, label='Start')
        plt.plot(aligned_est_points[-1, 0], aligned_est_points[-1, 1], 'rx', markersize=10, mew=2, label='End (Estimated)')
        plt.plot(gt_points[-1, 0], gt_points[-1, 1], 'bx', markersize=10, mew=2, label='End (Ground Truth)')

        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.title('Estimated vs. Ground Truth Trajectory (After Alignment)')
        plt.grid(True)
        plt.legend()
        plt.axis('equal') # This ensures the plot is not distorted

        # Save the plot to a file
        plot_filename = os.path.join(output_dir, "trajectory_comparison.png")
        plt.savefig(plot_filename)
        print(f"📈 Plot saved to '{plot_filename}'")

        # Display the plot in a window
        plt.show()

    except Exception as e:
        print(f"🚨 An error occurred during trajectory alignment or evaluation: {e}")

# ==========================================================================
# --- 4. SCRIPT ENTRY POINT ---
# ==========================================================================
if __name__ == "__main__":
    run_evaluation()
    print("\n--- Trajectory Evaluation Complete ---")